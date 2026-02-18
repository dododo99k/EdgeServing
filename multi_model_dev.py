"""
multi_model_dev.py

Multi-DNN (3 ResNets) serving system with:

  - Models:
        ResNet50, ResNet101, ResNet152
        all wrapped with early exits: "layer1","layer2","layer3","final"
  - 1 Poisson arrival process per model
  - 1 FIFO queue per model
  - 1 central scheduler running on CPU:
        * picks which model queue to serve
        * chooses batch size (up to per-model max_batch_size)
        * chooses early-exit point for the batch (except all_final)
  - 1 shared GPU (no CUDA streams here; we serialize batches explicitly)

Schedulers (select via --scheduler):
  - "early_exit"
      * profile-based early-exit (per model)
  - "all_final"
      * baseline, always exit="final"
  - "all_final_round_robin"
      * baseline, always exit="final", round-robin among models
  - "all_early"
      * baseline, always earliest exit, longest-queue-first
  - "symphony"
      * Symphony-style deferred batching with SLO-aware batch sizing
  - "ours_normalized"
      * our Lyapunov-style algorithm_ours:
            - uses virtual SLO queue Z[m]
            - trades off queue backlog, SLO penalty, and accuracy loss

Profiling:
  - Needs early_exit_latency_<ModelName>.pkl per model (from bench script):
        * each file contains per-exit, per-batch-size latency statistics
        * used to predict inference latency for scheduling

Diagnostics:
  - per-model / per-exit wait, infer, total times
  - dropped-task stats per model
  - batch-level predicted vs actual latency
  - latency CDFs and breakdown plots
"""

import argparse
from re import DEBUG
import threading
import time
import queue
from itertools import count
import os
import pickle
import json

import math
from math import sqrt

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from early_exit_resnets import build_early_exit_resnets

# ========================
# Async Logging System (High-Performance)
# ========================
import logging
import logging.handlers
import atexit

class AsyncFileLogger:
    """
    High-performance async logger that avoids GIL contention.
    Main thread quickly pushes to queue, background thread writes to file.
    """
    def __init__(self, log_file='simulation_debug.log'):
        # Create log queue (unlimited size)
        self.log_queue = queue.Queue(-1)

        # Configure file handler with timestamp
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s.%(msecs)03d - %(message)s',
                            datefmt='%H:%M:%S')
        )

        # Create queue listener (runs in background thread)
        self.queue_listener = logging.handlers.QueueListener(
            self.log_queue, file_handler, respect_handler_level=True
        )
        self.queue_listener.start()

        # Configure logger
        self.logger = logging.getLogger('DEBUG_LOGGER')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.handlers.QueueHandler(self.log_queue))

        # Register cleanup on exit
        atexit.register(self.close)

    def log(self, msg):
        """Fast non-blocking log (just pushes to queue)"""
        self.logger.debug(msg)

    def close(self):
        """Stop the background listener"""
        self.queue_listener.stop()

# Initialize global logger (will be set after DEBUG is defined)
_debug_logger = None

def debug_log(msg):
    """
    Fast logging function that replaces print for DEBUG output.
    Non-blocking - just pushes to queue, background thread handles I/O.
    """
    global _debug_logger
    if _debug_logger:
        _debug_logger.log(msg)

def print_and_log(msg, end='\n'):
    """
    Print to console AND log to file (if DEBUG enabled).
    Use this for important messages that should be both visible and saved.
    """
    print(msg, end=end)
    debug_log(msg)

def init_debug_logger(lam50, lam101, lam152, scheduler_type):
    """
    Initialize the debug logger with lambda values in the filename.
    Should be called in main() after parsing arguments.

    Log file will be saved to: simulation_logs/lam152_{}_{}.log
    """
    global _debug_logger

    if not DEBUG:
        return

    # Create simulation_logs directory
    log_dir = "simulation_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename with lambda152 value
    log_filename = f"lam152_{lam152:g}_{scheduler_type}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Initialize the async logger
    _debug_logger = AsyncFileLogger(log_path)

    # Log the initialization
    debug_log(f"=== Debug Logger Initialized ===")
    debug_log(f"Lambda values: R50={lam50}, R101={lam101}, R152={lam152}")
    debug_log(f"Scheduler: {scheduler_type}")
    debug_log(f"Log file: {log_path}")
    debug_log(f"=" * 40)

# ========================
# Lyapunov-related globals for 'ours' scheduler
# ========================

import os  # add this at the top of the file if not present

DEBUG = True

# Async logger will be initialized later in main() with lambda values
# (Do NOT initialize here as we need lambda values for the filename)

# Virtual SLO queue per model: model_name -> float
VIRTUAL_SLO_QUEUE = {}

# Z limits for preventing starvation
Z_MAX = 1000.0  # Maximum Z value to prevent unbounded growth

# Weights for SLO penalty and accuracy loss in algorithm_ours.
# Allow overrides via environment variables for grid search.
W_SLO = float(os.getenv("W_SLO", "1.5"))
W_ACC = float(os.getenv("W_ACC", "0.3"))

# Target average SLO penalty per batch used in virtual queue update
SLO_PENALTY_TARGET = float(os.getenv("SLO_PENALTY_TARGET", "0.1"))

# Weights for SLO penalty and accuracy loss in algorithm_ours_normalized.

# Target average SLO penalty per batch used in virtual queue update
# 0.03 0.2 0.8
SLO_PENALTY_TARGET_N = float(os.getenv("SLO_PENALTY_TARGET_N", "0.05"))
# W_SLO_N = float(os.getenv("W_SLO_N", "0.2"))
W_SLO_N = float(os.getenv("W_SLO_N", "1"))
# W_ACC_N = float(os.getenv("W_ACC_N", "0.8"))
W_ACC_N = float(os.getenv("W_ACC_N", "3.5"))
# W_ACC_N = float(os.getenv("W_ACC_N", "0"))
# W_ACC_N = float(os.getenv("W_ACC_N", "0"))

# Penalties designed based on inference time analysis to compensate for time normalization
# Score formula: (Q*B - W_SLO_N*Z*B*slo_pen - W_ACC_N*B*acc_pen) / sqrt(infer_ms)
# Using sqrt normalization to reduce speed advantage of early exits (3x -> 1.73x)
# Penalties use sqrt(1 - t_i/t_final) compression to create smooth ratios
#

# Inference times (batch=5): ResNet50: 1.65/2.79/4.20/5.06ms
#                            ResNet101: 1.66/2.79/8.32/9.24ms   
#                            ResNet152: 1.67/3.69/12.61/13.53ms

# Calculated penalties: ResNet50:  0.205/0.168/0.103/0.0
#                       ResNet101: 0.227/0.209/0.079/0.0
#                       ResNet152: 0.234/0.213/0.065/0.0
# Penalty ratios: ResNet50: 2.0:1.6:1.0:0 (vs old 7.5:3.1:1:0)
#                 ResNet101: 2.9:2.6:1.0:0 (vs old 15:8.75:1:0)
#                 ResNet152: 3.6:3.3:1.0:0 (vs old 20:10.7:1:0)
# ACCURACY_PENALTIES_N = {
#     "ResNet50":  {"layer1": 0.4*2, "layer2": 0.2, "layer3": 0.03, "final": 0.0},
#     "ResNet101": {"layer1": 0.4*2, "layer2": 0.3, "layer3": 0.03, "final": 0.0},
#     "ResNet152": {"layer1": 0.5*2, "layer2": 0.35, "layer3": 0.03, "final": 0.0},
# }
# use postive formulation, means get how much accuracy we can get for each exit and model
# ACCURACY_PENALTIES = {
#     "ResNet50":  {"layer1": -0.1, "layer2": -2.5, "layer3": -4, "final": -4.5},
#     "ResNet101": {"layer1": -0.1, "layer2": -3.5, "layer3": -6.6, "final": -6.8},
#     "ResNet152": {"layer1": -0.2, "layer2": -6, "layer3": -9.9, "final": -10},
# }
# r50_coeff = 3
# r101_coeff = 1.2
# r152_coeff = 0.8

# r50_coeff = 2.5
# r101_coeff = 1.6
# r152_coeff = 1.4
# r50_coeff = 2.5
# r101_coeff = 1.6
# r152_coeff = 2
# r50_coeff = 2.5
# r101_coeff = 1.8
# r152_coeff = 2.2

# ACCURACY_PENALTIES = {
#     "ResNet50":  {"layer1": -0.05*r50_coeff, "layer2": -2.5*r50_coeff, "layer3": -4*r50_coeff, "final": -4.5*r50_coeff},
#     "ResNet101": {"layer1": -0.1*r101_coeff, "layer2": -3.3*r101_coeff, "layer3": -6.6*r101_coeff, "final": -6.8*r101_coeff},
#     "ResNet152": {"layer1": -0.2*r152_coeff, "layer2": -6*r152_coeff, "layer3": -9.9*r152_coeff, "final": -10*r152_coeff},
# }
# 1 stage training for all exit points including final
# ACCURACY_1_STAGE = {
#     "ResNet50":  {"layer1": 0.5897, "layer2": 0.7549, "layer3": 0.8071, "final": 0.8061},
#     "ResNet101": {"layer1": 0.599, "layer2": 0.7497, "layer3": 0.8135, "final": 0.8134},
#     "ResNet152": {"layer1": 0.5836, "layer2": 0.7734, "layer3": 0.8091, "final": 0.8073},
# }
ACCURACY_2_STAGE = {
    "ResNet50":  {"layer1": 0.0764, "layer2": 0.1213, "layer3": 0.3082, "final": 0.7435},
    "ResNet101": {"layer1": 0.0739, "layer2": 0.1452, "layer3": 0.5433, "final": 0.7794},
    "ResNet152": {"layer1": 0.0729, "layer2": 0.1724, "layer3": 0.4744, "final": 0.7796},
}
# ACCURACY_2_STAGE = {
#     "ResNet50":  {"layer1": 0.0764-0.0763, "layer2": 0.1213-0.0764, "layer3": 0.3082-0.0764, "final": 0.7435-0.0764},
#     "ResNet101": {"layer1": 0.0739-0.0738, "layer2": 0.1452-0.0739, "layer3": 0.5433-0.0739, "final": 0.7794-0.0739},
#     "ResNet152": {"layer1": 0.0729-0.0728, "layer2": 0.1724-0.0729, "layer3": 0.4744-0.0729, "final": 0.7796-0.0729},
# }
# ========================
# Global task id generator
# ========================

TASK_ID_GEN = count()


# ========================
# Task structure
# ========================

class InferenceTask:
    """
    A single inference task with timing info.

    Fields:
      - task_id: global increasing id
      - model_name: which model this request targets
      - arrival_time: when the request is created
      - start_infer_time: when we start the model forward (batch launch)
      - end_infer_time: when we finish the model forward
      - exit_id_used: which early-exit was chosen
      - input_tensor: input data (C,H,W)
    """

    def __init__(self, model_name: str, input_tensor: torch.Tensor):
        self.task_id = next(TASK_ID_GEN)
        self.model_name = model_name
        self.input_tensor = input_tensor

        self.arrival_time = time.perf_counter()
        self.start_infer_time = None
        self.end_infer_time = None
        self.exit_id_used = None

    @property
    def wait_time(self):
        if self.start_infer_time is None:
            return None
        return self.start_infer_time - self.arrival_time

    @property
    def inference_time(self):
        if self.start_infer_time is None or self.end_infer_time is None:
            return None
        return self.end_infer_time - self.start_infer_time

    @property
    def total_time(self):
        if self.end_infer_time is None:
            return None
        return self.end_infer_time - self.arrival_time


# ========================
# Poisson request generator
# ========================

def poisson_request_generator(
    model_name: str,
    lam: float,
    req_queue: queue.Queue,
    stop_event: threading.Event,
    input_shape=(3, 224, 224),
    warmup_complete: threading.Event = None,
):
    """
    Generate tasks according to a Poisson process for a single model.
    lam: intensity (requests per second).
    Inter-arrival times ~ Exp(rate=lam), mean = 1/lam.
    """
    # Wait for scheduler warmup to complete before generating tasks
    if warmup_complete is not None:
        print_and_log(f"[Generator-{model_name}] Waiting for scheduler warmup...")
        warmup_complete.wait()
        print_and_log(f"[Generator-{model_name}] Warmup complete, starting generation")

    while not stop_event.is_set():
        if lam > 0:
            dt = np.random.exponential(scale=1.0 / lam)
        else:
            dt = 1.0
        time.sleep(dt)

        x = torch.randn(input_shape, dtype=torch.float32)
        task = InferenceTask(model_name=model_name, input_tensor=x)
        req_queue.put(task)

        # print(
        #     f"[Generator-{model_name}] New task {task.task_id} at "
        #     f"{task.arrival_time:.6f}"
        # )


# ========================
# Profile loading helpers
# ========================

def load_profile(model_name: str, profile_dir: str):
    fname = f"early_exit_latency_{model_name}.pkl"
    path = os.path.join(profile_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Profiling file not found: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    return payload


def get_profile_latency_ms(
    profile_results: dict,
    exit_id: str,
    batch_size: int,
    quantile_key: str = "p95_ms",
) -> float:
    """
    Lookup predicted inference latency (ms) for a given exit and batch size.
    Clamp batch_size to profiled range.
    """
    exit_dict = profile_results.get(exit_id, None)
    if not exit_dict:
        return None

    bs_keys = sorted(exit_dict.keys())
    if not bs_keys:
        return None

    eff_bs = max(bs_keys[0], min(batch_size, bs_keys[-1]))

    chosen_bs = bs_keys[0]
    for bs in bs_keys:
        if bs <= eff_bs:
            chosen_bs = bs
        else:
            break

    stats = exit_dict.get(chosen_bs, None)
    if stats is None:
        return None

    latency_ms = stats.get(quantile_key, None)
    if latency_ms is None:
        latency_ms = stats.get("mean_ms", None)

    return latency_ms


# =======================
# Profile-based exit choice
# =======================

def choose_exit_id_from_profile(
    batch_size: int,
    max_wait_ms: float,
    exit_points,
    profile_results: dict,
    latency_threshold_ms: float,
    quantile_key: str = "p95_ms",
):
    """
    Choose the deepest exit whose predicted TOTAL latency for the worst request
    in the batch is below the threshold:

        total_ms(exit) = max_wait_ms + predicted_infer_ms(exit, batch_size)

    Returns:
        (exit_id, pred_infer_ms, pred_total_ms)
    """
    for exit_id in reversed(exit_points):  # deepest -> shallowest
        infer_ms = get_profile_latency_ms(
            profile_results,
            exit_id=exit_id,
            batch_size=batch_size,
            quantile_key=quantile_key,
        )
        if infer_ms is None:
            continue

        total_ms = max_wait_ms + infer_ms
        if total_ms <= latency_threshold_ms:
            return exit_id, infer_ms, total_ms

    # If nothing satisfies SLO, fall back to shallowest exit, but still record prediction
    fallback_exit = exit_points[0]
    infer_ms = get_profile_latency_ms(
        profile_results,
        exit_id=fallback_exit,
        batch_size=batch_size,
        quantile_key=quantile_key,
    )
    total_ms = max_wait_ms + (infer_ms if infer_ms is not None else 0.0)
    return fallback_exit, infer_ms, total_ms




def compute_avg_slo_penalty(
    wait_ms_list,
    infer_ms: float,
    slo_ms: float,
    ignore_wait: bool = False,
) -> float:
    """
    Average SLO penalty over a list of requests.

    Two modes:
    1. ignore_wait=False (default):
       Used for virtual queue updates (measuring actual outcome)
       total_i = wait_i + infer_ms

    2. ignore_wait=True:
       Used for scheduling decisions (action cost, not outcome)
       total = 0 + infer_ms (same penalty for all tasks)
       Prevents penalizing models that are already backlogged.

    Penalty formula: y_i = -log((slo_ms - total_i) / slo_ms)
    with a small epsilon when slack <= 0.
    """
    if not wait_ms_list:
        return 0.0

    if infer_ms is None:
        infer_ms = 0.0  # graceful fallback

    eps_ms = 1e-3
    base_penalty = -math.log(eps_ms / slo_ms)
    
    if ignore_wait:
        # Action-cost mode: only consider inference time
        # All tasks in batch get same penalty
        ys = []
        for w in wait_ms_list:
            total = float(w) + float(infer_ms)
            slack = slo_ms - total
            if slack >0:
                ys.append(-math.log(slack / slo_ms))
            else:
                overshoot = -slack
                ys.append(base_penalty + overshoot / slo_ms * 10)  # linear penalty beyond SLO
            return float(sum(ys) / len(ys))
        # slack = slo_ms - infer_ms
        # if slack <= 0.0:
        #     slack = eps_ms
        # return float(-math.log(slack / slo_ms))
    else:
        # Outcome mode: consider wait + inference time
        # Each task gets penalty based on its total latency
        ys = []
        for w in wait_ms_list:
            total = float(w) + float(infer_ms)
            slack = slo_ms - total
            if slack <= 0.0:
                slack = eps_ms
            ys.append(-math.log(slack / slo_ms))
        return float(sum(ys) / len(ys))



# ========================
# Scheduling algorithms
# ========================

import queue as _queue
def activate_function(x: list, latency_threshold_ms: float):
    # Relu activation
    result = []
    for x_ in x:
        result.append(np.exp(x_/latency_threshold_ms - 1).clip(0, 10)) # clip to prevent overflow
    return result

def algorithm_early_exit(
    models: dict,
    queues: dict,
    max_batch_size_by_model: dict,
    exit_points,
    latency_threshold_ms: float,
    quantile_key: str,
    profile_results_by_model: dict,
    warmup_tasks: int,
):
    global DEBUG
    """
    Profile-based early-exit scheduler:

      - Pick model with longest queue.
      - Build a batch up to max_batch_size for that model.
      - Drop tasks whose waiting time already exceeds SLO.
      - Using profile, pick the deepest exit whose predicted TOTAL latency
        (max_wait + infer_with_profile) fits under SLO.
      - Returns (model_name, exit_id, batch_tasks, batch_size, max_wait_ms,
                pred_infer_ms, pred_total_ms).

    If no tasks are available, returns None.
    """
    model_names = list(models.keys())
    now = time.perf_counter()
    # Pick the model with the longest queue
    # longest-queue-first (LQF)model_name
    model_name = None
    longest_qsize = -1
    # calculate all wait times
    for m in model_names:
        qsize = queues[m].qsize()
        if qsize > longest_qsize:
            longest_qsize = qsize
            model_name = m # LQF
    
    q = queues[model_name]
    profile_results = profile_results_by_model[model_name]
    max_batch = max_batch_size_by_model[model_name]
    
    
    batch_size = min(longest_qsize, max_batch)
    # calculae the wait time of batch size task in this queue and get the max wait time
    wait_times_sec = []
    for i in range(batch_size):
        t = q.queue[i]  # Access task in queue without removing it
        wait_times_sec.append(now - t.arrival_time)
    max_wait_ms = max(wait_times_sec) * 1000.0 if wait_times_sec else 0.0

    # Choose exit based on profile SLO rule
    exit_id, pred_infer_ms, pred_total_ms = choose_exit_id_from_profile(
        batch_size=batch_size,
        max_wait_ms=max_wait_ms,
        exit_points=exit_points,
        profile_results=profile_results,
        latency_threshold_ms=latency_threshold_ms,
        quantile_key=quantile_key,
    )
    
    # Pop at least one task
    batch_tasks = []
    try:
        t0 = q.get_nowait()
        batch_tasks.append(t0)
    except _queue.Empty:
        return None

    # Fill batch up to max_batch_size
    while len(batch_tasks) < max_batch:
        try:
            t = q.get_nowait()
            batch_tasks.append(t)
        except _queue.Empty:
            break

    now = time.perf_counter()

    assert batch_size == len(batch_tasks)

    # Max waiting time among kept tasks
    wait_times_sec = [now - t.arrival_time for t in batch_tasks]
    max_wait_ms = max(wait_times_sec) * 1000.0
    return model_name, exit_id, batch_tasks, batch_size, max_wait_ms, pred_infer_ms, pred_total_ms


def algorithm_all_early(
    models: dict,
    queues: dict,
    max_batch_size_by_model: dict,
    exit_points,
    latency_threshold_ms: float,
    quantile_key: str,
    profile_results_by_model: dict,
    warmup_tasks: int,
):
    """
    All-early baseline: always use the earliest (shallowest) exit in exit_points
    (typically exit_points[0]), and always serve the longest queue first.
    """
    model_names = list(models.keys())

    # Pick model with the longest queue
    best_model = None
    best_qsize = 0
    for m in model_names:
        qsize = queues[m].qsize()
        if qsize > best_qsize:
            best_qsize = qsize
            best_model = m

    if best_model is None or best_qsize == 0:
        return None

    model_name = best_model
    q = queues[model_name]
    profile_results = profile_results_by_model[model_name]
    max_batch = max_batch_size_by_model[model_name]

    batch_tasks = []
    try:
        t0 = q.get_nowait()
        batch_tasks.append(t0)
    except _queue.Empty:
        return None

    while len(batch_tasks) < max_batch:
        try:
            t = q.get_nowait()
            batch_tasks.append(t)
        except _queue.Empty:
            break

    now = time.perf_counter()

    batch_size = len(batch_tasks)

    wait_times_sec = [now - t.arrival_time for t in batch_tasks]
    max_wait_ms = max(wait_times_sec) * 1000.0

    # Earliest exit
    exit_id = exit_points[0]
    pred_infer_ms = get_profile_latency_ms(
        profile_results,
        exit_id=exit_id,
        batch_size=batch_size,
        quantile_key=quantile_key,
    )
    pred_total_ms = max_wait_ms + (pred_infer_ms if pred_infer_ms is not None else 0.0)

    return model_name, exit_id, batch_tasks, batch_size, max_wait_ms, pred_infer_ms, pred_total_ms


def algorithm_early_exit_lowest_interference(
    models: dict,
    queues: dict,
    max_batch_size_by_model: dict,
    exit_points,
    latency_threshold_ms: float,
    quantile_key: str,
    profile_results_by_model: dict,
    warmup_tasks: int,
):
    global DEBUG
    """
    Profile-based early-exit scheduler:

      - Pick model with longest queue.
      - Build a batch up to max_batch_size for that model.
      - Drop tasks whose waiting time already exceeds SLO.
      - Using profile, pick the deepest exit whose predicted TOTAL latency
        (max_wait + infer_with_profile) fits under SLO.
      - Returns (model_name, exit_id, batch_tasks, batch_size, max_wait_ms,
                pred_infer_ms, pred_total_ms).

    If no tasks are available, returns None.
    """
    model_names = list(models.keys())
    now = time.perf_counter()
    # Pick the model with the longest queue
    # longest-queue-first (LQF)model_name
    model_name = None
    longest_qsize = -1
    # calculate all wait times
    wait_times={}
    for m in model_names:
        wait_times[m] = []
        for task in queues[m].queue:
            wait_times[m].append((now - task.arrival_time) * 1000.0)
    scores = {}
    max_wait_ms_s = {}
    for m in model_names:
        qsize = queues[m].qsize()
        if qsize > longest_qsize:
            longest_qsize = qsize
            model_name = m # LQF
        
        profile_results = profile_results_by_model[m]
        max_batch = max_batch_size_by_model[m]
        batch_size = min(qsize, max_batch)
        # Choose exit based on profile SLO rule
        # calculae the wait time of batch size task in this queue and get the max wait time
        wait_times_sec = []
        for i in range(batch_size):
            t = queues[m].queue[i]  # Access task in queue without removing it
            wait_times_sec.append(now - t.arrival_time)
        max_wait_ms = max(wait_times_sec) * 1000.0 if wait_times_sec else 0.0
        exit_id, pred_infer_ms, pred_total_ms = choose_exit_id_from_profile(
            batch_size=batch_size,
            max_wait_ms=max_wait_ms,
            exit_points=exit_points,
            profile_results=profile_results,
            latency_threshold_ms=latency_threshold_ms,
            quantile_key=quantile_key,
        )
        pred_wait_time = {}
        score = 0
        for m_ in model_names:
            if m_ == m:
                pred_wait_time[m_] = [w + pred_infer_ms for w in wait_times[m_][batch_size:]]
            else:
                pred_wait_time[m_] = [w + pred_infer_ms for w in wait_times[m_]]
            score += sum(activate_function(pred_wait_time[m_], latency_threshold_ms))
        scores[m] = score # * pred_infer_ms
        max_wait_ms_s[m] = max_wait_ms
     # choose lowest score
    best_score = float("inf")
    best_model = None
    for m in model_names:
        score_ = scores[m]
        if score_ < best_score:
            best_score = score_
            best_model = m
    model_name = best_model
    q = queues[model_name]
    max_wait_ms = max_wait_ms_s[model_name]
    profile_results = profile_results_by_model[model_name]
    max_batch = max_batch_size_by_model[model_name]
    # Choose exit based on profile SLO rule and best model
    exit_id, pred_infer_ms, pred_total_ms = choose_exit_id_from_profile(
                batch_size=max_batch,
                max_wait_ms=max_wait_ms,
                exit_points=exit_points,
                profile_results=profile_results,
                latency_threshold_ms=latency_threshold_ms,
                quantile_key=quantile_key,
            )
    # Find the model with the lowest predicted wait time
    # Pop at least one task
    batch_tasks = []
    try:
        t0 = q.get_nowait()
        batch_tasks.append(t0)
    except _queue.Empty:
        return None

    # Fill batch up to max_batch_size
    while len(batch_tasks) < max_batch:
        try:
            t = q.get_nowait()
            batch_tasks.append(t)
        except _queue.Empty:
            break

    now = time.perf_counter()

    batch_size = len(batch_tasks)

    # Max waiting time among kept tasks
    wait_times_sec = [now - t.arrival_time for t in batch_tasks]
    max_wait_ms = max(wait_times_sec) * 1000.0
    return model_name, exit_id, batch_tasks, batch_size, max_wait_ms, pred_infer_ms, pred_total_ms





def algorithm_all_early(
    models: dict,
    queues: dict,
    max_batch_size_by_model: dict,
    exit_points,
    latency_threshold_ms: float,
    quantile_key: str,
    profile_results_by_model: dict,
    warmup_tasks: int,
):
    """
    All-early baseline: always use the earliest (shallowest) exit in exit_points
    (typically exit_points[0]), and always serve the longest queue first.
    """
    model_names = list(models.keys())

    # Pick model with the longest queue
    best_model = None
    best_qsize = 0
    for m in model_names:
        qsize = queues[m].qsize()
        if qsize > best_qsize:
            best_qsize = qsize
            best_model = m

    if best_model is None or best_qsize == 0:
        return None

    model_name = best_model
    q = queues[model_name]
    profile_results = profile_results_by_model[model_name]
    max_batch = max_batch_size_by_model[model_name]

    batch_tasks = []
    try:
        t0 = q.get_nowait()
        batch_tasks.append(t0)
    except _queue.Empty:
        return None

    while len(batch_tasks) < max_batch:
        try:
            t = q.get_nowait()
            batch_tasks.append(t)
        except _queue.Empty:
            break

    now = time.perf_counter()

    batch_size = len(batch_tasks)

    wait_times_sec = [now - t.arrival_time for t in batch_tasks]
    max_wait_ms = max(wait_times_sec) * 1000.0

    # Earliest exit
    exit_id = exit_points[0]
    pred_infer_ms = get_profile_latency_ms(
        profile_results,
        exit_id=exit_id,
        batch_size=batch_size,
        quantile_key=quantile_key,
    )
    pred_total_ms = max_wait_ms + (pred_infer_ms if pred_infer_ms is not None else 0.0)

    return model_name, exit_id, batch_tasks, batch_size, max_wait_ms, pred_infer_ms, pred_total_ms


# LQF: least-queue-first baselines
# all final: always use exit="final"
def algorithm_all_final(
    models: dict,
    queues: dict,
    max_batch_size_by_model: dict,
    exit_points,
    latency_threshold_ms: float,
    quantile_key: str,
    profile_results_by_model: dict,
    warmup_tasks: int,
):
    """
    Baseline: always use exit="final" (no early-exit).
    Still applies queue dropping for tasks whose wait exceeds SLO.
    """
    model_names = list(models.keys())

    # Pick model with the longest queue
    best_model = None
    best_qsize = 0
    for m in model_names:
        qsize = queues[m].qsize()
        if qsize > best_qsize:
            best_qsize = qsize
            best_model = m

    if best_model is None or best_qsize == 0:
        return None

    model_name = best_model
    q = queues[model_name]
    profile_results = profile_results_by_model[model_name]
    max_batch = max_batch_size_by_model[model_name]

    batch_tasks = []
    try:
        t0 = q.get_nowait()
        batch_tasks.append(t0)
    except _queue.Empty:
        return None

    while len(batch_tasks) < max_batch:
        try:
            t = q.get_nowait()
            batch_tasks.append(t)
        except _queue.Empty:
            break

    now = time.perf_counter()

    batch_size = len(batch_tasks)

    wait_times_sec = [now - t.arrival_time for t in batch_tasks]
    max_wait_ms = max(wait_times_sec) * 1000.0

    # Always use "final"
    exit_id = "final"
    pred_infer_ms = get_profile_latency_ms(
        profile_results,
        exit_id=exit_id,
        batch_size=batch_size,
        quantile_key=quantile_key,
    )
    pred_total_ms = max_wait_ms + (pred_infer_ms if pred_infer_ms is not None else 0.0)

    return model_name, exit_id, batch_tasks, batch_size, max_wait_ms, pred_infer_ms, pred_total_ms

# Round-robin
# all final: always use exit="final"
def algorithm_all_final_round_robin(
    models: dict,
    queues: dict,
    max_batch_size_by_model: dict,
    exit_points,
    latency_threshold_ms: float,
    quantile_key: str,
    profile_results_by_model: dict,
    warmup_tasks: int,
):
    """
    Baseline: always use exit="final" (no early-exit).
    Still applies queue dropping for tasks whose wait exceeds SLO.
    """
    model_names = list(models.keys())
    if not model_names:
        return None
    # Initialize round-robin index
    if not hasattr(algorithm_all_final_round_robin, "rr_index"):
        algorithm_all_final_round_robin.rr_index = 0
    # Pick first non-empty queue in round-robin order
    start_idx = algorithm_all_final_round_robin.rr_index % len(model_names)
    batch_tasks = []
    model_name = None
    q = None
    for offset in range(len(model_names)):
        idx = (start_idx + offset) % len(model_names)
        candidate = model_names[idx]
        candidate_q = queues[candidate]
        try:
            t0 = candidate_q.get_nowait()
        except _queue.Empty:
            continue
        model_name = candidate
        q = candidate_q
        batch_tasks.append(t0)
        algorithm_all_final_round_robin.rr_index = (idx + 1) % len(model_names)
        break

    # # Pick model with the longest queue
    # best_model = None
    # best_qsize = 0
    # for m in model_names:
    #     qsize = queues[m].qsize()
    #     if qsize > best_qsize:
    #         best_qsize = qsize
    #         best_model = m

    # if best_model is None or best_qsize == 0:
    #     return None

    if not batch_tasks:
        return None

    profile_results = profile_results_by_model[model_name]
    max_batch = max_batch_size_by_model[model_name]

    while len(batch_tasks) < max_batch:
        try:
            t = q.get_nowait()
            batch_tasks.append(t)
        except _queue.Empty:
            break

    now = time.perf_counter()

    batch_size = len(batch_tasks)

    wait_times_sec = [now - t.arrival_time for t in batch_tasks]
    max_wait_ms = max(wait_times_sec) * 1000.0

    # Always use "final"
    exit_id = "final"
    pred_infer_ms = get_profile_latency_ms(
        profile_results,
        exit_id=exit_id,
        batch_size=batch_size,
        quantile_key=quantile_key,
    )
    pred_total_ms = max_wait_ms + (pred_infer_ms if pred_infer_ms is not None else 0.0)

    return model_name, exit_id, batch_tasks, batch_size, max_wait_ms, pred_infer_ms, pred_total_ms



# Paper
# Symphony: Optimized DNN Model Serving using Deferred Batch Scheduling

def algorithm_symphony(
    models: dict,
    queues: dict,
    max_batch_size_by_model: dict,
    exit_points,
    latency_threshold_ms: float,
    quantile_key: str,
    profile_results_by_model: dict,
    warmup_tasks: int,
    dropped_wait_times_by_model: dict = None,
):
    """
    Symphony-style deferred batching:

      - For each model, choose batch size B such that the (B+1)-th batch
        would miss the SLO for the oldest task; then dispatch B.
      - If no (B+1) would miss the SLO, defer by returning None.
      - If we are already at the max batch cap, dispatch when B itself
        is at/over its latest start.
      - When scheduling, drop tasks that already exceed SLO and then choose
        the deepest exit that meets the SLO for the resulting batch.
    """
    if dropped_wait_times_by_model is None:
        dropped_wait_times_by_model = {}
    model_names = list(models.keys())
    if not model_names:
        return None

    now = time.perf_counter()
    slo_sec = latency_threshold_ms / 1000.0
    candidates = []

    for m in model_names:
        q = queues[m]
        qsize = q.qsize()
        if qsize <= 0:
            continue

        pending_tasks = list(q.queue)
        if not pending_tasks:
            continue

        max_batch = min(len(pending_tasks), max_batch_size_by_model[m])
        if max_batch <= 0:
            continue

        chosen_B = None
        chosen_latest_start = None

        for B in range(1, max_batch + 1):
            infer_ms = get_profile_latency_ms(
                profile_results_by_model[m],
                exit_id="final",
                batch_size=B,
                quantile_key=quantile_key,
            )
            if infer_ms is None:
                continue

            earliest_arrival = min(t.arrival_time for t in pending_tasks[:B])
            latest_start = earliest_arrival + slo_sec - (infer_ms / 1000.0)

            if B < max_batch:
                infer_ms_next = get_profile_latency_ms(
                    profile_results_by_model[m],
                    exit_id="final",
                    batch_size=B + 1,
                    quantile_key=quantile_key,
                )
                if infer_ms_next is None:
                    continue

                earliest_arrival_next = min(t.arrival_time for t in pending_tasks[: B + 1])
                latest_start_next = earliest_arrival_next + slo_sec - (infer_ms_next / 1000.0)

                if latest_start_next <= now:
                    chosen_B = B
                    chosen_latest_start = latest_start_next
                    break
            else:
                if latest_start <= now:
                    chosen_B = B
                    chosen_latest_start = latest_start
                    break

        if chosen_B is None:
            continue

        candidates.append(
            {
                "model": m,
                "batch_size": chosen_B,
                "latest_start": chosen_latest_start,
                "qsize": qsize,
            }
        )

    if not candidates:
        return None

    urgent = [c for c in candidates if c["latest_start"] <= now]
    if not urgent:
        return None

    chosen = min(
        urgent,
        key=lambda c: (c["latest_start"], -c["batch_size"], -c["qsize"]),
    )

    model_name = chosen["model"]
    q = queues[model_name]
    max_batch = chosen["batch_size"]
    profile_results = profile_results_by_model[model_name]

    batch_tasks = []
    # Pull tasks until batch is full
    while len(batch_tasks) < max_batch:
        try:
            t = q.get_nowait()
        except _queue.Empty:
            break

        batch_tasks.append(t)

    if not batch_tasks:
        return None

    now2 = time.perf_counter()
    slo_sec = latency_threshold_ms / 1000.0

    # Drop tasks that have already exceeded SLO (Symphony behavior)
    kept_tasks = []
    if model_name not in dropped_wait_times_by_model:
        dropped_wait_times_by_model[model_name] = []

    for t in batch_tasks:
        wait_time_sec = now2 - t.arrival_time
        if wait_time_sec > slo_sec:
            # Task has already exceeded SLO - drop it
            dropped_wait_times_by_model[model_name].append(wait_time_sec)
        else:
            kept_tasks.append(t)

    if not kept_tasks:
        # All tasks were dropped
        return None

    batch_tasks = kept_tasks
    batch_size = len(batch_tasks)

    wait_times_sec = [now2 - t.arrival_time for t in batch_tasks]
    max_wait_ms = max(wait_times_sec) * 1000.0

    # Always use "final"
    exit_id = "final"
    pred_infer_ms = get_profile_latency_ms(
        profile_results,
        exit_id=exit_id,
        batch_size=batch_size,
        quantile_key=quantile_key,
    )
    pred_total_ms = max_wait_ms + (pred_infer_ms if pred_infer_ms is not None else 0.0)

    return model_name, exit_id, batch_tasks, batch_size, max_wait_ms, pred_infer_ms, pred_total_ms



def algorithm_ours_normalized(
    models: dict,
    queues: dict,
    max_batch_size_by_model: dict,
    exit_points,
    latency_threshold_ms: float,
    quantile_key: str,
    profile_results_by_model: dict,
    warmup_tasks: int,
):
    """
    Our Lyapunov-style scheduler, with score normalized by inference time.

      - Same as 'ours', but the score function is divided by the
        predicted inference latency for the action (m, B, e).
      - This changes the optimization objective from maximizing raw score
        per action to maximizing the *rate* of score accumulation.
      - It tends to favor shorter, more efficient batches that free up
        the GPU faster.

      - For each candidate action, compute:
            Q_m       = current backlog size for model m
            avg_slo   = average SLO penalty over the first B tasks
                        using wait-based penalty:
                            y_i = -log((SLO - wait_i)/SLO)
            acc_pen(e)= accuracy loss penalty per request for exit e

        Score(m, B, e) = ( Q_m * B
                           - W_SLO * Z[m] * avg_slo
                           - W_ACC * B * acc_pen(e)
                         ) / sqrt(infer_ms(e, B))

      - The rest of the logic is identical to 'ours'.
    """
    global VIRTUAL_SLO_QUEUE
    global DEBUG
    # Initialize virtual queues lazily
    for m in models.keys():
        VIRTUAL_SLO_QUEUE.setdefault(m, 0.0)

    model_names = list(models.keys())

    # Identify models that currently have pending work
    non_empty_models = [m for m in model_names if not queues[m].empty()]
    if not non_empty_models:
        return None
    
    model_sizes = {}
    for m in non_empty_models:
        q = queues[m]
        Qm = q.qsize()
        model_sizes[m] = Qm
    #choose max model size if model size >10
    max_model_size = max(model_sizes.values())
    if max_model_size >10:
        non_empty_models = [m for m in non_empty_models if model_sizes[m] == max_model_size]

    now = time.perf_counter()

    # before calling this function

    best_score = -float("inf")
    best_model = None
    best_B = None
    best_exit = None
    best_pred_infer_ms = None
    best_part1 = None
    best_part2 = None
    best_part3 = None

    scores = {}
    # # print Q and Z snapshots for debugging
    if DEBUG:
        debug_log(f"\n[Lyapunov State]")
        debug_log("Model          | Q (queue size) | Z (virtual SLO queue)")
        debug_log("-" * 60)
        for m in sorted(models.keys()):
            Q_m = queues[m].qsize()
            Z_m = VIRTUAL_SLO_QUEUE.get(m, 0.0)
            debug_log(f"{m:<14} | {Q_m:>14d} | {Z_m:>22.4f}")
        debug_log("-" * 60)
    # -------------------------
    # Search over (m, B, e)
    # -------------------------
    for m in non_empty_models:
        q = queues[m]
        Qm = q.qsize()
        if Qm <= 0: continue

        max_batch = max_batch_size_by_model[m]
        max_B = min(Qm, max_batch)
        if max_B <= 0: continue
        
        scores[m] = []
        # Snapshot of queue contents for *peeking* (do not modify queue yet)
        pending_tasks = list(q.queue)
        
        # Wait time for first 10 B tasks (in ms)
        waits_ms = [ (now - pending_tasks[i].arrival_time) * 1000.0 for i in range(max_B) ]
        waits_ms_t = [round(w, 3) for w in waits_ms]
        waits_ms_round = []
        for t in waits_ms:
            if t < latency_threshold_ms:
                waits_ms_round.append(1)
            else:
                waits_ms_round.append(t/latency_threshold_ms)
        #
        max_wait = max(waits_ms) if waits_ms else 0.0
        if DEBUG:
            debug_log(f"[DEBUG] Model: {m}, waits_ms: {waits_ms_t}")
        for e in exit_points:
            # B = max_B
            for B in range(1, max_B + 1):
                infer_ms = get_profile_latency_ms(
                    profile_results_by_model[m],
                    exit_id=e,
                    batch_size=B,
                    quantile_key=quantile_key,
                )
                if infer_ms is None or infer_ms <= 1e-6: continue # Avoid division by zero/small number
                # if False and infer_ms + max_wait > latency_threshold_ms: # Skip actions that cannot meet SLO even for max wait
                #     if e == exit_points[0]:
                #         pass
                #     else:
                #         continue
                
                # SLO penalty for scheduling: only consider action cost (ignore historical wait)
                # This prevents penalizing backlogged models and avoids starvation
                avg_slo_pen = compute_avg_slo_penalty(
                    wait_ms_list=waits_ms[:B],
                    infer_ms=infer_ms,
                    slo_ms=latency_threshold_ms,
                    ignore_wait=True,  # Use action-cost mode for scheduling
                )
                slo_violation_number = sum(1 for w in waits_ms[:B] if (w + infer_ms) > latency_threshold_ms)

                Zm = VIRTUAL_SLO_QUEUE[m]
                # acc_pen = ACCURACY_1_STAGE[m][e]

                acc = ACCURACY_2_STAGE[m][e]

                # time normalization factor
                # temp_time = infer_ms ** 0.4 #math.cbrt(infer_ms)
                # temp_time = infer_ms ** 0.4
                temp_time = infer_ms
                # Linear normalization (default - works with properly tuned W_ACC_N)
                # All terms normalized by inference time
                # part1 = Qm * B / temp_time
                # part1 = sum(waits_ms[:B]) / temp_time
                part1 = Qm * B * acc / temp_time
                # part1 = Qm / 10 * sum(waits_ms[:B]) / temp_time
                # part2 = - W_SLO_N * Zm * slo_violation_number + Zm * B * SLO_PENALTY_TARGET_N
                # part1 = Qm * sum(waits_ms_round[:B]) / temp_time
                part2 = (- W_SLO_N * Zm * slo_violation_number + Zm * B * SLO_PENALTY_TARGET_N)# / temp_time
                # part3 = - W_ACC_N * B * acc_pen
                # part3 = W_ACC_N * B * acc ** 0.25
                part3 = 0
                
                score = part1 + part2 + part3
                if DEBUG:
                    debug_log(f"[DEBUG] infer={infer_ms:.4f} Action: (B={B}, Exit={e}) Violation:{slo_violation_number} => Score Parts: {part1:.4f}, {part2:.4f}, {part3:.4f} => Total Score: {score:.4f}")
                if score > best_score:
                    best_part1 = part1
                    best_part2 = part2
                    best_part3 = part3
                    best_score = score
                    best_model = m
                    best_B = B
                    best_exit = e
                    best_pred_infer_ms = infer_ms

                scores[m].append(score)
    # debug
    if DEBUG:
        debug_log(f"[DEBUG] [Decision Details]: {best_model}, {best_B}, {best_exit}")
        debug_log(f"[DEBUG] [Score Details]: {best_part1}, {best_part2}, {best_part3}")
    # If no candidate found, fall back to all_early baseline
    if best_model is None:
        print_and_log("[Warning!!] No candidate found in ours_normalized; falling back to all_early")
        return algorithm_all_early(
            models=models,
            queues=queues,
            max_batch_size_by_model=max_batch_size_by_model,
            exit_points=exit_points,
            latency_threshold_ms=latency_threshold_ms,
            quantile_key=quantile_key,
            profile_results_by_model=profile_results_by_model,
            warmup_tasks=warmup_tasks,
        )

    # -------------------------
    # Build actual batch from chosen (best_model, best_B, best_exit)
    # -------------------------
    model_name = best_model
    q = queues[model_name]
    max_batch = best_B
    profile_results = profile_results_by_model[model_name]

    batch_tasks = []
    try:
        t0 = q.get_nowait()
        batch_tasks.append(t0)
    except _queue.Empty:
        return None

    while len(batch_tasks) < max_batch:
        try:
            t = q.get_nowait()
            batch_tasks.append(t)
        except _queue.Empty:
            break

    now2 = time.perf_counter()

    batch_size = len(batch_tasks)

    wait_times_sec = [now2 - t.arrival_time for t in batch_tasks]
    max_wait_ms = max(wait_times_sec) * 1000.0

    exit_id = best_exit
    # Use profile for chosen exit / batch
    pred_infer_ms = get_profile_latency_ms(
        profile_results,
        exit_id=exit_id,
        batch_size=batch_size,
        quantile_key=quantile_key,
    )
    if pred_infer_ms is None:
        pred_infer_ms = best_pred_infer_ms

    pred_total_ms = max_wait_ms + (pred_infer_ms if pred_infer_ms is not None else 0.0)

    return model_name, exit_id, batch_tasks, batch_size, max_wait_ms, pred_infer_ms, pred_total_ms



# ========================
# Central scheduler
# ========================

def scheduler(
    models: dict,
    queues: dict,
    device: torch.device,
    stop_event: threading.Event,
    scheduler_type: str,
    exit_points,
    profile_results_by_model: dict,
    latency_threshold_ms: float,
    quantile_key: str,
    max_batch_size_by_model: dict,
    warmup_tasks: int,
    # stats containers (mutated in-place)
    total_time_by_model: dict,
    total_time_by_model_exit: dict,
    wait_time_by_model_exit: dict,
    infer_time_by_model_exit: dict,
    batch_diag: list,
    dropped_wait_times_by_model: dict = None,
    warmup_complete: threading.Event = None,
):
    global DEBUG
    """
    Central scheduler loop:

      - While not stop_event:
            - choose scheduling algorithm based on scheduler_type
                 * "early_exit"
                 * "all_final"
                 * "all_final_round_robin"
                 * "all_early"
                 * "symphony"
                 * "ours_normalized"
            - get (model_name, exit_id, batch_tasks, batch_size, ...)
            - run inference on GPU
            - record timing and diagnostics
            - update virtual SLO queue if scheduler_type == "ours_normalized"
    """
    models_eval = {k: v.eval() for k, v in models.items()}

    # Initialize dropped_wait_times tracking if not provided
    if dropped_wait_times_by_model is None:
        dropped_wait_times_by_model = {}

    # Warmup GPU in scheduler thread to eliminate first-inference overhead
    print_and_log("[Scheduler] Warming up GPU in scheduler thread...")
    warmup_start = time.time()
    for model_name, model_eval in models_eval.items():
        for exit_id in exit_points:
            dummy_input = torch.randn(10, 3, 224, 224, dtype=torch.float32).to(device)
            with torch.no_grad():
                _ = model_eval(dummy_input, exit_id=exit_id)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
    warmup_time = time.time() - warmup_start
    print_and_log(f"[Scheduler] GPU warmup completed in {warmup_time:.2f}s")

    # Signal that warmup is complete so generators can start
    if warmup_complete is not None:
        warmup_complete.set()
        print_and_log("[Scheduler] Signaled warmup completion to generators")

    while not stop_event.is_set():
        # If all queues are empty, sleep briefly
        if all(q.empty() for q in queues.values()):
            time.sleep(0.001)
            continue

        # --- Choose batch according to scheduler_type ---
        if scheduler_type == "early_exit":
            result = algorithm_early_exit(
                models=models,
                queues=queues,
                max_batch_size_by_model=max_batch_size_by_model,
                exit_points=exit_points,
                latency_threshold_ms=latency_threshold_ms,
                quantile_key=quantile_key,
                profile_results_by_model=profile_results_by_model,
                warmup_tasks=warmup_tasks,
            )
        elif scheduler_type == "early_exit_lowest_interference":
            result = algorithm_early_exit_lowest_interference(
                models=models,
                queues=queues,
                max_batch_size_by_model=max_batch_size_by_model,
                exit_points=exit_points,
                latency_threshold_ms=latency_threshold_ms,
                quantile_key=quantile_key,
                profile_results_by_model=profile_results_by_model,
                warmup_tasks=warmup_tasks,
            )
        elif scheduler_type == "all_early":
            result = algorithm_all_early(
                models=models,
                queues=queues,
                max_batch_size_by_model=max_batch_size_by_model,
                exit_points=exit_points,
                latency_threshold_ms=latency_threshold_ms,
                quantile_key=quantile_key,
                profile_results_by_model=profile_results_by_model,
                warmup_tasks=warmup_tasks,
            )
        elif scheduler_type == "all_final":
            result = algorithm_all_final(
                models=models,
                queues=queues,
                max_batch_size_by_model=max_batch_size_by_model,
                exit_points=exit_points,
                latency_threshold_ms=latency_threshold_ms,
                quantile_key=quantile_key,
                profile_results_by_model=profile_results_by_model,
                warmup_tasks=warmup_tasks,
            )
        elif scheduler_type == "all_final_round_robin":
            result = algorithm_all_final_round_robin(
                models=models,
                queues=queues,
                max_batch_size_by_model=max_batch_size_by_model,
                exit_points=exit_points,
                latency_threshold_ms=latency_threshold_ms,
                quantile_key=quantile_key,
                profile_results_by_model=profile_results_by_model,
                warmup_tasks=warmup_tasks,
            )
        elif scheduler_type == "symphony":
            result = algorithm_symphony(
                models=models,
                queues=queues,
                max_batch_size_by_model=max_batch_size_by_model,
                exit_points=exit_points,
                latency_threshold_ms=latency_threshold_ms,
                quantile_key=quantile_key,
                profile_results_by_model=profile_results_by_model,
                warmup_tasks=warmup_tasks,
                dropped_wait_times_by_model=dropped_wait_times_by_model,
            )
        elif scheduler_type == "ours_normalized":
            result = algorithm_ours_normalized(
                models=models,
                queues=queues,
                max_batch_size_by_model=max_batch_size_by_model,
                exit_points=exit_points,
                latency_threshold_ms=latency_threshold_ms,
                quantile_key=quantile_key,
                profile_results_by_model=profile_results_by_model,
                warmup_tasks=warmup_tasks,
            )
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

        if result is None:
            # No work chosen this round
            time.sleep(0.0005)
            continue

        (   model_name,
            exit_id,
            batch_tasks,
            batch_size,
            max_wait_ms,
            pred_infer_ms,
            pred_total_ms,
        ) = result

        if not batch_tasks or batch_size <= 0: continue

        model = models_eval[model_name]

        # Build batch tensor
        inputs = [t.input_tensor for t in batch_tasks]
        batch_input = torch.stack(inputs, dim=0)

        # Mark start_infer_time
        start_infer_time = time.perf_counter()
        for t in batch_tasks:
            t.start_infer_time = start_infer_time
            t.exit_id_used = exit_id

        # Run inference (synchronously, single GPU)
        with torch.no_grad():
            batch_input = batch_input.to(device)
            _ = model(batch_input, exit_id=exit_id)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        end_infer_time = time.perf_counter()
        actual_infer_ms = (end_infer_time - start_infer_time) * 1000.0
        debug_log(f"[Scheduler-{model_name}] Batch of size {batch_size} at exit {exit_id} took {actual_infer_ms:.2f} ms (predicted infer ms: {pred_infer_ms:.2f} ms, predicted total ms: {pred_total_ms:.2f} ms, max wait ms: {max_wait_ms:.2f} ms)")
        if actual_infer_ms > pred_infer_ms + 10:
            debug_log(f"[Warning] Actual inference time {actual_infer_ms:.2f} ms significantly exceeds predicted {pred_infer_ms:.2f} ms for model {model_name}, exit {exit_id}, batch size {batch_size}")
        # Per-task stats and per-batch diag (non-warmup only)
        max_total_ms = 0.0
        non_warmup_count = 0
        wait_ms_list_for_slo = []




        # batch information
        # print(f"[Scheduler-{model_name}] Batch_id={len(batch_diag)}, exit={exit_id}, batch_size={batch_size}")
        slo_violations = 0
        for i,t in enumerate(batch_tasks):
            t.end_infer_time = end_infer_time
            total_s = t.total_time
            if total_s is None: continue

            wait_s = t.wait_time
            infer_s = t.inference_time
            total_ms = total_s * 1000.0
            if total_ms > latency_threshold_ms:
                slo_violations += 1
            # if i ==0:
            #     print(f"Infer={infer_s*1000:.2f} ms, Task {t.task_id}, wait={wait_s*1000:.2f} ms, total={total_ms:.2f} ms", end="")
            # else:
            #     print(
            #         f", Task {t.task_id}, "
            #         f"wait={wait_s*1000:.2f} ms, "
            #         f"total={total_ms:.2f} ms, ",end=""
            #     )

            if t.task_id >= warmup_tasks:
                non_warmup_count += 1
                total_time_by_model[model_name].append(total_s)
                total_time_by_model_exit[model_name][exit_id].append(total_s)
                wait_time_by_model_exit[model_name][exit_id].append(wait_s)
                infer_time_by_model_exit[model_name][exit_id].append(infer_s)
                wait_ms_list_for_slo.append(wait_s * 1000.0)
                if total_ms > max_total_ms:
                    max_total_ms = total_ms
        # print("")  # Newline after batch task info

        if non_warmup_count > 0:
            batch_diag.append(
                {
                    "model_name": model_name,
                    "batch_index": len(batch_diag),
                    "batch_size": batch_size,
                    "exit_id": exit_id,
                    "scheduler": scheduler_type,
                    "max_wait_ms": max_wait_ms,
                    "pred_infer_ms": pred_infer_ms,
                    "pred_total_ms": pred_total_ms,
                    "actual_infer_ms": actual_infer_ms,
                    "actual_max_total_ms": max_total_ms,
                }
            )

            if scheduler_type in [ "ours_normalized"] and non_warmup_count > 0 and wait_ms_list_for_slo:
                # Use actual per-batch inference latency (ms) in the penalty
                avg_pen_actual = compute_avg_slo_penalty(
                    wait_ms_list=wait_ms_list_for_slo,
                    infer_ms=actual_infer_ms,          # measured from this batch
                    slo_ms=latency_threshold_ms,
                )

                Z_old = VIRTUAL_SLO_QUEUE.get(model_name, 0.0)

                # Calculate decay for the entire time period since last update
                Z_new = max(min(Z_old + slo_violations - SLO_PENALTY_TARGET_N*non_warmup_count,Z_MAX),0)
                VIRTUAL_SLO_QUEUE[model_name] = Z_new

                if DEBUG:
                    debug_log(
                        f"[Scheduler-{model_name}][{scheduler_type}] Z_old={Z_old:.4f}, "
                        f"avg_slo_pen_actual={avg_pen_actual:.4f}, violations={slo_violations}, "
                        f"Z_new={Z_new:.4f} (max={Z_MAX})"
                    )



# ========================
# Plot helpers
# ========================

def ensure_dirs(figures_dir: str = "figures", logs_dir: str = "logs"):
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)


def compute_avg_early_exit_stats(total_time_by_model_exit: dict, exit_points):
    """
    Compute average early-exit depth per model (and overall).

    We interpret exit depth using the order in `exit_points`:
        exit_points[0] -> depth 1  (shallowest)
        ...
        exit_points[-1] -> depth N (deepest, usually "final")

    Returns:
        stats: dict model_name -> {"avg_depth": float|None, "counts": {exit_id: int}, "total": int}
               includes "ALL_MODELS".
        depth_map: dict exit_id -> int depth
    """
    # Use global absolute depth map for fair comparison across different exit configurations
    GLOBAL_DEPTH_MAP = {"layer1": 1, "layer2": 2, "layer3": 3, "final": 4}
    depth_map = {e: GLOBAL_DEPTH_MAP.get(e, len(GLOBAL_DEPTH_MAP) + 1) for e in exit_points}
    stats = {}

    for model_name, by_exit in total_time_by_model_exit.items():
        counts = {e: len(by_exit.get(e, [])) for e in exit_points}
        total = sum(counts.values())
        if total == 0:
            avg_depth = None
        else:
            avg_depth = sum(depth_map[e] * counts[e] for e in exit_points) / float(total)
        stats[model_name] = {"avg_depth": avg_depth, "counts": counts, "total": total}

    # Aggregate across all models
    all_counts = {e: sum(stats[m]["counts"][e] for m in stats.keys()) for e in exit_points} if stats else {e: 0 for e in exit_points}
    all_total = sum(all_counts.values())
    if all_total == 0:
        all_avg = None
    else:
        all_avg = sum(depth_map[e] * all_counts[e] for e in exit_points) / float(all_total)

    stats["ALL_MODELS"] = {"avg_depth": all_avg, "counts": all_counts, "total": all_total}
    return stats, depth_map


def plot_avg_early_exit(
    total_time_by_model_exit: dict,
    exit_points,
    scheduler_type: str,
    out_dir: str = "figures",
):
    """
    Bar plot of average early-exit depth per model + overall.
    """
    ensure_dirs(figures_dir=out_dir)

    stats, depth_map = compute_avg_early_exit_stats(total_time_by_model_exit, exit_points)

    # Prefer stable ordering for readability
    preferred = ["ResNet50", "ResNet101", "ResNet152", "ALL_MODELS"]
    labels = [m for m in preferred if m in stats]
    values = [stats[m]["avg_depth"] for m in labels]

    # Filter models with no data
    filtered_labels = []
    filtered_values = []
    for lab, val in zip(labels, values):
        if val is None:
            continue
        filtered_labels.append(lab)
        filtered_values.append(val)

    if not filtered_labels:
        print_and_log("No early-exit data to plot (avg early-exit).")
        return

    max_depth = len(exit_points)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(filtered_labels))
    bars = ax.bar(x, filtered_values)

    ax.set_xticks(x)
    ax.set_xticklabels(filtered_labels)
    ax.set_ylim(1.0, float(max_depth) + 0.05)
    ax.set_ylabel(f"Average exit depth (1=shallowest, {max_depth}=deepest)")
    ax.set_title(f"Average early-exit depth ({scheduler_type})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Annotate bars with numeric values
    for rect, val in zip(bars, filtered_values):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"multi_model_avg_early_exit_{scheduler_type}.png")
    plt.savefig(out_path, dpi=150)
    print_and_log(f"Average early-exit figure saved to: {out_path}")
    plt.close(fig)

def plot_latency_cdfs(
    total_time_by_model_exit,
    exit_points,
    scheduler_type: str,
    out_dir: str = "figures",
):
    """
    total_time_by_model_exit: dict
        { model_name: { exit_id: [total_times...] } }
    """
    ensure_dirs(figures_dir=out_dir)

    models = list(total_time_by_model_exit.keys())
    num_models = len(models)

    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4), sharey=True)

    if num_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        ax.set_title(f"{model_name} ({scheduler_type})")
        ax.set_xlabel("Total time (ms)")
        ax.set_ylabel("CDF")
        ax.grid(True, linestyle="--", alpha=0.5)

        for exit_id in exit_points:
            times = total_time_by_model_exit[model_name].get(exit_id, [])
            if not times:
                continue
            arr = np.array(times)
            arr_ms = arr * 1000.0
            arr_ms_sorted = np.sort(arr_ms)
            n = len(arr_ms_sorted)
            cdf = np.arange(1, n + 1) / n
            ax.plot(arr_ms_sorted, cdf, label=exit_id)

        ax.legend(title="Exit")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"multi_model_latency_{scheduler_type}.png")
    plt.savefig(out_path, dpi=150)
    print_and_log(f"CDF figure saved to: {out_path}")
    plt.close(fig)


def plot_wait_infer_breakdown_multi(
    wait_by_model_exit: dict,
    infer_by_model_exit: dict,
    exit_points,
    scheduler_type: str,
    out_dir: str = "figures",
):
    """
    For each model, show median wait vs inference per exit.
    """
    ensure_dirs(figures_dir=out_dir)

    models = list(wait_by_model_exit.keys())

    for model_name in models:
        med_wait_ms = []
        med_infer_ms = []
        labels = []

        for exit_id in exit_points:
            waits = wait_by_model_exit[model_name].get(exit_id, [])
            infs = infer_by_model_exit[model_name].get(exit_id, [])
            if not waits or not infs:
                continue
            labels.append(exit_id)
            med_wait_ms.append(np.median(np.array(waits) * 1000.0))
            med_infer_ms.append(np.median(np.array(infs) * 1000.0))

        if not labels:
            print_and_log(f"{model_name}: No data for wait/infer breakdown.")
            continue

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width / 2, med_wait_ms, width, label="Median wait (ms)")
        ax.bar(x + width / 2, med_infer_ms, width, label="Median infer (ms)")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{model_name}: median wait vs inference per exit ({scheduler_type})")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"multi_model_wait_infer_{model_name}_{scheduler_type}.png")
        plt.savefig(out_path, dpi=150)
        print_and_log(f"Wait/infer breakdown figure saved to: {out_path}")
        plt.close(fig)


def plot_pred_vs_actual_batches_multi(
    batch_diag: list,
    scheduler_type: str,
    out_dir: str = "figures",
):
    """
    Scatter for predicted vs actual max total latency across all models/batches.
    """
    ensure_dirs(figures_dir=out_dir)

    if not batch_diag:
        print_and_log("No batch diagnostics to plot.")
        return

    pred_total = []
    actual_total = []

    for bd in batch_diag:
        if bd["pred_total_ms"] is None:
            continue
        pred_total.append(bd["pred_total_ms"])
        actual_total.append(bd["actual_max_total_ms"])

    if len(pred_total) == 0:
        print_and_log("No valid pred/actual data for plot.")
        return

    pred_total = np.array(pred_total)
    actual_total = np.array(actual_total)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pred_total, actual_total, alpha=0.6, edgecolors="none")
    max_val = max(pred_total.max(), actual_total.max())
    ax.plot([0, max_val], [0, max_val], "k--", label="y=x")

    ax.set_xlabel("Predicted max total latency (ms)")
    ax.set_ylabel("Actual max total latency (ms)")
    ax.set_title(f"Pred vs actual (per batch, {scheduler_type})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"multi_model_pred_vs_actual_{scheduler_type}.png")
    plt.savefig(out_path, dpi=150)
    print_and_log(f"Predicted-vs-actual figure saved to: {out_path}")
    plt.close(fig)


# ========================
# Main setup
# ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheduler",
        type=str,
        default="early_exit_lowest_interference",
        choices=["early_exit", "early_exit_lowest_interference", "all_early", "all_final", "all_final_round_robin", "symphony", "ours_normalized"],
        help="Scheduling policy: 'early_exit', 'early_exit_lowest_interference', 'all_final' (no early-exit), 'all_final_round_robin' (round-robin), 'all_early' (earliest exit, longest-queue-first), 'symphony' (deferred batching) or 'ours_normalized' (time-normalized score).",
    )
    parser.add_argument(
        "--lambda-50", dest="lam50", type=float, default=300.0,
        help="Total Poisson arrival rate for all ResNet50 instances (req/s).",
    )
    parser.add_argument(
        "--lambda-101", dest="lam101", type=float, default=200.0,
        help="Total Poisson arrival rate for all ResNet101 instances (req/s).",
    )
    parser.add_argument(
        "--lambda-152", dest="lam152", type=float, default=100.0,
        help="Total Poisson arrival rate for all ResNet152 instances (req/s).",
    )
    parser.add_argument(
        "--num-r50", type=int, default=1,
        help="Number of ResNet50 model instances.",
    )
    parser.add_argument(
        "--num-r101", type=int, default=1,
        help="Number of ResNet101 model instances.",
    )
    parser.add_argument(
        "--num-r152", type=int, default=1,
        help="Number of ResNet152 model instances.",
    )
    parser.add_argument(
        "--run-seconds", type=int, default=30,
        help="Simulation duration in seconds.",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="saves",
        help="Directory containing early_exit_latency_<ModelName>.pkl.",
    )
    parser.add_argument(
        "--slo-ms",
        type=float,
        default=50.0,
        help="Total latency SLO in milliseconds (wait + inference, approximated via profile).",
    )
    parser.add_argument(
        "--slo-quantile",
        type=str,
        default="p95_ms",
        choices=["mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms"],
        help="Which profile statistic to enforce against the SLO.",
    )
    parser.add_argument(
        "--warmup-tasks",
        type=int,
        default=100,
        help="Exclude first N tasks per model from stats (but still scheduled and possibly dropped).",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Append a subfolder named by this tag to figures/logs paths.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=666666, # 66666
        help="Seed for NumPy RNG (Poisson arrivals).",
    )
    parser.add_argument(
        "--exit-points",
        type=str,
        default="layer1,layer2,layer3,final",
        help="Comma-separated list of active exit points (e.g., 'layer1,final' or 'layer2,final').",
    )
    args = parser.parse_args()

    scheduler_type = args.scheduler
    lam50 = args.lam50
    lam101 = args.lam101
    lam152 = args.lam152
    num_r50 = args.num_r50
    num_r101 = args.num_r101
    num_r152 = args.num_r152
    run_seconds = args.run_seconds
    profile_dir = args.profile_dir
    latency_threshold_ms = args.slo_ms
    quantile_key = args.slo_quantile
    warmup_tasks = args.warmup_tasks
    output_tag = args.output_tag
    
    np.random.seed(args.seed)
    
    # Parse exit points from command line
    exit_points_list = [x.strip() for x in args.exit_points.split(",") if x.strip()]
    exit_points = tuple(exit_points_list)
    
    if not exit_points:
        raise ValueError("At least one exit point must be specified.")

    # Initialize debug logger with lambda values
    init_debug_logger(lam50, lam101, lam152, scheduler_type)

    print_and_log(f"Active exit points: {exit_points}")
    print_and_log(f"Model configuration: {num_r50}×R50, {num_r101}×R101, {num_r152}×R152")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_and_log(f"Using device: {device}")
    if scheduler_type == "ours_normalized":
        print_and_log(f"Scheduler: {scheduler_type}, w_slo={W_SLO_N}, w_acc={W_ACC_N}, slo_target={SLO_PENALTY_TARGET_N}")
    else:
        print_and_log(f"Scheduler: {scheduler_type}")
    print_and_log(f"SLO (total latency): {latency_threshold_ms} ms on {quantile_key}")
    print_and_log(f"Warmup tasks (per model): {warmup_tasks}")

    # Build early-exit ResNets dynamically based on counts
    r50, r101, r152 = build_early_exit_resnets(device, exit_points=exit_points)
    
    models = {}
    queues = {}
    gen_threads = []
    
    # Create ResNet50 instances
    for i in range(num_r50):
        model_name = f"ResNet50_{i}" if num_r50 > 1 else "ResNet50"
        models[model_name] = r50  # Share the same model weights
        queues[model_name] = queue.Queue()
        # Divide lambda equally among instances
        lam_per_instance = lam50 / num_r50 if num_r50 > 0 else 0
        if lam_per_instance > 0:
            gen_threads.append(
                threading.Thread(
                    target=poisson_request_generator,
                    args=(model_name, lam_per_instance, queues[model_name], None),  # stop_event set later
                    daemon=True,
                )
            )
    
    # Create ResNet101 instances
    for i in range(num_r101):
        model_name = f"ResNet101_{i}" if num_r101 > 1 else "ResNet101"
        models[model_name] = r101
        queues[model_name] = queue.Queue()
        lam_per_instance = lam101 / num_r101 if num_r101 > 0 else 0
        if lam_per_instance > 0:
            gen_threads.append(
                threading.Thread(
                    target=poisson_request_generator,
                    args=(model_name, lam_per_instance, queues[model_name], None),
                    daemon=True,
                )
            )
    
    # Create ResNet152 instances
    for i in range(num_r152):
        model_name = f"ResNet152_{i}" if num_r152 > 1 else "ResNet152"
        models[model_name] = r152
        queues[model_name] = queue.Queue()
        lam_per_instance = lam152 / num_r152 if num_r152 > 0 else 0
        if lam_per_instance > 0:
            gen_threads.append(
                threading.Thread(
                    target=poisson_request_generator,
                    args=(model_name, lam_per_instance, queues[model_name], None),
                    daemon=True,
                )
            )

    # Load profiles (shared across instances of same architecture)
    profile_payload_50 = load_profile("ResNet50", profile_dir)
    profile_payload_101 = load_profile("ResNet101", profile_dir)
    profile_payload_152 = load_profile("ResNet152", profile_dir)

    profile_results_by_model = {}
    for model_name in models.keys():
        if "ResNet50" in model_name:
            profile_results_by_model[model_name] = profile_payload_50["results"]
        elif "ResNet101" in model_name:
            profile_results_by_model[model_name] = profile_payload_101["results"]
        elif "ResNet152" in model_name:
            profile_results_by_model[model_name] = profile_payload_152["results"]

    # Per-model max batch sizes
    max_batch_size_by_model = {m: 10 for m in models.keys()}

    # Stats containers
    total_time_by_model = {m: [] for m in models.keys()}
    total_time_by_model_exit = {m: {e: [] for e in exit_points} for m in models.keys()}
    wait_time_by_model_exit = {m: {e: [] for e in exit_points} for m in models.keys()}
    infer_time_by_model_exit = {m: {e: [] for e in exit_points} for m in models.keys()}
    batch_diag = []
    dropped_wait_times_by_model = {m: [] for m in models.keys()}

    # ========================
    # GPU Warmup (eliminate first-inference overhead)
    # ========================
    print_and_log("\nWarming up GPU...")
    warmup_start = time.time()

    for model_name, model in models.items():
        model_eval = model.eval()
        for exit_id in exit_points:
            # Create dummy input
            dummy_input = torch.randn(10, 3, 224, 224, dtype=torch.float32).to(device)

            # Run inference
            with torch.no_grad():
                _ = model_eval(dummy_input, exit_id=exit_id)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

    warmup_time = time.time() - warmup_start
    print_and_log(f"GPU warmup completed in {warmup_time:.2f}s")
    print_and_log(f"All models and exits are now ready for simulation.\n")

    # Control events
    stop_event = threading.Event()
    warmup_complete = threading.Event()  # Signal when scheduler warmup is done

    # Central scheduler thread (start FIRST so it can warmup before generators start)
    sched_thread = threading.Thread(
        target=scheduler,
        args=(
            models,
            queues,
            device,
            stop_event,
            scheduler_type,
            exit_points,
            profile_results_by_model,
            latency_threshold_ms,
            quantile_key,
            max_batch_size_by_model,
            warmup_tasks,
            total_time_by_model,
            total_time_by_model_exit,
            wait_time_by_model_exit,
            infer_time_by_model_exit,
            batch_diag,
            dropped_wait_times_by_model,
            warmup_complete,  # Pass warmup_complete event
        ),
        daemon=True,
    )
    sched_thread.start()

    # Update generator threads with stop_event and warmup_complete
    for thread in gen_threads:
        # Recreate thread args: (model_name, lam, queue, stop_event, input_shape, warmup_complete)
        old_args = thread._args
        # old_args = (model_name, lam, queue, None)
        thread._args = (old_args[0], old_args[1], old_args[2], stop_event, (3, 224, 224), warmup_complete)
    for t in gen_threads:
        t.start()

    # Run simulation
    print_and_log(f"Running simulation for {run_seconds} seconds...")
    time.sleep(run_seconds)

    # Stop everything
    stop_event.set()
    time.sleep(2)

    # Summaries
    def summarize_overall(name, total_times):
        if not total_times:
            print_and_log(f"{name}: no completed tasks.")
            return
        arr = np.array(total_times)
        msg = (
            f"{name} overall total_time: count={len(arr)}, "
            f"mean={arr.mean()*1000:.2f} ms, "
            f"p50={np.percentile(arr, 50)*1000:.2f} ms, "
            f"p90={np.percentile(arr, 90)*1000:.2f} ms, "
            f"p95={np.percentile(arr, 95)*1000:.2f} ms"
        )
        print_and_log(msg)

    def summarize_by_exit(name, total_by_exit_dict):
        for exit_id, times in total_by_exit_dict.items():
            if not times:
                print_and_log(f"{name} exit={exit_id}: no data.")
                continue
            arr = np.array(times)
            msg = (
                f"{name} exit={exit_id} total_time: count={len(arr)}, "
                f"mean={arr.mean()*1000:.2f} ms, "
                f"p50={np.percentile(arr, 50)*1000:.2f} ms, "
                f"p90={np.percentile(arr, 90)*1000:.2f} ms, "
                f"p95={np.percentile(arr, 95)*1000:.2f} ms"
            )
            print_and_log(msg)

    def summarize_slo_violations(name, total_times, slo_ms):
        if not total_times:
            print_and_log(f"{name}: no completed tasks for SLO violation check.")
            return 0, 0.0
        arr = np.array(total_times)
        arr_ms = arr * 1000.0
        violations = np.sum(arr_ms > slo_ms)
        total = len(arr_ms)
        violation_ratio = violations / total if total > 0 else 0.0
        msg = (
            f"{name} SLO violations: {violations}/{total} "
            f"({violation_ratio*100:.2f}% exceeded {slo_ms:.0f} ms SLO)"
        )
        print_and_log(msg)
        return violations, violation_ratio

    print_and_log(f"\n=== Multi-model stats [{scheduler_type}] ===")
    total_violations = 0
    total_tasks = 0
    for m in models.keys():
        summarize_overall(m, total_time_by_model[m])
        summarize_by_exit(m, total_time_by_model_exit[m])
        violations, _ = summarize_slo_violations(m, total_time_by_model[m], latency_threshold_ms)
        total_violations += violations
        total_tasks += len(total_time_by_model[m])

    # Overall SLO violation summary across all models
    if total_tasks > 0:
        overall_violation_ratio = total_violations / total_tasks
        print_and_log(f"\n=== Overall SLO violations [{scheduler_type}] ===")
        msg = (
            f"Total: {total_violations}/{total_tasks} tasks "
            f"({overall_violation_ratio*100:.2f}% exceeded {latency_threshold_ms:.0f} ms SLO)"
        )
        print_and_log(msg)

    # Dropped tasks summary (only for schedulers that drop)
    total_dropped = sum(len(dropped_wait_times_by_model[m]) for m in models.keys())
    if total_dropped > 0:
        print_and_log(f"\n=== Dropped tasks [{scheduler_type}] ===")
        for m in models.keys():
            num_dropped = len(dropped_wait_times_by_model[m])
            if num_dropped > 0:
                drop_waits_ms = np.array(dropped_wait_times_by_model[m]) * 1000.0
                msg = (
                    f"{m}: {num_dropped} dropped, "
                    f"avg_wait={drop_waits_ms.mean():.2f} ms, "
                    f"max_wait={drop_waits_ms.max():.2f} ms"
                )
                print_and_log(msg)
        num_completed = sum(len(total_time_by_model[m]) for m in models.keys())
        total_generated = total_dropped + num_completed
        drop_ratio = total_dropped / total_generated if total_generated > 0 else 0.0
        msg = (
            f"Total: {total_dropped}/{total_generated} tasks dropped "
            f"({drop_ratio*100:.2f}% drop ratio)"
        )
        print_and_log(msg)

    # Average early-exit depth (weighted by completed, non-warmup tasks)
    avg_stats, _depth_map = compute_avg_early_exit_stats(total_time_by_model_exit, exit_points)
    max_depth = len(exit_points)

    print_and_log(f"\n=== Average early-exit depth [{scheduler_type}] ===")
    for model_name in ["ResNet50", "ResNet101", "ResNet152", "ALL_MODELS"]:
        if model_name not in avg_stats:
            continue

        s = avg_stats[model_name]
        if s["avg_depth"] is None:
            print_and_log(f"{model_name}: avg_exit: no data.")
            continue

        norm_depth = (s["avg_depth"] - 1.0) / (max_depth - 1.0) if max_depth > 1 else 0.0
        counts_str = ", ".join([f"{e}={s['counts'].get(e, 0)}" for e in exit_points])
        msg = (
            f"{model_name} avg_exit: {s['avg_depth']:.2f}/{max_depth} "
            f"(norm_depth={norm_depth:.3f}), total={s['total']}, counts: {counts_str}"
        )
        print_and_log(msg)

    figures_dir = os.path.join("figures", f"lam152_{lam152:g}")
    logs_dir = os.path.join("logs", f"lam152_{lam152:g}")
    if output_tag:
        figures_dir = os.path.join(figures_dir, output_tag)
        logs_dir = os.path.join(logs_dir, output_tag)

    # Compute SLO violations for diagnostics
    slo_violations_by_model = {}
    for m in models.keys():
        if total_time_by_model[m]:
            arr = np.array(total_time_by_model[m])
            arr_ms = arr * 1000.0
            violations = int(np.sum(arr_ms > latency_threshold_ms))
            total = len(arr_ms)
            violation_ratio = violations / total if total > 0 else 0.0
            slo_violations_by_model[m] = {
                "violations": violations,
                "total": total,
                "violation_ratio": violation_ratio,
            }
        else:
            slo_violations_by_model[m] = {
                "violations": 0,
                "total": 0,
                "violation_ratio": 0.0,
            }

    # Save diagnostics
    ensure_dirs(figures_dir=figures_dir, logs_dir=logs_dir)
    diag_payload = {
        "scheduler": scheduler_type,
        "slo_ms": latency_threshold_ms,
        "slo_quantile": quantile_key,
        "exit_points": exit_points,
        "total_time_by_model": total_time_by_model,
        "total_time_by_model_exit": total_time_by_model_exit,
        "wait_time_by_model_exit": wait_time_by_model_exit,
        "infer_time_by_model_exit": infer_time_by_model_exit,
        "slo_violations_by_model": slo_violations_by_model,
        "dropped_wait_times_by_model": dropped_wait_times_by_model,
        "batch_diag": batch_diag,
    }
    diag_path = os.path.join(logs_dir, f"multi_model_diag_{scheduler_type}.pkl")
    with open(diag_path, "wb") as f:
        pickle.dump(diag_payload, f)
    print_and_log(f"Diagnostic data saved to: {diag_path}")
    # Figures
    plot_avg_early_exit(total_time_by_model_exit, exit_points, scheduler_type, out_dir=figures_dir)
    plot_latency_cdfs(total_time_by_model_exit, exit_points, scheduler_type, out_dir=figures_dir)
    plot_wait_infer_breakdown_multi(
        wait_time_by_model_exit,
        infer_time_by_model_exit,
        exit_points,
        scheduler_type,
        out_dir=figures_dir,
    )
    plot_pred_vs_actual_batches_multi(batch_diag, scheduler_type, out_dir=figures_dir)

    print_and_log("Done.")
    #time.sleep(0) # gpu cooldown


if __name__ == "__main__":
    main()
