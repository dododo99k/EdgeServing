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
  - "ours"
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
import threading
import time
import queue
from itertools import count
import os
import pickle
import json

import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from early_exit_resnets import build_early_exit_resnets

# ========================
# Lyapunov-related globals for 'ours' scheduler
# ========================

import os  # add this at the top of the file if not present

# Virtual SLO queue per model: model_name -> float
VIRTUAL_SLO_QUEUE = {}

# Weights for SLO penalty and accuracy loss in algorithm_ours.
# Allow overrides via environment variables for grid search.
W_SLO = float(os.getenv("W_SLO", "1.5"))
W_ACC = float(os.getenv("W_ACC", "0.1"))

# Target average SLO penalty per batch used in virtual queue update
SLO_PENALTY_TARGET = float(os.getenv("SLO_PENALTY_TARGET", "0.1"))

# Weights for SLO penalty and accuracy loss in algorithm_ours_normalized.
# Allow overrides via environment variables for grid search.
# first version 1 0.3 1.5
# second  0.1 1.2 1.3]
W_SLO_N = float(os.getenv("W_SLO_N", "1"))
W_ACC_N = float(os.getenv("W_ACC_N", "0.3"))

# Target average SLO penalty per batch used in virtual queue update
SLO_PENALTY_TARGET_N = float(os.getenv("SLO_PENALTY_TARGET_N", "1.5"))
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
):
    """
    Generate tasks according to a Poisson process for a single model.
    lam: intensity (requests per second).
    Inter-arrival times ~ Exp(rate=lam), mean = 1/lam.
    """
    while not stop_event.is_set():
        if lam > 0:
            dt = np.random.exponential(scale=1.0 / lam)
        else:
            dt = 1.0
        time.sleep(dt)

        x = torch.randn(input_shape, dtype=torch.float32)
        task = InferenceTask(model_name=model_name, input_tensor=x)
        req_queue.put(task)

        print(
            f"[Generator-{model_name}] New task {task.task_id} at "
            f"{task.arrival_time:.6f}"
        )


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


def build_accuracy_penalty(exit_points):
    """
    Simple per-exit accuracy penalty (per request).
    Larger for shallower exits (more accuracy loss).
    """
    base = {
        "layer1": 3.0,
        "layer2": 2.0,
        "layer3": 1.0,
        "final": 0.0,
    }
    return {e: base.get(e, 0.0) for e in exit_points}


def compute_avg_slo_penalty(
    wait_ms_list,
    infer_ms: float,
    slo_ms: float,
) -> float:
    """
    Average SLO penalty over a list of requests, where for each
    request i we approximate its TOTAL latency as:

        total_i = wait_i + infer_ms

    and the penalty is

        y_i = -log((slo_ms - total_i) / slo_ms),

    with a small epsilon when slack <= 0.

    This makes the penalty depend on:
      - the current waits, and
      - the batch's inference time (which depends on exit and B).
    """
    if not wait_ms_list:
        return 0.0

    if infer_ms is None:
        infer_ms = 0.0  # graceful fallback

    eps_ms = 1e-3
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


def algorithm_early_exit(
    models: dict,
    queues: dict,
    max_batch_size_by_model: dict,
    exit_points,
    latency_threshold_ms: float,
    quantile_key: str,
    profile_results_by_model: dict,
    warmup_tasks: int,
    dropped_wait_times_by_model: dict,
):
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

    # Pick the model with the longest queue
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

    # Drop tasks whose waiting time already exceeds SLO
    kept_tasks = []
    for t in batch_tasks:
        wait_ms = (now - t.arrival_time) * 1000.0
        if wait_ms > latency_threshold_ms:
            if t.task_id >= warmup_tasks:
                dropped_wait_times_by_model[model_name].append(wait_ms / 1000.0)
            print(
                f"[Scheduler-{model_name}][early_exit] DROPPED Task {t.task_id} "
                f"due to wait={wait_ms:.2f} ms > SLO={latency_threshold_ms:.2f} ms"
            )
        else:
            kept_tasks.append(t)

    if not kept_tasks:
        return None

    batch_tasks = kept_tasks
    batch_size = len(batch_tasks)

    # Max waiting time among kept tasks
    wait_times_sec = [now - t.arrival_time for t in batch_tasks]
    max_wait_ms = max(wait_times_sec) * 1000.0

    # Choose exit based on profile SLO rule
    exit_id, pred_infer_ms, pred_total_ms = choose_exit_id_from_profile(
        batch_size=batch_size,
        max_wait_ms=max_wait_ms,
        exit_points=exit_points,
        profile_results=profile_results,
        latency_threshold_ms=latency_threshold_ms,
        quantile_key=quantile_key,
    )

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
    dropped_wait_times_by_model: dict,
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

    # Drop tasks whose waiting time already exceeds SLO
    kept_tasks = []
    for t in batch_tasks:
        wait_ms = (now - t.arrival_time) * 1000.0
        if wait_ms > latency_threshold_ms:
            if t.task_id >= warmup_tasks:
                dropped_wait_times_by_model[model_name].append(wait_ms / 1000.0)
            print(
                f"[Scheduler-{model_name}][all_early] DROPPED Task {t.task_id} "
                f"due to wait={wait_ms:.2f} ms > SLO={latency_threshold_ms:.2f} ms"
            )
        else:
            kept_tasks.append(t)

    if not kept_tasks:
        return None

    batch_tasks = kept_tasks
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
    dropped_wait_times_by_model: dict,
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

    # Drop tasks whose waiting time already exceeds SLO
    kept_tasks = []
    for t in batch_tasks:
        wait_ms = (now - t.arrival_time) * 1000.0
        if wait_ms > latency_threshold_ms:
            if t.task_id >= warmup_tasks:
                dropped_wait_times_by_model[model_name].append(wait_ms / 1000.0)
            print(
                f"[Scheduler-{model_name}][all_final] DROPPED Task {t.task_id} "
                f"due to wait={wait_ms:.2f} ms > SLO={latency_threshold_ms:.2f} ms"
            )
        else:
            kept_tasks.append(t)

    if not kept_tasks:
        return None

    batch_tasks = kept_tasks
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
    dropped_wait_times_by_model: dict,
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

    # Drop tasks whose waiting time already exceeds SLO
    kept_tasks = []
    for t in batch_tasks:
        wait_ms = (now - t.arrival_time) * 1000.0
        if wait_ms > latency_threshold_ms:
            if t.task_id >= warmup_tasks:
                dropped_wait_times_by_model[model_name].append(wait_ms / 1000.0)
            print(
                f"[Scheduler-{model_name}][all_final] DROPPED Task {t.task_id} "
                f"due to wait={wait_ms:.2f} ms > SLO={latency_threshold_ms:.2f} ms"
            )
        else:
            kept_tasks.append(t)

    if not kept_tasks:
        return None

    batch_tasks = kept_tasks
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
    dropped_wait_times_by_model: dict,
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
    # Pull tasks until batch is full; drop over-SLO tasks and keep filling.
    while len(batch_tasks) < max_batch:
        try:
            t = q.get_nowait()
        except _queue.Empty:
            break

        wait_ms = (time.perf_counter() - t.arrival_time) * 1000.0
        if wait_ms > latency_threshold_ms:
            if t.task_id >= warmup_tasks:
                dropped_wait_times_by_model[model_name].append(wait_ms / 1000.0)
            print(
                f"[Scheduler-{model_name}][symphony] DROPPED Task {t.task_id} "
                f"due to wait={wait_ms:.2f} ms > SLO={latency_threshold_ms:.2f} ms"
            )
            continue

        batch_tasks.append(t)

    if not batch_tasks:
        return None

    now2 = time.perf_counter()
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


def algorithm_ours(
    models: dict,
    queues: dict,
    max_batch_size_by_model: dict,
    exit_points,
    latency_threshold_ms: float,
    quantile_key: str,
    profile_results_by_model: dict,
    warmup_tasks: int,
    dropped_wait_times_by_model: dict,
):
    """
    Our Lyapunov-style scheduler:

      - Maintains a virtual SLO queue Z[m] (global VIRTUAL_SLO_QUEUE).
      - At each decision epoch, searches over actions (m, B, e) where:
            m: model name
            B: batch size (1..max_batch_size_by_model[m], capped by queue size)
            e: exit point in exit_points

      - For each candidate action, compute:
            Q_m       = current backlog size for model m
            avg_slo   = average SLO penalty over the first B tasks
                        using wait-based penalty:
                            y_i = -log((SLO - wait_i)/SLO)
            acc_pen(e)= accuracy loss penalty per request for exit e

        Score(m, B, e) = Q_m * B
                         - W_SLO * Z[m] * avg_slo
                         - W_ACC * B * acc_pen(e)

      - Picks the (m, B, e) with maximum score.
      - If no candidate is found (e.g., queues empty), falls back
        to algorithm_all_early.
      - When a best action is found, actually pops B tasks from the
        chosen model queue, drops tasks whose wait already exceeds SLO,
        and returns the resulting batch and exit.

    Virtual queue Z[m] is updated in scheduler after the batch
    actually runs, based on the *actual* SLO penalties observed.
    """
    global VIRTUAL_SLO_QUEUE

    # Initialize virtual queues lazily
    for m in models.keys():
        VIRTUAL_SLO_QUEUE.setdefault(m, 0.0)

    model_names = list(models.keys())

    # Identify models that currently have pending work
    non_empty_models = [m for m in model_names if not queues[m].empty()]
    if not non_empty_models:
        return None

    acc_penalty = build_accuracy_penalty(exit_points)
    now = time.perf_counter()

    best_score = -float("inf")
    best_model = None
    best_B = None
    best_exit = None
    best_pred_infer_ms = None

    scores = {}
    
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

        ##### note that we use the max_B directly, without optimization #####
        
        # Waits for first B tasks (in ms)
        waits_ms = [ (now - pending_tasks[i].arrival_time) * 1000.0 for i in range(max_B) ]

        for e in exit_points:
            infer_ms = get_profile_latency_ms(
                profile_results_by_model[m],
                exit_id=e,
                batch_size=max_B,
                quantile_key=quantile_key,
            )
            if infer_ms is None: continue

            # SLO penalty now depends on BOTH waits and infer_ms(exit, B)
            avg_slo_pen = compute_avg_slo_penalty(
                wait_ms_list=waits_ms,
                infer_ms=infer_ms,
                slo_ms=latency_threshold_ms,
            )

            Zm = VIRTUAL_SLO_QUEUE[m]
            acc_pen = acc_penalty[e]

            score = (
                Qm * max_B
                - W_SLO * Zm * avg_slo_pen
                - W_ACC * max_B * acc_pen
            )

            if score > best_score:
                best_score = score
                best_model = m
                best_B = max_B
                best_exit = e
                best_pred_infer_ms = infer_ms

            scores[m].append(score)

    # If no candidate found, fall back to all_early baseline
    if best_model is None:
        return algorithm_all_early(
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

    # Drop tasks whose waiting time already exceeds SLO
    kept_tasks = []
    for t in batch_tasks:
        wait_ms = (now2 - t.arrival_time) * 1000.0
        if wait_ms > latency_threshold_ms:
            if t.task_id >= warmup_tasks:
                dropped_wait_times_by_model[model_name].append(wait_ms / 1000.0)
            print(
                f"[Scheduler-{model_name}][ours] DROPPED Task {t.task_id} "
                f"due to wait={wait_ms:.2f} ms > SLO={latency_threshold_ms:.2f} ms"
            )
        else:
            kept_tasks.append(t)

    if not kept_tasks:
        return None

    batch_tasks = kept_tasks
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


def algorithm_ours_normalized(
    models: dict,
    queues: dict,
    max_batch_size_by_model: dict,
    exit_points,
    latency_threshold_ms: float,
    quantile_key: str,
    profile_results_by_model: dict,
    warmup_tasks: int,
    dropped_wait_times_by_model: dict,
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
                         ) / infer_ms(e, B)

      - The rest of the logic is identical to 'ours'.
    """
    global VIRTUAL_SLO_QUEUE

    # Initialize virtual queues lazily
    for m in models.keys():
        VIRTUAL_SLO_QUEUE.setdefault(m, 0.0)

    model_names = list(models.keys())

    # Identify models that currently have pending work
    non_empty_models = [m for m in model_names if not queues[m].empty()]
    if not non_empty_models:
        return None

    acc_penalty = build_accuracy_penalty(exit_points)
    now = time.perf_counter()

    best_score = -float("inf")
    best_model = None
    best_B = None
    best_exit = None
    best_pred_infer_ms = None

    scores = {}
    
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

        ##### note that we use the max_B directly, without optimization #####
        
        # Waits for first B tasks (in ms)
        waits_ms = [ (now - pending_tasks[i].arrival_time) * 1000.0 for i in range(max_B) ]

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

                # SLO penalty now depends on BOTH waits and infer_ms(exit, B)
                avg_slo_pen = compute_avg_slo_penalty(
                    wait_ms_list=waits_ms[:B],
                    infer_ms=infer_ms,
                    slo_ms=latency_threshold_ms,
                )

                Zm = VIRTUAL_SLO_QUEUE[m]
                acc_pen = acc_penalty[e]

                unnormalized_score = (
                    Qm * B
                    - W_SLO_N * Zm * B * avg_slo_pen
                    - W_ACC_N * B * acc_pen
                )
                
                score = unnormalized_score / infer_ms

                if score > best_score:
                    best_score = score
                    best_model = m
                    best_B = B
                    best_exit = e
                    best_pred_infer_ms = infer_ms

                scores[m].append(score)

    # If no candidate found, fall back to all_early baseline
    if best_model is None:
        return algorithm_all_early(
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

    # Drop tasks whose waiting time already exceeds SLO
    kept_tasks = []
    for t in batch_tasks:
        wait_ms = (now2 - t.arrival_time) * 1000.0
        if wait_ms > latency_threshold_ms:
            if t.task_id >= warmup_tasks:
                dropped_wait_times_by_model[model_name].append(wait_ms / 1000.0)
            print(
                f"[Scheduler-{model_name}][ours_normalized] DROPPED Task {t.task_id} "
                f"due to wait={wait_ms:.2f} ms > SLO={latency_threshold_ms:.2f} ms"
            )
        else:
            kept_tasks.append(t)

    if not kept_tasks:
        return None

    batch_tasks = kept_tasks
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
    dropped_wait_times_by_model: dict,
    batch_diag: list,
):
    """
    Central scheduler loop:

      - While not stop_event:
            - choose scheduling algorithm based on scheduler_type
                 * "early_exit"
                 * "all_final"
                 * "all_final_round_robin"
                 * "all_early"
                 * "symphony"
                 * "ours"
                 * "ours_normalized"
            - get (model_name, exit_id, batch_tasks, batch_size, ...)
            - run inference on GPU
            - record timing and diagnostics
            - update virtual SLO queue if scheduler_type == "ours" or "ours_normalized"
    """
    models_eval = {k: v.eval() for k, v in models.items()}

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
                dropped_wait_times_by_model=dropped_wait_times_by_model,
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
                dropped_wait_times_by_model=dropped_wait_times_by_model,
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
                dropped_wait_times_by_model=dropped_wait_times_by_model,
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
                dropped_wait_times_by_model=dropped_wait_times_by_model,
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
        elif scheduler_type == "ours":
            result = algorithm_ours(
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
                dropped_wait_times_by_model=dropped_wait_times_by_model,
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

        # Per-task stats and per-batch diag (non-warmup only)
        max_total_ms = 0.0
        non_warmup_count = 0
        wait_ms_list_for_slo = []

        for t in batch_tasks:
            t.end_infer_time = end_infer_time
            total_s = t.total_time
            if total_s is None: continue

            wait_s = t.wait_time
            infer_s = t.inference_time
            total_ms = total_s * 1000.0

            print(
                f"[Scheduler-{model_name}][{scheduler_type}] Task {t.task_id}, "
                f"exit={t.exit_id_used}, "
                f"wait={wait_s*1000:.2f} ms, "
                f"infer={infer_s*1000:.2f} ms, "
                f"total={total_ms:.2f} ms, "
                f"batch_size={len(batch_tasks)}"
            )

            if t.task_id >= warmup_tasks:
                non_warmup_count += 1
                total_time_by_model[model_name].append(total_s)
                total_time_by_model_exit[model_name][exit_id].append(total_s)
                wait_time_by_model_exit[model_name][exit_id].append(wait_s)
                infer_time_by_model_exit[model_name][exit_id].append(infer_s)
                wait_ms_list_for_slo.append(wait_s * 1000.0)
                if total_ms > max_total_ms:
                    max_total_ms = total_ms

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

            if scheduler_type in ["ours", "ours_normalized"] and non_warmup_count > 0 and wait_ms_list_for_slo:
                # Use actual per-batch inference latency (ms) in the penalty
                avg_pen_actual = compute_avg_slo_penalty(
                    wait_ms_list=wait_ms_list_for_slo,
                    infer_ms=actual_infer_ms,          # measured from this batch
                    slo_ms=latency_threshold_ms,
                )

                Z_old = VIRTUAL_SLO_QUEUE.get(model_name, 0.0)
                if scheduler_type == "ours":
                    Z_new = max(
                        Z_old + (avg_pen_actual - SLO_PENALTY_TARGET),
                        0.0,
                    )
                else: # ours_normalized
                    Z_new = max(
                        Z_old + (avg_pen_actual - SLO_PENALTY_TARGET_N),
                        0.0,
                    )
                VIRTUAL_SLO_QUEUE[model_name] = Z_new

                print(
                    f"[Scheduler-{model_name}][{scheduler_type}] Z_old={Z_old:.4f}, "
                    f"avg_slo_pen_actual={avg_pen_actual:.4f}, "
                    f"Z_new={Z_new:.4f}"
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
    depth_map = {e: i + 1 for i, e in enumerate(exit_points)}
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
        print("No early-exit data to plot (avg early-exit).")
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
    print(f"Average early-exit figure saved to: {out_path}")
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
    print(f"CDF figure saved to: {out_path}")
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
            print(f"{model_name}: No data for wait/infer breakdown.")
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
        print(f"Wait/infer breakdown figure saved to: {out_path}")
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
        print("No batch diagnostics to plot.")
        return

    pred_total = []
    actual_total = []

    for bd in batch_diag:
        if bd["pred_total_ms"] is None:
            continue
        pred_total.append(bd["pred_total_ms"])
        actual_total.append(bd["actual_max_total_ms"])

    if len(pred_total) == 0:
        print("No valid pred/actual data for plot.")
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
    print(f"Predicted-vs-actual figure saved to: {out_path}")
    plt.close(fig)


# ========================
# Main setup
# ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ours",
        choices=["early_exit", "all_early", "all_final", "all_final_round_robin", "symphony", "ours", "ours_normalized"],
        help="Scheduling policy: 'early_exit', 'all_final' (no early-exit), 'all_final_round_robin' (round-robin), 'all_early' (earliest exit, longest-queue-first), 'symphony' (deferred batching), 'ours' (Lyapunov-based), or 'ours_normalized' (time-normalized score).",
    )
    parser.add_argument(
        "--lambda-50", dest="lam50", type=float, default=60.0,
        help="Poisson arrival rate for ResNet50 (req/s).",
    )
    parser.add_argument(
        "--lambda-101", dest="lam101", type=float, default=40.0,
        help="Poisson arrival rate for ResNet101 (req/s).",
    )
    parser.add_argument(
        "--lambda-152", dest="lam152", type=float, default=20.0,
        help="Poisson arrival rate for ResNet152 (req/s).",
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
        default=66666,
        help="Seed for NumPy RNG (Poisson arrivals).",
    )
    args = parser.parse_args()

    scheduler_type = args.scheduler
    lam50 = args.lam50
    lam101 = args.lam101
    lam152 = args.lam152
    run_seconds = args.run_seconds
    profile_dir = args.profile_dir
    latency_threshold_ms = args.slo_ms
    quantile_key = args.slo_quantile
    warmup_tasks = args.warmup_tasks
    output_tag = args.output_tag
    
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Scheduler: {scheduler_type}, w_slo={W_SLO}, w_acc={W_ACC}, slo_target={SLO_PENALTY_TARGET}")
    print(f"SLO (total latency): {latency_threshold_ms} ms on {quantile_key}")
    print(f"Warmup tasks (per model): {warmup_tasks}")

    exit_points = ("layer1", "layer2", "layer3", "final")

    # Build early-exit ResNets
    r50, r101, r152 = build_early_exit_resnets(device, exit_points=exit_points)
    models = {
        "ResNet50": r50,
        "ResNet101": r101,
        "ResNet152": r152,
    }

    # Load profiles
    profile_payload_50 = load_profile("ResNet50", profile_dir)
    profile_payload_101 = load_profile("ResNet101", profile_dir)
    profile_payload_152 = load_profile("ResNet152", profile_dir)

    profile_results_by_model = {
        "ResNet50": profile_payload_50["results"],
        "ResNet101": profile_payload_101["results"],
        "ResNet152": profile_payload_152["results"],
    }

    # Queues
    q_r50 = queue.Queue()
    q_r101 = queue.Queue()
    q_r152 = queue.Queue()
    queues = {
        "ResNet50": q_r50,
        "ResNet101": q_r101,
        "ResNet152": q_r152,
    }

    # Per-model max batch sizes
    max_batch_size_by_model = {
        "ResNet50": 10,
        "ResNet101": 10,
        "ResNet152": 10,
    }

    # Stats containers
    total_time_by_model = {m: [] for m in models.keys()}
    total_time_by_model_exit = {m: {e: [] for e in exit_points} for m in models.keys()}
    wait_time_by_model_exit = {m: {e: [] for e in exit_points} for m in models.keys()}
    infer_time_by_model_exit = {m: {e: [] for e in exit_points} for m in models.keys()}
    dropped_wait_times_by_model = {m: [] for m in models.keys()}
    batch_diag = []

    # Control
    stop_event = threading.Event()

    # Generators
    gen_threads = [
        threading.Thread(
            target=poisson_request_generator,
            args=("ResNet50", lam50, q_r50, stop_event),
            daemon=True,
        ),
        threading.Thread(
            target=poisson_request_generator,
            args=("ResNet101", lam101, q_r101, stop_event),
            daemon=True,
        ),
        threading.Thread(
            target=poisson_request_generator,
            args=("ResNet152", lam152, q_r152, stop_event),
            daemon=True,
        ),
    ]
    for t in gen_threads:
        t.start()

    # Central scheduler thread
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
            dropped_wait_times_by_model,
            batch_diag,
        ),
        daemon=True,
    )
    sched_thread.start()

    # Run simulation
    print(f"Running simulation for {run_seconds} seconds...")
    time.sleep(run_seconds)

    # Stop everything
    stop_event.set()
    time.sleep(2)

    # Summaries
    def summarize_overall(name, total_times):
        if not total_times:
            print(f"{name}: no completed tasks.")
            return
        arr = np.array(total_times)
        print(
            f"{name} overall total_time: count={len(arr)}, "
            f"mean={arr.mean()*1000:.2f} ms, "
            f"p50={np.percentile(arr, 50)*1000:.2f} ms, "
            f"p90={np.percentile(arr, 90)*1000:.2f} ms, "
            f"p95={np.percentile(arr, 95)*1000:.2f} ms"
        )

    def summarize_by_exit(name, total_by_exit_dict):
        for exit_id, times in total_by_exit_dict.items():
            if not times:
                print(f"{name} exit={exit_id}: no data.")
                continue
            arr = np.array(times)
            print(
                f"{name} exit={exit_id} total_time: count={len(arr)}, "
                f"mean={arr.mean()*1000:.2f} ms, "
                f"p50={np.percentile(arr, 50)*1000:.2f} ms, "
                f"p90={np.percentile(arr, 90)*1000:.2f} ms, "
                f"p95={np.percentile(arr, 95)*1000:.2f} ms"
            )

    def summarize_drops(name, dropped_waits_sec):
        if not dropped_waits_sec:
            print(f"{name}: no dropped tasks.")
            return
        arr = np.array(dropped_waits_sec) * 1000.0
        print(
            f"{name} dropped tasks: count={len(arr)}, "
            f"min_wait={arr.min():.2f} ms, "
            f"mean_wait={arr.mean():.2f} ms, "
            f"p50_wait={np.percentile(arr, 50):.2f} ms, "
            f"p95_wait={np.percentile(arr, 95):.2f} ms"
        )

    print(f"\n=== Multi-model stats [{scheduler_type}] ===")
    for m in models.keys():
        summarize_overall(m, total_time_by_model[m])
        summarize_by_exit(m, total_time_by_model_exit[m])
        summarize_drops(m, dropped_wait_times_by_model[m])
    # Average early-exit depth (weighted by completed, non-warmup tasks)
    avg_stats, _depth_map = compute_avg_early_exit_stats(total_time_by_model_exit, exit_points)
    max_depth = len(exit_points)

    print(f"\n=== Average early-exit depth [{scheduler_type}] ===")
    for model_name in ["ResNet50", "ResNet101", "ResNet152", "ALL_MODELS"]:
        if model_name not in avg_stats:
            continue

        s = avg_stats[model_name]
        if s["avg_depth"] is None:
            print(f"{model_name}: avg_exit: no data.")
            continue

        norm_depth = (s["avg_depth"] - 1.0) / (max_depth - 1.0) if max_depth > 1 else 0.0
        counts_str = ", ".join([f"{e}={s['counts'].get(e, 0)}" for e in exit_points])
        print(
            f"{model_name} avg_exit: {s['avg_depth']:.2f}/{max_depth} "
            f"(norm_depth={norm_depth:.3f}), total={s['total']}, counts: {counts_str}"
        )

    figures_dir = os.path.join("figures", f"lam152_{lam152:g}")
    logs_dir = os.path.join("logs", f"lam152_{lam152:g}")
    if output_tag:
        figures_dir = os.path.join(figures_dir, output_tag)
        logs_dir = os.path.join(logs_dir, output_tag)

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
        "dropped_wait_times_by_model": dropped_wait_times_by_model,
        "batch_diag": batch_diag,
    }
    diag_path = os.path.join(logs_dir, f"multi_model_diag_{scheduler_type}.pkl")
    with open(diag_path, "wb") as f:
        pickle.dump(diag_payload, f)
    print(f"Diagnostic data saved to: {diag_path}")
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

    print("Done.")
    #time.sleep(0) # gpu cooldown


if __name__ == "__main__":
    main()
