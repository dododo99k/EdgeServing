"""
run_multi_model_intensity.py

Run multi-model serving experiments to compare:
  - p95 total latency (across all models and exits)
  - dropout ratio (dropped / (completed + dropped))

between:
  - early_exit scheduler
  - all_final scheduler

for a range of traffic intensities (Poisson arrival rates).

This script assumes:
  - multi_model.py is in the same directory,
    - that script writes diagnostics to:
        logs/lam152_<lambda-152>/multi_model_diag_<scheduler>.pkl

We:
  - sweep lambda_list = [10, 30, 50, 70, 90] (configurable)
  - for each scheduler in ["early_exit", "all_final"]
  - for each λ:
        * run the serving script
        * rename the diag file to include λ in its filename
        * load diag and compute:
              - global p95 latency (ms) across all models and exits
              - global dropout ratio
  - save experiment results to logs/multi_model_scheduler_experiment.pkl
  - plot comparison figure:
        x-axis: λ
        left y-axis: p95 total latency (ms)
        right y-axis: dropout ratio
"""

import argparse
import os
import re
import subprocess
import pickle
import shutil

import numpy as np
import matplotlib.pyplot as plt


SERVING_SCRIPT = "multi_model_dev.py"


def run_one_experiment(
    scheduler: str,
    lam: float,
    run_seconds: int,
    slo_ms: float,
    slo_quantile: str,
    warmup_tasks: int,
    profile_dir: str,
    logs_dir: str,
    exit_config: list,
):
    """
    Run the multi-model serving script once for given (scheduler, λ),
    rename the diagnostic pickle, and return its path.

    We set the same λ for all three models:
        --lambda-50 λ, --lambda-101 λ, --lambda-152 λ
    """
    lam_label = f"{lam:g}"
    
    # multi_model_dev.py outputs to logs/ and figures/
    default_logs_dir = "logs"
    default_figures_dir = "figures"
    
    # Convert exit_config list to comma-separated string
    exit_str = ",".join(exit_config)

    cmd = [
        "python",
        SERVING_SCRIPT,
        "--scheduler",
        scheduler,
        "--lambda-50",
        str(lam*3),
        "--lambda-101",
        str(lam*2),
        "--lambda-152",
        str(lam*1),
        "--run-seconds",
        str(run_seconds),
        "--slo-ms",
        str(slo_ms),
        "--slo-quantile",
        slo_quantile,
        "--warmup-tasks",
        str(warmup_tasks),
        "--profile-dir",
        profile_dir,
        "--exit-points",
        exit_str,
    ]

    print(f"\n=== Running {scheduler} @ λ={lam} ===")
    print("Command:", " ".join(cmd))

    subprocess.run(cmd, check=True)

    # Organize logs and figures into parent directories if logs_dir differs from default
    if logs_dir != default_logs_dir:
        # Derive figures_dir from logs_dir (replace "logs" with "figures")
        if "logs" in logs_dir:
            figures_dir = logs_dir.replace("logs", "figures")
        else:
            figures_dir = "figures_" + logs_dir
        
        # Move logs/ to target logs_dir
        if os.path.exists(default_logs_dir):
            if os.path.exists(logs_dir):
                # Merge directories
                for item in os.listdir(default_logs_dir):
                    src = os.path.join(default_logs_dir, item)
                    dst = os.path.join(logs_dir, item)
                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                            shutil.rmtree(src)
                        else:
                            shutil.move(src, dst)
                    else:
                        shutil.move(src, dst)
                shutil.rmtree(default_logs_dir)
            else:
                os.makedirs(os.path.dirname(logs_dir) if os.path.dirname(logs_dir) else ".", exist_ok=True)
                shutil.move(default_logs_dir, logs_dir)
        
        # Move figures/ to target figures_dir
        if os.path.exists(default_figures_dir):
            if os.path.exists(figures_dir):
                # Merge directories
                for item in os.listdir(default_figures_dir):
                    src = os.path.join(default_figures_dir, item)
                    dst = os.path.join(figures_dir, item)
                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                            shutil.rmtree(src)
                        else:
                            shutil.move(src, dst)
                    else:
                        shutil.move(src, dst)
                shutil.rmtree(default_figures_dir)
            else:
                os.makedirs(os.path.dirname(figures_dir) if os.path.dirname(figures_dir) else ".", exist_ok=True)
                shutil.move(default_figures_dir, figures_dir)
    else:
        # logs_dir is default, no moving needed
        pass

    # Find the diagnostic file
    diag_dir = os.path.join(logs_dir, f"lam152_{lam_label}")
    base_diag_name = f"multi_model_diag_{scheduler}.pkl"
    base_diag_path = os.path.join(diag_dir, base_diag_name)

    if not os.path.exists(base_diag_path):
        raise FileNotFoundError(
            f"Expected diag file not found: {base_diag_path}. "
            f"Did multi_model.py run successfully?"
        )

    # Rename to include lambda so multiple experiments don't overwrite each other
    diag_with_lam = os.path.join(
        diag_dir,
        f"multi_model_diag_{scheduler}_lam{lam_label}.pkl",
    )
    if os.path.exists(diag_with_lam):
        os.remove(diag_with_lam)
    os.rename(base_diag_path, diag_with_lam)

    return diag_with_lam



def compute_metrics_from_diag(diag_path: str):
    """
    Given a diag pickle produced by the serving script, compute:

      - global p95 total latency (ms) across all models and exits (non-warmup tasks),
      - global dropout ratio = dropped / (completed + dropped),
      - counts of completed and dropped (for reference),
      - average early-exit depth (weighted by completed task count), per-model and global.

    Notes
    -----
    * We treat each exit point as an ordinal depth in the order of `exit_points`
      (e.g., layer1=1, layer2=2, layer3=3, final=4).
    * If `exit_points` is missing from the payload, we infer an order from keys.
    """
    with open(diag_path, "rb") as f:
        payload = pickle.load(f)

    total_time_by_model_exit = payload["total_time_by_model_exit"]
    dropped_wait_times_by_model = payload["dropped_wait_times_by_model"]

    # Exit ordering (depth)
    exit_points = payload.get("exit_points")
    if not exit_points:
        # Infer a stable order from observed exit keys
        observed = set()
        for exit_dict in total_time_by_model_exit.values():
            observed.update(exit_dict.keys())

        preferred = ["layer1", "layer2", "layer3", "final"]
        exit_points = [e for e in preferred if e in observed] + sorted([e for e in observed if e not in preferred])

    # Use global absolute depth map for fair comparison across different exit configurations
    GLOBAL_DEPTH_MAP = {"layer1": 1, "layer2": 2, "layer3": 3, "final": 4}
    depth_map = {e: GLOBAL_DEPTH_MAP.get(e, len(GLOBAL_DEPTH_MAP) + 1) for e in exit_points}
    max_depth = max(depth_map.values()) if depth_map else 0

    # Flatten total_times across all models and exits (completed tasks)
    all_times = []
    for _, exit_dict in total_time_by_model_exit.items():
        for _, times in exit_dict.items():
            all_times.extend(times)

    all_times = np.array(all_times, dtype=np.float64)

    if all_times.size == 0:
        p95_ms = float("nan")
        p99_ms = float("nan")
    else:
        p95_ms = float(np.percentile(all_times * 1000.0, 95))
        p99_ms = float(np.percentile(all_times * 1000.0, 99))

    num_completed = int(all_times.size)
    num_dropped = int(sum(len(v) for v in dropped_wait_times_by_model.values()))
    total = num_completed + num_dropped
    drop_ratio = float(num_dropped) / total if total > 0 else 0.0

    # Average exit depth per model and global (completed tasks only)
    avg_exit_by_model = {}
    counts_by_model_exit = {}

    total_all = 0
    sum_depth_all = 0.0

    for model_name, exit_dict in total_time_by_model_exit.items():
        counts = {}
        total_m = 0
        sum_depth_m = 0.0

        for e in exit_points:
            times = exit_dict.get(e, [])
            c = int(len(times))
            counts[e] = c
            total_m += c
            sum_depth_m += float(depth_map[e]) * c

        counts_by_model_exit[model_name] = counts
        avg_exit_by_model[model_name] = (sum_depth_m / total_m) if total_m > 0 else float("nan")

        total_all += total_m
        sum_depth_all += sum_depth_m

    avg_exit_all = (sum_depth_all / total_all) if total_all > 0 else float("nan")

    return {
        "p95_ms": float(p95_ms),
        "p99_ms": float(p99_ms),
        "drop_ratio": float(drop_ratio),
        "num_completed": int(num_completed),
        "num_dropped": int(num_dropped),
        "exit_points": list(exit_points),
        "max_exit_depth": int(max_depth),
        "avg_exit_all": float(avg_exit_all),
        "avg_exit_by_model": {k: float(v) for k, v in avg_exit_by_model.items()},
        "counts_by_model_exit": counts_by_model_exit,
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lambdas",
        type=str,
        # default="10,20,30,40,50,60,70,80",
        default="40,80,120,160,200",
        help="Comma-separated list of Poisson arrival rates per model (req/s).",
    )
    parser.add_argument(
        "--exit-config",
        type=str,
        default="layer1,layer2,layer3,final",
        help="Comma-separated list of exit points to use, e.g., 'layer1,final' or 'layer2,final'.",
    )
    parser.add_argument(
        "--run-seconds",
        type=int,
        default=30,
        help="Duration of each run in seconds.",
    )
    parser.add_argument(
        "--slo-ms",
        type=float,
        default=15.0,
        help="Total latency SLO in milliseconds (same as serving script).",
    )
    parser.add_argument(
        "--slo-quantile",
        type=str,
        default="p95_ms",
        choices=["mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms"],
        help="Profile statistic used inside the scheduler.",
    )
    parser.add_argument(
        "--warmup-tasks",
        type=int,
        default=100,
        help="Number of initial tasks (by id) excluded from stats (serving script arg).",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="saves",
        help="Directory containing early_exit_latency_<ModelName>.pkl.",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs_baseline",
        help="Directory where diag and experiment results are stored.",
    )
    args = parser.parse_args()

    lambda_list = [float(x) for x in args.lambdas.split(",") if x.strip()]
    
    # Parse exit configuration
    exit_config = [x.strip() for x in args.exit_config.split(",") if x.strip()]
    if not exit_config:
        raise ValueError("Exit config cannot be empty")
    
    run_seconds = args.run_seconds
    slo_ms = args.slo_ms
    slo_quantile = args.slo_quantile
    warmup_tasks = args.warmup_tasks
    profile_dir = args.profile_dir
    logs_dir = args.logs_dir

    schedulers = ["early_exit", "all_early", "all_final", "all_final_round_robin", "symphony", "ours_normalized"]
    schedulers = ["early_exit", "all_early", "all_final", "all_final_round_robin", "symphony"]
    schedulers = ["all_final_round_robin","ours_normalized"]
    # results[scheduler][lam] = metrics
    results = {sch: {} for sch in schedulers}

    print(f"\n{'='*60}")
    print(f"Multi-Model Intensity Experiment")
    print(f"{'='*60}")
    print(f"Exit configuration: {exit_config}")
    print(f"Lambda values: {lambda_list}")
    print(f"Schedulers: {schedulers}")
    print(f"{'='*60}\n")

    for scheduler in schedulers:
        for lam in lambda_list:
            try:
                diag_path = run_one_experiment(
                    scheduler=scheduler,
                    lam=lam,
                    run_seconds=run_seconds,
                    slo_ms=slo_ms,
                    slo_quantile=slo_quantile,
                    warmup_tasks=warmup_tasks,
                    profile_dir=profile_dir,
                    logs_dir=logs_dir,
                    exit_config=exit_config,
                )

                metrics = compute_metrics_from_diag(diag_path)
                results[scheduler][lam] = metrics

                print(
                    f"[RESULT] scheduler={scheduler}, λ={lam}: "
                    f"p95={metrics['p95_ms']:.2f} ms, "
                    f"drop_ratio={metrics['drop_ratio']:.3f}, "
                    f"completed={metrics['num_completed']}, "
                    f"dropped={metrics['num_dropped']}"
                )
            except Exception as e:
                print(f"[ERROR] Failed for scheduler={scheduler}, λ={lam}: {e}")
                continue

    # Save experiment data
    os.makedirs(logs_dir, exist_ok=True)
    exp_path = os.path.join(logs_dir, "multi_model_scheduler_experiment.pkl")
    with open(exp_path, "wb") as f:
        pickle.dump(
            {
                "lambda_list": lambda_list,
                "exit_config": exit_config,
                "results": results,
                "slo_ms": slo_ms,
                "slo_quantile": slo_quantile,
                "warmup_tasks": warmup_tasks,
            },
            f,
        )
    print(f"\nExperiment data saved to: {exp_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    for scheduler in schedulers:
        print(f"\nScheduler: {scheduler}")
        print(f"{'-'*60}")
        print(f"{'Lambda':<10} {'P95(ms)':<10} {'Drop%':<10} {'AvgExit':<10} {'Completed':<12} {'Dropped':<10}")
        print(f"{'-'*60}")
        for lam in sorted(results[scheduler].keys()):
            m = results[scheduler][lam]
            print(f"{lam:<10.0f} {m['p95_ms']:<10.2f} "
                  f"{m['drop_ratio']*100:<10.2f} {m['avg_exit_all']:<10.2f} "
                  f"{m['num_completed']:<12} {m['num_dropped']:<10}")
    


if __name__ == "__main__":
    main()
