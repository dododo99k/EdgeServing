"""
run_ours_normalized_exit_ablation.py

Ablation study for algorithm_ours_normalized with different exit point configurations.

This script tests the performance of our normalized scheduler with different
combinations of active exit points:
  - layer1/final
  - layer2/final  
  - layer3/final
  - layer1/layer2/layer3/final (baseline)

For each exit configuration, we:
  - sweep lambda_list (traffic intensities)
  - run the serving script with scheduler="ours_normalized"
  - collect metrics (p95 latency, drop ratio, avg exit depth)
  - save results and generate comparison plots
"""

import argparse
import os
import subprocess
import pickle

import numpy as np
import matplotlib.pyplot as plt


SERVING_SCRIPT = "multi_model_dev.py"


def run_one_experiment(
    exit_config: list,
    lam: float,
    run_seconds: int,
    slo_ms: float,
    slo_quantile: str,
    warmup_tasks: int,
    profile_dir: str,
    logs_dir: str,
):
    """
    Run the multi-model serving script once for given (exit_config, λ).
    
    Parameters
    ----------
    exit_config : list
        List of active exit points, e.g., ["layer1", "final"] or ["layer2", "final"]
    lam : float
        Poisson arrival rate for model 152
    run_seconds : int
        Duration of experiment
    slo_ms : float
        SLO threshold in milliseconds
    slo_quantile : str
        Profile quantile key
    warmup_tasks : int
        Number of warmup tasks to exclude
    profile_dir : str
        Directory with latency profiles
    logs_dir : str
        Directory to save diagnostics
        
    Returns
    -------
    str
        Path to the diagnostic pickle file
    """
    lam_label = f"{lam:g}"
    exit_label = "_".join(exit_config)
    
    # multi_model_dev.py outputs to logs/lam152_x and figures/lam152_x
    # We'll rename these after running
    default_logs_dir = "logs"
    default_figures_dir = os.path.join("figures", f"lam152_{lam_label}")

    # Convert exit_config list to comma-separated string
    exit_str = ",".join(exit_config)

    cmd = [
        "python",
        SERVING_SCRIPT,
        "--scheduler",
        "ours_normalized",
        "--lambda-50",
        str(lam * 3),
        "--lambda-101",
        str(lam * 2),
        "--lambda-152",
        str(lam * 1),
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

    print(f"\n=== Running ours_normalized @ λ={lam}, exits={exit_config} ===")
    print("Command:", " ".join(cmd))

    subprocess.run(cmd, check=True)

    # Organize logs and figures into parent directories
    import shutil

    # Move logs/ to logs_exit_point_ablation/
    if os.path.exists(default_logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

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

    # Move figures/lam152_XX/ to figures_exit_point_ablation/lam152_XX/
    if os.path.exists(default_figures_dir):
        # Derive figures_dir from logs_dir
        if "logs" in logs_dir:
            figures_base_dir = logs_dir.replace("logs", "figures")
        else:
            figures_base_dir = "figures_" + logs_dir

        # Create lam152_XX subdirectory structure matching logs
        figures_lam_dir = os.path.join(figures_base_dir, f"lam152_{lam_label}")
        os.makedirs(figures_lam_dir, exist_ok=True)

        for item in os.listdir(default_figures_dir):
            src = os.path.join(default_figures_dir, item)
            dst = os.path.join(figures_lam_dir, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    shutil.rmtree(src)
                else:
                    shutil.move(src, dst)
            else:
                shutil.move(src, dst)
        shutil.rmtree(default_figures_dir)
    
    # Find the diagnostic file
    diag_dir = os.path.join(logs_dir, f"lam152_{lam_label}")
    base_diag_name = "multi_model_diag_ours_normalized.pkl"
    base_diag_path = os.path.join(diag_dir, base_diag_name)

    if not os.path.exists(base_diag_path):
        raise FileNotFoundError(
            f"Expected diag file not found: {base_diag_path}. "
            f"Did multi_model_dev.py run successfully?"
        )

    # Rename with exit config suffix
    diag_with_config = os.path.join(
        diag_dir,
        f"multi_model_diag_ours_normalized_exits_{exit_label}.pkl",
    )
    if os.path.exists(diag_with_config):
        os.remove(diag_with_config)

    shutil.move(base_diag_path, diag_with_config)

    return diag_with_config


def compute_metrics_from_diag(diag_path: str):
    """
    Given a diag pickle, compute:
      - global p95 total latency (ms)
      - global dropout ratio = dropped / (completed + dropped)
      - counts of completed and dropped
      - average early-exit depth (weighted by completed task count), per-model and global
      - distribution of exits used
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
    else:
        p95_ms = float(np.percentile(all_times * 1000.0, 95))

    num_completed = int(all_times.size)
    num_dropped = int(sum(len(v) for v in dropped_wait_times_by_model.values()))
    total = num_completed + num_dropped
    drop_ratio = float(num_dropped) / total if total > 0 else 0.0

    # Average exit depth and exit distribution
    exit_counts = {e: 0 for e in exit_points}
    total_all = 0
    sum_depth_all = 0.0

    for model_name, exit_dict in total_time_by_model_exit.items():
        for e in exit_points:
            times = exit_dict.get(e, [])
            c = int(len(times))
            exit_counts[e] += c
            total_all += c
            sum_depth_all += float(depth_map[e]) * c

    avg_exit_all = (sum_depth_all / total_all) if total_all > 0 else float("nan")

    # Convert exit counts to distribution
    exit_dist = {e: (exit_counts[e] / total_all) if total_all > 0 else 0.0 for e in exit_points}

    return {
        "p95_ms": float(p95_ms),
        "drop_ratio": float(drop_ratio),
        "num_completed": int(num_completed),
        "num_dropped": int(num_dropped),
        "exit_points": list(exit_points),
        "max_exit_depth": int(max_depth),
        "avg_exit_all": float(avg_exit_all),
        "exit_counts": exit_counts,
        "exit_dist": exit_dist,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for ours_normalized with different exit point configurations"
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="40,80,120,160,200",
        help="Comma-separated list of Poisson arrival rates for model 152 (req/s).",
    )
    parser.add_argument(
        "--exit-config",
        type=str,
        nargs="+",
        default=["layer1,layer2,layer3,final","layer1,final", "layer2,final", "layer3,final",\
                "layer2,layer3,final", "layer1,layer3,final", "layer1,layer2,final"],
        help="List of exit configurations to test. Each entry is a comma-separated string of exit points. "
             "Example: --exit-config 'layer1,final' 'layer2,final' 'layer3,final' 'layer1,layer2,layer3,final'",
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
        default=20.0,
        help="Total latency SLO in milliseconds.",
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
        help="Number of initial tasks excluded from stats.",
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
        default="logs_exit_point_ablation",
        help="Directory where diag and experiment results are stored.",
    )
    args = parser.parse_args()

    lambda_list = [float(x) for x in args.lambdas.split(",") if x.strip()]

    # Parse exit configurations (now a list of configurations)
    exit_configs = []
    for config_str in args.exit_config:
        exit_points = [x.strip() for x in config_str.split(",") if x.strip()]
        if not exit_points:
            raise ValueError(f"Exit config cannot be empty: '{config_str}'")
        exit_configs.append(exit_points)

    if not exit_configs:
        raise ValueError("At least one exit configuration must be provided")

    run_seconds = args.run_seconds
    slo_ms = args.slo_ms
    slo_quantile = args.slo_quantile
    warmup_tasks = args.warmup_tasks
    profile_dir = args.profile_dir
    logs_dir = args.logs_dir

    print(f"\n{'='*60}")
    print(f"Exit Point Ablation Study for ours_normalized")
    print(f"{'='*60}")
    print(f"Number of exit configurations to test: {len(exit_configs)}")
    print(f"Exit configurations:")
    for i, exit_config in enumerate(exit_configs, 1):
        print(f"  {i}. {','.join(exit_config)}")
    print(f"Lambda values: {lambda_list}")
    print(f"Total experiments: {len(exit_configs)} exit configs × {len(lambda_list)} lambda values = {len(exit_configs) * len(lambda_list)}")
    print(f"{'='*60}\n")

    # Loop over all exit configurations
    for exit_idx, exit_config in enumerate(exit_configs, 1):
        exit_label = "_".join(exit_config)
        print(f"\n{'='*70}")
        print(f"EXIT CONFIG {exit_idx}/{len(exit_configs)}: {','.join(exit_config)}")
        print(f"{'='*70}")

        # results[lam] = metrics
        results = {}

        for lam in lambda_list:
            try:
                diag_path = run_one_experiment(
                    exit_config=exit_config,
                    lam=lam,
                    run_seconds=run_seconds,
                    slo_ms=slo_ms,
                    slo_quantile=slo_quantile,
                    warmup_tasks=warmup_tasks,
                    profile_dir=profile_dir,
                    logs_dir=logs_dir,
                )

                metrics = compute_metrics_from_diag(diag_path)
                results[lam] = metrics

                print(
                    f"[RESULT] exits={exit_label}, λ={lam}: "
                    f"p95={metrics['p95_ms']:.2f} ms, "
                    f"drop_ratio={metrics['drop_ratio']:.3f}, "
                    f"completed={metrics['num_completed']}, "
                    f"dropped={metrics['num_dropped']}, "
                    f"avg_exit_depth={metrics['avg_exit_all']:.2f}"
                )
            except Exception as e:
                print(f"[ERROR] Failed for exits={exit_label}, λ={lam}: {e}")
                continue

        # Save experiment data for this exit configuration
        os.makedirs(logs_dir, exist_ok=True)
        exp_path = os.path.join(logs_dir, f"exit_{exit_label}_results.pkl")

        with open(exp_path, "wb") as f:
            pickle.dump(
                {
                    "lambda_list": lambda_list,
                    "exit_config": exit_config,
                    "exit_label": exit_label,
                    "results": results,
                    "slo_ms": slo_ms,
                    "slo_quantile": slo_quantile,
                    "warmup_tasks": warmup_tasks,
                },
                f,
            )
        print(f"\nExperiment data saved to: {exp_path}")

        # Print summary table for this exit configuration
        print(f"\n{'='*60}")
        print(f"SUMMARY TABLE - Exit Config: {exit_label}")
        print(f"{'='*60}")
        print(f"Exit Points: {','.join(exit_config)}")
        print(f"{'-'*60}")
        print(f"{'Lambda':<10} {'P95(ms)':<10} {'Drop%':<10} {'AvgExit':<10} {'Completed':<12} {'Dropped':<10}")
        print(f"{'-'*60}")
        for lam in sorted(results.keys()):
            m = results[lam]
            print(f"{lam:<10.0f} {m['p95_ms']:<10.2f} "
                  f"{m['drop_ratio']*100:<10.2f} {m['avg_exit_all']:<10.2f} "
                  f"{m['num_completed']:<12} {m['num_dropped']:<10}")

    print(f"\n{'='*70}")
    print(f"All experiments completed!")
    print(f"  - {len(exit_configs)} exit configurations")
    print(f"  - {len(lambda_list)} lambda values")
    print(f"  - Total: {len(exit_configs) * len(lambda_list)} experiments")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
