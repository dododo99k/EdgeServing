"""
run_early_exit_lowest_interference_slo_ablation.py

SLO ablation study for algorithm_early_exit_lowest_interference.

This script tests the performance of our normalized scheduler with different
SLO thresholds:
  - 10ms, 15ms, 20ms, 25ms, 30ms, etc.

For each SLO value, we:
  - sweep lambda_list (traffic intensities)
  - run the serving script with scheduler="early_exit_lowest_interference"
  - collect metrics (p95 latency, violate ratio, avg exit depth)
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
    slo_ms: float,
    lam: float,
    run_seconds: int,
    slo_quantile: str,
    warmup_tasks: int,
    profile_dir: str,
    logs_dir: str,
    exit_config: list,
):
    """
    Run the multi-model serving script once for given (slo_ms, λ).
    
    Parameters
    ----------
    slo_ms : float
        SLO threshold in milliseconds
    lam : float
        Poisson arrival rate for model 152
    run_seconds : int
        Duration of experiment
    slo_quantile : str
        Profile quantile key
    warmup_tasks : int
        Number of warmup tasks to exclude
    profile_dir : str
        Directory with latency profiles
    logs_dir : str
        Directory to save diagnostics
    exit_config : list
        List of active exit points
        
    Returns
    -------
    str
        Path to the diagnostic pickle file
    """
    lam_label = f"{lam:g}"
    slo_label = f"{slo_ms:g}"
    
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
        "early_exit_lowest_interference",
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

    print(f"\n=== Running early_exit_lowest_interference @ λ={lam}, SLO={slo_ms}ms ===")
    print("Command:", " ".join(cmd))

    subprocess.run(cmd, check=True)

    # Organize logs and figures into parent directories
    import shutil

    # Move logs/ to logs_slo_ablation/lam152_XX/
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

    # Move figures/lam152_XX/ to figures_slo_ablation/lam152_XX/
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
    base_diag_name = "multi_model_diag_early_exit_lowest_interference.pkl"
    base_diag_path = os.path.join(diag_dir, base_diag_name)

    if not os.path.exists(base_diag_path):
        raise FileNotFoundError(
            f"Expected diag file not found: {base_diag_path}. "
            f"Did multi_model_dev.py run successfully?"
        )

    # Rename with SLO suffix
    diag_with_config = os.path.join(
        diag_dir,
        f"multi_model_diag_early_exit_lowest_interference_slo_{slo_label}ms.pkl",
    )
    if os.path.exists(diag_with_config):
        os.remove(diag_with_config)

    shutil.move(base_diag_path, diag_with_config)

    return diag_with_config



def compute_metrics_from_diag(diag_path: str):
    """
    Given a diag pickle produced by the serving script, compute:

      - global p95 total latency (ms) across all models and exits (non-warmup tasks),
      - global SLO violation ratio = violated / (completed + violated),
      - counts of completed and violated tasks (for reference),
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

    # Try to get SLO violations from new format, fallback to old num_violated format
    slo_violations_by_model = payload.get("slo_violations_by_model", None)

    num_violated = sum(v["violations"] for v in slo_violations_by_model.values())


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
    # num_violated is a subset of num_completed (completed tasks that exceeded SLO)
    # So total should be num_completed, not num_completed + num_violated
    violate_ratio = float(num_violated) / num_completed if num_completed > 0 else 0.0

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
        "violate_ratio": float(violate_ratio),
        "num_completed": int(num_completed),
        "num_violated": int(num_violated),
        "exit_points": list(exit_points),
        "max_exit_depth": int(max_depth),
        "avg_exit_all": float(avg_exit_all),
        "avg_exit_by_model": {k: float(v) for k, v in avg_exit_by_model.items()},
        "counts_by_model_exit": counts_by_model_exit,
    }


def main():
    parser = argparse.ArgumentParser(
        description="SLO ablation study for early_exit_lowest_interference"
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="40,80,120,160,200",
        help="Comma-separated list of Poisson arrival rates for model 152 (req/s).",
    )
    parser.add_argument(
        "--slo-ms",
        type=str,
        default="20,30,40,50,60,70",
        help="Comma-separated list of SLO thresholds in milliseconds to test (e.g., '10,15,20,25,30').",
    )
    parser.add_argument(
        "--exit-config",
        type=str,
        default="layer1,layer2,layer3,final",
        help="Comma-separated list of exit points to use.",
    )
    parser.add_argument(
        "--run-seconds",
        type=int,
        default=20,
        help="Duration of each run in seconds.",
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
        default=None,
        help="Directory where diag and experiment results are stored. Default: logs_slo<value>",
    )
    args = parser.parse_args()

    lambda_list = [float(x) for x in args.lambdas.split(",") if x.strip()]
    
    # Parse SLO values as list
    slo_list = [float(x) for x in args.slo_ms.split(",") if x.strip()]
    
    if not slo_list:
        raise ValueError("SLO list cannot be empty")
    
    # Parse exit configuration
    exit_config = [x.strip() for x in args.exit_config.split(",") if x.strip()]
    
    if not exit_config:
        raise ValueError("Exit config cannot be empty")
    
    run_seconds = args.run_seconds
    slo_quantile = args.slo_quantile
    warmup_tasks = args.warmup_tasks
    profile_dir = args.profile_dir

    print(f"\n{'='*60}")
    print(f"SLO Ablation Study for early_exit_lowest_interference")
    print(f"{'='*60}")
    print(f"SLO thresholds: {slo_list} ms")
    print(f"Exit configuration: {exit_config}")
    print(f"Lambda values: {lambda_list}")
    print(f"{'='*60}\n")

    # results[slo_ms][lam] = metrics
    results = {}

    for slo_ms in slo_list:
        print(f"\n{'='*60}")
        print(f"Testing SLO = {slo_ms} ms")
        print(f"{'='*60}")
        
        # Use logs_slo_ablation directory for all SLO experiments
        if args.logs_dir is None:
            logs_dir = "logs_slo_ablation"
        else:
            logs_dir = args.logs_dir
        
        results[slo_ms] = {}
        
        for lam in lambda_list:
            try:
                diag_path = run_one_experiment(
                    slo_ms=slo_ms,
                    lam=lam,
                    run_seconds=run_seconds,
                    slo_quantile=slo_quantile,
                    warmup_tasks=warmup_tasks,
                    profile_dir=profile_dir,
                    logs_dir=logs_dir,
                    exit_config=exit_config,
                )

                metrics = compute_metrics_from_diag(diag_path)
                results[slo_ms][lam] = metrics

                print(
                    f"[RESULT] SLO={slo_ms}ms, λ={lam}: "
                    f"p95={metrics['p95_ms']:.2f} ms, "
                    f"violate_ratio={metrics['violate_ratio']:.3f}, "
                    f"completed={metrics['num_completed']}, "
                    f"num_violated={metrics['num_violated']}, "
                    f"avg_exit_depth={metrics['avg_exit_all']:.2f}"
                )
            except Exception as e:
                print(f"[ERROR] Failed for SLO={slo_ms}ms, λ={lam}: {e}")
                continue

        # Save experiment data for this SLO
        os.makedirs(logs_dir, exist_ok=True)

        slo_label_save = f"{slo_ms:g}".replace(".", "p")
        exp_path = os.path.join(logs_dir, f"slo_{slo_label_save}ms_results.pkl")
        
        with open(exp_path, "wb") as f:
            pickle.dump(
                {
                    "lambda_list": lambda_list,
                    "slo_ms": slo_ms,
                    "exit_config": exit_config,
                    "results": results[slo_ms],
                    "slo_quantile": slo_quantile,
                    "warmup_tasks": warmup_tasks,
                },
                f,
            )
        print(f"\nExperiment data saved to: {exp_path}")
    
    # Print summary tables for all SLOs
    print(f"\n{'='*60}")
    print("SUMMARY TABLES")
    print(f"{'='*60}")
    
    for slo_ms in sorted(results.keys()):
        print(f"\nSLO: {slo_ms} ms")
        print(f"{'-'*60}")
        print(f"{'Lambda':<10} {'P95(ms)':<10} {'Violate%':<10} {'AvgExit':<10} {'Completed':<12} {'num_violated':<10}")
        print(f"{'-'*60}")
        for lam in sorted(results[slo_ms].keys()):
            m = results[slo_ms][lam]
            print(f"{lam:<10.0f} {m['p95_ms']:<10.2f} "
                  f"{m['violate_ratio']*100:<10.2f} {m['avg_exit_all']:<10.2f} "
                  f"{m['num_completed']:<12} {m['num_violated']:<10}")


if __name__ == "__main__":
    main()
    subprocess.run(["python", "plot_slo_ablation_comparison.py"], check=True)
