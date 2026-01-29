"""
run_ours_normalized_quantile_study.py

Quantile study for algorithm_ours_normalized.

This script tests the performance of our normalized scheduler with different
quantile_key settings:
  - mean_ms
  - p50_ms
  - p90_ms
  - p95_ms
  - p99_ms

For each quantile, we:
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
    num_r50: int,
    num_r101: int,
    num_r152: int,
    lam50: float,
    lam101: float,
    lam152: float,
    run_seconds: int,
    slo_ms: float,
    slo_quantile: str,
    warmup_tasks: int,
    profile_dir: str,
    logs_dir: str,
    exit_config: list,
):
    """
    Run the multi-model serving script once for given (quantile, λ).
    
    Returns
    -------
    str
        Path to the diagnostic pickle file
    """
    # multi_model_dev.py outputs to logs/lam152_x and figures/lam152_x
    # We'll rename these after running
    default_logs_dir = "logs"
    default_figures_dir = os.path.join("figures", f"lam152_{lam152:g}")

    # Convert exit_config list to comma-separated string
    exit_str = ",".join(exit_config)
    
    # Calculate total number of instances
    total_instances = num_r50 + num_r101 + num_r152
    if total_instances == 0:
        raise ValueError("At least one model instance must be specified")

    cmd = [
        "python",
        SERVING_SCRIPT,
        "--scheduler",
        "ours_normalized",
        "--num-r50", str(num_r50),
        "--num-r101", str(num_r101),
        "--num-r152", str(num_r152),
        "--lambda-50", str(lam50),
        "--lambda-101", str(lam101),
        "--lambda-152", str(lam152),
        "--run-seconds", str(run_seconds),
        "--slo-ms", str(slo_ms),
        "--slo-quantile", slo_quantile,
        "--warmup-tasks", str(warmup_tasks),
        "--profile-dir", profile_dir,
        "--exit-points", exit_str,
    ]

    print(f"\n=== Running ours_normalized @ λ152={lam152}, quantile={slo_quantile} ===")
    print(f"Model combination: {num_r50}×R50, {num_r101}×R101, {num_r152}×R152")
    print(f"Lambda distribution: R50={lam50:.1f}, R101={lam101:.1f}, R152={lam152:.1f} req/s")
    print(f"Quantile: {slo_quantile}")
    print("Command:", " ".join(cmd))

    subprocess.run(cmd, check=True)

    # Organize logs and figures into parent directories
    import shutil

    # Move logs/ to logs_quantile_study/lam152_XX/
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

    # Move figures/lam152_XX/ to figures_quantile_study/lam152_XX/
    if os.path.exists(default_figures_dir):
        # Derive figures_dir from logs_dir
        if "logs" in logs_dir:
            figures_base_dir = logs_dir.replace("logs", "figures")
        else:
            figures_base_dir = "figures_" + logs_dir

        # Create lam152_XX subdirectory structure matching logs
        figures_lam_dir = os.path.join(figures_base_dir, f"lam152_{lam152:g}")
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
    diag_dir = os.path.join(logs_dir, f"lam152_{lam152:g}")
    base_diag_name = "multi_model_diag_ours_normalized.pkl"
    base_diag_path = os.path.join(diag_dir, base_diag_name)

    if not os.path.exists(base_diag_path):
        raise FileNotFoundError(
            f"Expected diag file not found: {base_diag_path}. "
            f"Did multi_model_dev.py run successfully?"
        )

    # Rename with quantile suffix
    diag_with_config = os.path.join(
        diag_dir,
        f"multi_model_diag_ours_normalized_quantile_{slo_quantile}.pkl",
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
        description="Quantile study for ours_normalized"
    )
    parser.add_argument(
        "--lambda-ratio",
        type=str,
        default="3:2:1",
        help="Lambda ratio for R50:R101:R152 (e.g., '3:2:1' means R50 gets 3x, R101 gets 2x, R152 gets 1x of base lambda).",
    )
    parser.add_argument(
        "--lambda-152",
        dest="lam152",
        type=str,
        default="40,80,120,160,200",
        help="Comma-separated list of base Poisson arrival rates for ResNet152 instances (req/s). Other models' lambdas are calculated based on --lambda-ratio.",
    )
    parser.add_argument(
        "--num-r50",
        type=int,
        default=1,
        help="Number of ResNet50 model instances.",
    )
    parser.add_argument(
        "--num-r101",
        type=int,
        default=1,
        help="Number of ResNet101 model instances.",
    )
    parser.add_argument(
        "--num-r152",
        type=int,
        default=1,
        help="Number of ResNet152 model instances.",
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
        default=30,
        help="Duration of each run in seconds.",
    )
    parser.add_argument(
        "--slo-ms",
        type=float,
        default=20.0,
        help="SLO threshold in milliseconds.",
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
    args = parser.parse_args()

    # Quantile keys to test
    quantile_keys = ["mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms"]
    
    # Parse lambda ratio (e.g., "3:2:1" for R50:R101:R152)
    ratio_parts = args.lambda_ratio.split(":")
    if len(ratio_parts) != 3:
        raise ValueError(f"Lambda ratio must have 3 parts (R50:R101:R152), got: {args.lambda_ratio}")
    
    try:
        ratio_r50 = float(ratio_parts[0])
        ratio_r101 = float(ratio_parts[1])
        ratio_r152 = float(ratio_parts[2])
    except ValueError:
        raise ValueError(f"Lambda ratio parts must be numeric, got: {args.lambda_ratio}")
    
    # Parse base lambda for ResNet152
    lambda_152_list = [float(x) for x in args.lam152.split(",") if x.strip()]
    
    # Calculate lambda for other models based on ratio
    lambda_50_list = [lam152 * ratio_r50 / ratio_r152 for lam152 in lambda_152_list]
    lambda_101_list = [lam152 * ratio_r101 / ratio_r152 for lam152 in lambda_152_list]
    
    # Parse exit configuration
    exit_config = [x.strip() for x in args.exit_config.split(",") if x.strip()]
    
    if not exit_config:
        raise ValueError("Exit config cannot be empty")
    
    num_r50 = args.num_r50
    num_r101 = args.num_r101
    num_r152 = args.num_r152
    
    slo_ms = args.slo_ms
    run_seconds = args.run_seconds
    warmup_tasks = args.warmup_tasks
    profile_dir = args.profile_dir

    print(f"\n{'='*70}")
    print(f"Quantile Study for ours_normalized")
    print(f"{'='*70}")
    print(f"Model combination: {num_r50}×ResNet50, {num_r101}×ResNet101, {num_r152}×ResNet152")
    print(f"Lambda ratio (R50:R101:R152): {args.lambda_ratio}")
    print(f"Exit configuration: {exit_config}")
    print(f"Lambda R152 values (base): {[f'{x:.1f}' for x in lambda_152_list]}")
    print(f"SLO: {slo_ms} ms")
    print(f"Quantile keys to test: {quantile_keys}")
    print(f"{'='*70}\n")

    # results[quantile_key][lambda_tuple] = metrics
    results = {q: {} for q in quantile_keys}

    for slo_quantile in quantile_keys:
        print(f"\n{'#'*70}")
        print(f"# Testing quantile: {slo_quantile}")
        print(f"{'#'*70}\n")
        
        # Use logs_quantile_study directory for all quantile experiments
        logs_dir = "logs_quantile_study"
        
        for lam50, lam101, lam152 in zip(lambda_50_list, lambda_101_list, lambda_152_list):
            lambda_key = (lam50, lam101, lam152)
            try:
                diag_path = run_one_experiment(
                    num_r50=num_r50,
                    num_r101=num_r101,
                    num_r152=num_r152,
                    lam50=lam50,
                    lam101=lam101,
                    lam152=lam152,
                    run_seconds=run_seconds,
                    slo_ms=slo_ms,
                    slo_quantile=slo_quantile,
                    warmup_tasks=warmup_tasks,
                    profile_dir=profile_dir,
                    logs_dir=logs_dir,
                    exit_config=exit_config,
                )

                metrics = compute_metrics_from_diag(diag_path)
                results[slo_quantile][lambda_key] = metrics

                print(
                    f"[RESULT] quantile={slo_quantile}, "
                    f"λ50={lam50}, λ101={lam101}, λ152={lam152}: "
                    f"p95={metrics['p95_ms']:.2f} ms, "
                    f"drop_ratio={metrics['drop_ratio']:.3f}, "
                    f"completed={metrics['num_completed']}, "
                    f"dropped={metrics['num_dropped']}, "
                    f"avg_exit_depth={metrics['avg_exit_all']:.2f}"
                )
            except Exception as e:
                print(f"[ERROR] Failed for quantile={slo_quantile}, "
                      f"λ50={lam50}, λ101={lam101}, λ152={lam152}: {e}")
                continue

    # Save experiment data
    os.makedirs("logs_quantile_study", exist_ok=True)

    exp_path = os.path.join("logs_quantile_study", "quantile_study_results.pkl")
    
    # Convert tuple keys to string for pickle
    results_serializable = {}
    for quantile, quantile_results in results.items():
        results_serializable[quantile] = {
            f"{k[0]}_{k[1]}_{k[2]}": v for k, v in quantile_results.items()
        }
    
    with open(exp_path, "wb") as f:
        pickle.dump(
            {
                "quantile_keys": quantile_keys,
                "lambda_ratio": args.lambda_ratio,
                "lambda_50_list": lambda_50_list,
                "lambda_101_list": lambda_101_list,
                "lambda_152_list": lambda_152_list,
                "num_r50": num_r50,
                "num_r101": num_r101,
                "num_r152": num_r152,
                "exit_config": exit_config,
                "slo_ms": slo_ms,
                "results": results_serializable,
                "results_with_lambda_tuples": results,  # Keep original for plotting
                "warmup_tasks": warmup_tasks,
            },
            f,
        )
    print(f"\nExperiment data saved to: {exp_path}")
    
    # Print summary tables
    print(f"\n{'='*70}")
    print("SUMMARY TABLES BY QUANTILE")
    print(f"{'='*70}")
    
    for quantile in quantile_keys:
        if not results[quantile]:
            continue
            
        print(f"\n{'-'*70}")
        print(f"Quantile: {quantile}")
        print(f"{'-'*70}")
        print(f"{'λ50':<8} {'λ101':<8} {'λ152':<8} {'P95(ms)':<10} {'Drop%':<10} {'AvgExit':<10} {'Completed':<12} {'Dropped':<10}")
        print(f"{'-'*70}")
        for lambda_key in sorted(results[quantile].keys()):
            m = results[quantile][lambda_key]
            lam50, lam101, lam152 = lambda_key
            print(f"{lam50:<8.0f} {lam101:<8.0f} {lam152:<8.0f} {m['p95_ms']:<10.2f} "
                  f"{m['drop_ratio']*100:<10.2f} {m['avg_exit_all']:<10.2f} "
                  f"{m['num_completed']:<12} {m['num_dropped']:<10}")


if __name__ == "__main__":
    main()
