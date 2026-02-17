"""
run_early_exit_lowest_interference_model_combination.py

Model combination study for algorithm_early_exit_lowest_interference.

This script tests the performance of our normalized scheduler with different
model instance combinations:
  - 1×R50 + 1×R101 + 1×R152 (baseline)
  - 2×R101 + 1×R152
  - 3×R152
  - etc.

For each combination, we:
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
    Run the multi-model serving script once for given (model_combination, λ).
    
    Parameters
    ----------
    num_r50 : int
        Number of ResNet50 instances
    num_r101 : int
        Number of ResNet101 instances
    num_r152 : int
        Number of ResNet152 instances
    lam50 : float
        Total Poisson arrival rate for ResNet50 instances (req/s)
    lam101 : float
        Total Poisson arrival rate for ResNet101 instances (req/s)
    lam152 : float
        Total Poisson arrival rate for ResNet152 instances (req/s)
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
    exit_config : list
        List of active exit points
        
    Returns
    -------
    str
        Path to the diagnostic pickle file
    """
    combo_label = f"{num_r50}r50_{num_r101}r101_{num_r152}r152"
    
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
        "early_exit_lowest_interference",
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

    print(f"\n=== Running early_exit_lowest_interference @ λ152={lam152}, combo={combo_label} ===")
    print(f"Model combination: {num_r50}×R50, {num_r101}×R101, {num_r152}×R152")
    print(f"Lambda distribution: R50={lam50:.1f}, R101={lam101:.1f}, R152={lam152:.1f} req/s")
    print("Command:", " ".join(cmd))

    subprocess.run(cmd, check=True)

    # Organize logs and figures into parent directories
    import shutil

    # Move logs/ to logs_diff_model_combination/lam152_XX/
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

    # Move figures/lam152_XX/ to figures_diff_model_combination/lam152_XX/
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
    base_diag_name = "multi_model_diag_early_exit_lowest_interference.pkl"
    base_diag_path = os.path.join(diag_dir, base_diag_name)

    if not os.path.exists(base_diag_path):
        raise FileNotFoundError(
            f"Expected diag file not found: {base_diag_path}. "
            f"Did multi_model_dev.py run successfully?"
        )

    # Rename with combo config suffix
    diag_with_config = os.path.join(
        diag_dir,
        f"multi_model_diag_early_exit_lowest_interference_combo_{combo_label}.pkl",
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
        description="Model combination study for early_exit_lowest_interference"
    )
    parser.add_argument(
        "--lambda-ratio",
        type=str,
        default="1:1:1",
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
        "--r50-r101-r152",
        type=str,
        nargs="+",
        default=["1,1,1", "0,1,2", "0,2,1", "3,0,0", "0,3,0", "0,0,3"],
        help="Model combinations to test. Each entry is 'num_r50,num_r101,num_r152'. "
             "Sum must be 3. Example: --r50-r101-r152 '1,1,1' '0,1,2' '3,0,0'",
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
        "--slo-ms",
        type=float,
        default=50.0,
        help="SLO threshold in milliseconds.",
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
        help="Directory where diag and experiment results are stored. Default: logs_combo_{r50}r50_{r101}r101_{r152}r152",
    )
    args = parser.parse_args()

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
    print(f"Calculated lambda lists based on ratio {args.lambda_ratio}:")
    print(f"  Lambda R50: {[f'{x:.1f}' for x in lambda_50_list]}")
    print(f"  Lambda R101: {[f'{x:.1f}' for x in lambda_101_list]}")
    print(f"  Lambda R152: {[f'{x:.1f}' for x in lambda_152_list]}")

    # Parse exit configurations (now a list of configurations)
    # Parse exit configuration
    exit_config = [x.strip() for x in args.exit_config.split(",") if x.strip()]

    if not exit_config:
        raise ValueError("Exit config cannot be empty")


    # Parse model combinations
    model_combinations = []
    for combo_str in getattr(args, "r50_r101_r152"):
        parts = combo_str.split(",")
        if len(parts) != 3:
            raise ValueError(f"Each combination must have 3 parts (r50,r101,r152), got: {combo_str}")
        try:
            num_r50 = int(parts[0])
            num_r101 = int(parts[1])
            num_r152 = int(parts[2])
        except ValueError:
            raise ValueError(f"Model numbers must be integers, got: {combo_str}")

        total = num_r50 + num_r101 + num_r152
        if total != 3:
            raise ValueError(f"Model combination sum must be 3, got {total} for '{combo_str}'")

        model_combinations.append((num_r50, num_r101, num_r152))

    slo_ms = args.slo_ms
    run_seconds = args.run_seconds
    slo_quantile = args.slo_quantile
    warmup_tasks = args.warmup_tasks
    profile_dir = args.profile_dir

    # Auto-generate logs_dir based on model combination if not provided
    if args.logs_dir is None:
        logs_dir = "logs_diff_model_combination"
    else:
        logs_dir = args.logs_dir

    print(f"\n{'='*60}")
    print(f"Model Combination Study for early_exit_lowest_interference")
    print(f"{'='*60}")
    print(f"Number of combinations to test: {len(model_combinations)}")
    print(f"Combinations:")
    for num_r50, num_r101, num_r152 in model_combinations:
        print(f"  - {num_r50}×R50 + {num_r101}×R101 + {num_r152}×R152")
    print(f"Lambda ratio (R50:R101:R152): {args.lambda_ratio}")
    print(f"Lambda R50 values: {[f'{x:.1f}' for x in lambda_50_list]}")
    print(f"Lambda R101 values: {[f'{x:.1f}' for x in lambda_101_list]}")
    print(f"Lambda R152 values (base): {[f'{x:.1f}' for x in lambda_152_list]}")
    print(f"SLO: {slo_ms} ms")
    print(f"{'='*60}\n")


    # Loop over all model combinations
    for num_r50, num_r101, num_r152 in model_combinations:
        combo_label = f"{num_r50}r50_{num_r101}r101_{num_r152}r152"
        print(f"\n{'='*60}")
        print(f"Testing combination: {num_r50}×R50 + {num_r101}×R101 + {num_r152}×R152")
        print(f"{'='*60}")

        # results[lambda_tuple] = metrics
        results = {}

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
                results[lambda_key] = metrics

                print(
                    f"[RESULT] combo={combo_label}, "
                    f"λ50={lam50:.1f}, λ101={lam101:.1f}, λ152={lam152:.1f}: "
                    f"p95={metrics['p95_ms']:.2f} ms, "
                    f"violate_ratio={metrics['violate_ratio']:.3f}, "
                    f"completed={metrics['num_completed']}, "
                    f"num_violated={metrics['num_violated']}, "
                    f"avg_exit_depth={metrics['avg_exit_all']:.2f}"
                )
            except Exception as e:
                print(f"[ERROR] Failed for combo={combo_label}, "
                        f"λ50={lam50:.1f}, λ101={lam101:.1f}, λ152={lam152:.1f}: {e}")
                continue

        # Save experiment data for this combination
        os.makedirs(logs_dir, exist_ok=True)

        exp_path = os.path.join(logs_dir, f"combo_{combo_label}_results.pkl")

        # Convert tuple keys to string for pickle
        results_serializable = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in results.items()}

        with open(exp_path, "wb") as f:
            pickle.dump(
                {
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
                    "slo_quantile": slo_quantile,
                    "warmup_tasks": warmup_tasks,
                },
                f,
            )
        print(f"\nExperiment data saved to: {exp_path}")

        # Print summary table for this combination
        print(f"\n{'='*60}")
        print(f"SUMMARY TABLE - {combo_label}")
        print(f"{'='*60}")
        print(f"Model Combination: {num_r50}×R50 + {num_r101}×R101 + {num_r152}×R152")
        print(f"{'-'*60}")
        print(f"{'λ50':<8} {'λ101':<8} {'λ152':<8} {'P95(ms)':<10} {'Violate%':<10} {'AvgExit':<10} {'Completed':<12} {'num_violated':<10}")
        print(f"{'-'*60}")
        for lambda_key in sorted(results.keys()):
            m = results[lambda_key]
            lam50, lam101, lam152 = lambda_key
            print(f"{lam50:<8.0f} {lam101:<8.0f} {lam152:<8.0f} {m['p95_ms']:<10.2f} "
                    f"{m['violate_ratio']*100:<10.2f} {m['avg_exit_all']:<10.2f} "
                    f"{m['num_completed']:<12} {m['num_violated']:<10}")

    print(f"\n{'='*60}")
    print(f"All {len(model_combinations)} combinations completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
    subprocess.run(["python", "plot_model_combination_comparison.py"], check=True)
