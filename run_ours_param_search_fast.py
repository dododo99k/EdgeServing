"""
run_ours_param_search_fast.py

Fast parameter search using:
1. Coarse-to-fine search strategy
2. Bayesian optimization with Optuna
3. Shorter run times for exploration
4. Early stopping for bad configurations
"""

import argparse
import os
import subprocess
import pickle
import numpy as np
from typing import Dict, Any, Optional, Tuple
import json

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not installed. Using grid search fallback.")
    print("Install with: pip install optuna")


SERVING_SCRIPT = "multi_model_dev.py"


def run_single_experiment(
    lam: float,
    run_seconds: int,
    slo_ms: float,
    slo_quantile: str,
    warmup_tasks: int,
    profile_dir: str,
    logs_dir: str,
    w_slo_n: float,
    w_acc_n: float,
    slo_penalty_target_n: float,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one experiment and return metrics."""
    
    lam_label = f"{lam:g}"
    output_tag = f"W_SLO_{w_slo_n:g}_W_ACC_{w_acc_n:g}_SLO_PENALTY_TARGET_{slo_penalty_target_n:g}"
    
    # NOTE: multi_model_dev.py hardcodes logs_dir as "logs/lam152_xxx/output_tag"
    # So we read from there, not from the user-specified logs_dir
    actual_logs_dir = os.path.join("logs", f"lam152_{lam_label}", output_tag)
    os.makedirs(actual_logs_dir, exist_ok=True)
    
    # Also create our summary logs dir
    summary_dir = os.path.join(logs_dir, f"lam152_{lam_label}", output_tag)
    os.makedirs(summary_dir, exist_ok=True)

    base_diag_name = "multi_model_diag_ours_normalized.pkl"
    base_diag_path = os.path.join(actual_logs_dir, base_diag_name)

    if os.path.exists(base_diag_path):
        os.remove(base_diag_path)

    cmd = [
        "python", SERVING_SCRIPT,
        "--scheduler", "ours_normalized",
        "--lambda-50", str(lam * 3),
        "--lambda-101", str(lam * 2),
        "--lambda-152", str(lam * 1),
        "--run-seconds", str(run_seconds),
        "--slo-ms", str(slo_ms),
        "--slo-quantile", slo_quantile,
        "--warmup-tasks", str(warmup_tasks),
        "--profile-dir", profile_dir,
        "--output-tag", output_tag,
    ]

    env = os.environ.copy()
    env["W_SLO_N"] = str(w_slo_n)
    env["W_ACC_N"] = str(w_acc_n)
    env["SLO_PENALTY_TARGET_N"] = str(slo_penalty_target_n)

    if verbose:
        print(f"\n=== Running: W_SLO={w_slo_n}, W_ACC={w_acc_n}, SLO_TARGET={slo_penalty_target_n} ===")

    result = subprocess.run(cmd, check=True, env=env, capture_output=not verbose)

    # The diag file is in the actual_logs_dir (hardcoded by multi_model_dev.py)
    diag_with_lam = os.path.join(actual_logs_dir, f"multi_model_diag_ours_lam{lam_label}.pkl")
    if os.path.exists(diag_with_lam):
        os.remove(diag_with_lam)
    os.rename(base_diag_path, diag_with_lam)

    return compute_metrics(diag_with_lam)


def compute_metrics(diag_path: str) -> Dict[str, Any]:
    """Compute metrics from diagnostic pickle."""
    with open(diag_path, "rb") as f:
        payload = pickle.load(f)

    total_time_by_model_exit = payload["total_time_by_model_exit"]
    slo_ms = payload.get("slo_ms", 50.0)

    # Try to get SLO violations from new format, fallback to old dropped format
    slo_violations_by_model = payload.get("slo_violations_by_model", None)

    num_violations = sum(v["violations"] for v in slo_violations_by_model.values())
    num_completed = sum(v["total"] for v in slo_violations_by_model.values())
    violation_ratio = num_violations / num_completed if num_completed > 0 else 0.0

    exit_points = payload.get("exit_points")
    if not exit_points:
        observed = set()
        for exit_dict in total_time_by_model_exit.values():
            observed.update(exit_dict.keys())
        preferred = ["layer1", "layer2", "layer3", "final"]
        exit_points = [e for e in preferred if e in observed] + sorted(
            [e for e in observed if e not in preferred]
        )

    GLOBAL_DEPTH_MAP = {"layer1": 1, "layer2": 2, "layer3": 3, "final": 4}
    depth_map = {e: GLOBAL_DEPTH_MAP.get(e, len(GLOBAL_DEPTH_MAP) + 1) for e in exit_points}

    all_times = []
    for _, exit_dict in total_time_by_model_exit.items():
        for _, times in exit_dict.items():
            all_times.extend(times)

    all_times = np.array(all_times, dtype=np.float64)

    if all_times.size == 0:
        p95_ms = float("nan")
        mean_ms = float("nan")
    else:
        p95_ms = float(np.percentile(all_times * 1000.0, 95))
        mean_ms = float(np.mean(all_times * 1000.0))

    # Note: num_completed and violation_ratio already computed above
    slo_violation_ratio = violation_ratio
    throughput = num_completed

    # Average exit depth (lower = earlier exit = faster but less accurate)
    total_all = 0
    sum_depth_all = 0.0
    for _, exit_dict in total_time_by_model_exit.items():
        for e in exit_points:
            c = int(len(exit_dict.get(e, [])))
            total_all += c
            sum_depth_all += float(depth_map[e]) * c

    avg_exit_depth = (sum_depth_all / total_all) if total_all > 0 else float("nan")

    return {
        "p95_ms": p95_ms,
        "mean_ms": mean_ms,
        "slo_violation_ratio": slo_violation_ratio,
        "num_completed": num_completed,
        "num_violations": int(num_violations),
        "avg_exit_depth": avg_exit_depth,
        "throughput": throughput,
        # Keep old names for backward compatibility
        # "drop_ratio": slo_violation_ratio,
        # "num_dropped": int(num_violations),
    }


def compute_objective(metrics: Dict[str, Any], slo_ms: float,
                      w_violation: float = 30.0,
                      w_depth: float = 100.0,
                      violation_threshold: float = 0.05) -> float:
    """
    Compute optimization objective (higher is better).

    Optimizes based on:
    - SLO violation ratio: heavily penalize tasks exceeding SLO with nonlinear penalty
    - Exit depth: prefer using deeper exits (better accuracy)
    - Throughput: more completed tasks is better

    Args:
        w_violation: weight for violation ratio penalty (higher = stricter SLO compliance)
        w_depth: weight for exit depth reward (higher = prefer deeper/more accurate exits)
        violation_threshold: violation ratio above this triggers exponential penalty

    Note: p95 latency is tracked but not used in optimization objective.
          Only the actual violation ratio (% of tasks exceeding SLO) is optimized.
    """
    # Support both new and old metric names
    violation_ratio = metrics.get("slo_violation_ratio")
    avg_depth = metrics["avg_exit_depth"]
    throughput = metrics["num_completed"]

    if np.isnan(avg_depth):
        return -1e6

    # Violation ratio penalty with nonlinear scaling
    violation_penalty = w_violation * violation_ratio

    # Additional exponential penalty when violation_ratio > threshold
    if violation_ratio > violation_threshold:
        excess_violation = violation_ratio - violation_threshold
        violation_penalty += 10.0 * (np.exp(excess_violation * 10) - 1)

    # Hard constraint: if violation_ratio > 20%, severely penalize
    if violation_ratio > 0.20:
        violation_penalty += 1000.0

    # Objective: maximize depth while minimizing violations
    # Use sqrt to create non-linear reward: strongly discourage depth=1, diminishing returns at higher depths

    score = (
        - violation_penalty              # Heavily penalize SLO violations (ratio)
        + w_depth * np.sqrt((avg_depth - 1) / 3)         # Reward deeper exits with non-linear scaling
        + 0.001 * throughput / 1000      # Small bonus for throughput
    )

    return score


def bayesian_search(args) -> Dict[str, Any]:
    """Use Optuna for Bayesian optimization."""
    
    def objective(trial: optuna.Trial) -> float:
        # Sample parameters with informed priors
        w_slo_n = trial.suggest_float("w_slo_n", 0.00001, 20.0, log=True)
        w_acc_n = trial.suggest_float("w_acc_n", 0.00001, 20.0, log=True)
        slo_penalty_target_n = trial.suggest_float("slo_penalty_target_n", 0.01, 3.0)
        
        try:
            metrics = run_single_experiment(
                lam=args.lam,
                run_seconds=args.run_seconds,
                slo_ms=args.slo_ms,
                slo_quantile=args.slo_quantile,
                warmup_tasks=args.warmup_tasks,
                profile_dir=args.profile_dir,
                logs_dir=args.logs_dir,
                w_slo_n=w_slo_n,
                w_acc_n=w_acc_n,
                slo_penalty_target_n=slo_penalty_target_n,
                verbose=args.verbose,
            )
            
            score = compute_objective(metrics, args.slo_ms)
            
            # Log intermediate results
            trial.set_user_attr("p95_ms", metrics["p95_ms"])
            trial.set_user_attr("slo_violation_ratio", metrics["slo_violation_ratio"])
            trial.set_user_attr("avg_exit_depth", metrics["avg_exit_depth"])

            print(f"  → p95={metrics['p95_ms']:.2f}ms, slo_vio={metrics['slo_violation_ratio']:.3f}, "
                  f"depth={metrics['avg_exit_depth']:.2f}, score={score:.4f}")
            
            return score
            
        except Exception as e:
            print(f"  → Failed: {e}")
            return -1e6
    
    # Create study with TPE sampler (good for small budgets)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    
    # Add some good starting points based on prior knowledge
    # Strategy: Use much higher w_acc_n to overcome layer1's 3x speed advantage with linear normalization
    study.enqueue_trial({"w_slo_n": 0.01, "w_acc_n": 15.0, "slo_penalty_target_n": 1.0})    # Very high accuracy weight
    study.enqueue_trial({"w_slo_n": 0.001, "w_acc_n": 10.0, "slo_penalty_target_n": 1.5})  # High accuracy, very low SLO
    study.enqueue_trial({"w_slo_n": 0.05, "w_acc_n": 12.0, "slo_penalty_target_n": 0.5})   # High accuracy, low target
    study.enqueue_trial({"w_slo_n": 0.5, "w_acc_n": 8.0, "slo_penalty_target_n": 2.0})     # Medium-high accuracy, balanced
    study.enqueue_trial({"w_slo_n": 0.1, "w_acc_n": 5.0, "slo_penalty_target_n": 1.0})     # Medium accuracy weight
    study.enqueue_trial({"w_slo_n": 0.1, "w_acc_n": 1.45, "slo_penalty_target_n": 1.64})   # Previous best (baseline)
    
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    print("\n" + "="*60)
    print("BEST PARAMETERS FOUND:")
    print("="*60)
    print(f"  W_SLO_N = {study.best_params['w_slo_n']:.4f}")
    print(f"  W_ACC_N = {study.best_params['w_acc_n']:.4f}")
    print(f"  SLO_PENALTY_TARGET_N = {study.best_params['slo_penalty_target_n']:.4f}")
    print(f"\nBest score: {study.best_value:.4f}")
    print(f"  p95_ms: {study.best_trial.user_attrs.get('p95_ms', 'N/A')}")
    print(f"  slo_violation_ratio: {study.best_trial.user_attrs.get('slo_violation_ratio', 'N/A')}")
    print(f"  avg_exit_depth: {study.best_trial.user_attrs.get('avg_exit_depth', 'N/A')}")
    
    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "best_trial": study.best_trial,
        "study": study,
    }


def coarse_to_fine_search(args) -> Dict[str, Any]:
    """Two-stage coarse-to-fine grid search."""
    
    print("\n=== STAGE 1: Coarse Search ===")
    # Coarse grid
    coarse_w_slo = [0.2, 0.5, 1.0, 1.5]
    coarse_w_acc = [0.1, 0.3, 0.6, 1.0]
    coarse_slo_target = [0.5, 1.0, 1.5, 2.0]
    
    best_score = -float("inf")
    best_params = None
    results = []
    
    total = len(coarse_w_slo) * len(coarse_w_acc) * len(coarse_slo_target)
    count = 0
    
    for w_slo in coarse_w_slo:
        for w_acc in coarse_w_acc:
            for slo_target in coarse_slo_target:
                count += 1
                print(f"\n[{count}/{total}] Testing: W_SLO={w_slo}, W_ACC={w_acc}, SLO_TARGET={slo_target}")
                
                try:
                    metrics = run_single_experiment(
                        lam=args.lam,
                        run_seconds=args.run_seconds_coarse,
                        slo_ms=args.slo_ms,
                        slo_quantile=args.slo_quantile,
                        warmup_tasks=args.warmup_tasks,
                        profile_dir=args.profile_dir,
                        logs_dir=args.logs_dir,
                        w_slo_n=w_slo,
                        w_acc_n=w_acc,
                        slo_penalty_target_n=slo_target,
                        verbose=args.verbose,
                    )
                    
                    score = compute_objective(metrics, args.slo_ms)
                    results.append({
                        "w_slo_n": w_slo, "w_acc_n": w_acc, 
                        "slo_penalty_target_n": slo_target,
                        "score": score, **metrics
                    })
                    
                    print(f"  → p95={metrics['p95_ms']:.2f}ms, slo_viol={metrics['slo_violation_ratio']:.3f}, "
                          f"depth={metrics['avg_exit_depth']:.2f}, score={score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = (w_slo, w_acc, slo_target)
                        
                except Exception as e:
                    print(f"  → Failed: {e}")
    
    if best_params is None:
        print("No valid results in coarse search!")
        return {"results": results}
    
    print(f"\n=== STAGE 1 Best: W_SLO={best_params[0]}, W_ACC={best_params[1]}, "
          f"SLO_TARGET={best_params[2]}, score={best_score:.4f} ===")
    
    # Fine search around best coarse params
    print("\n=== STAGE 2: Fine Search ===")
    
    w_slo_center, w_acc_center, slo_target_center = best_params
    
    fine_w_slo = [max(0.05, w_slo_center - 0.2), w_slo_center, w_slo_center + 0.2]
    fine_w_acc = [max(0.05, w_acc_center - 0.1), w_acc_center, w_acc_center + 0.1]
    fine_slo_target = [max(0.2, slo_target_center - 0.3), slo_target_center, slo_target_center + 0.3]
    
    total = len(fine_w_slo) * len(fine_w_acc) * len(fine_slo_target)
    count = 0
    
    for w_slo in fine_w_slo:
        for w_acc in fine_w_acc:
            for slo_target in fine_slo_target:
                count += 1
                print(f"\n[{count}/{total}] Fine-tuning: W_SLO={w_slo:.2f}, W_ACC={w_acc:.2f}, "
                      f"SLO_TARGET={slo_target:.2f}")
                
                try:
                    metrics = run_single_experiment(
                        lam=args.lam,
                        run_seconds=args.run_seconds,  # Full duration for fine search
                        slo_ms=args.slo_ms,
                        slo_quantile=args.slo_quantile,
                        warmup_tasks=args.warmup_tasks,
                        profile_dir=args.profile_dir,
                        logs_dir=args.logs_dir,
                        w_slo_n=w_slo,
                        w_acc_n=w_acc,
                        slo_penalty_target_n=slo_target,
                        verbose=args.verbose,
                    )
                    
                    score = compute_objective(metrics, args.slo_ms)
                    results.append({
                        "w_slo_n": w_slo, "w_acc_n": w_acc, 
                        "slo_penalty_target_n": slo_target,
                        "score": score, "stage": "fine", **metrics
                    })
                    
                    print(f"  → p95={metrics['p95_ms']:.2f}ms, slo_viol={metrics['slo_violation_ratio']:.3f}, "
                          f"depth={metrics['avg_exit_depth']:.2f}, score={score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = (w_slo, w_acc, slo_target)
                        
                except Exception as e:
                    print(f"  → Failed: {e}")
    
    print("\n" + "="*60)
    print("BEST PARAMETERS FOUND:")
    print("="*60)
    print(f"  W_SLO_N = {best_params[0]:.4f}")
    print(f"  W_ACC_N = {best_params[1]:.4f}")
    print(f"  SLO_PENALTY_TARGET_N = {best_params[2]:.4f}")
    print(f"  Best score: {best_score:.4f}")
    
    return {
        "best_params": {
            "w_slo_n": best_params[0],
            "w_acc_n": best_params[1], 
            "slo_penalty_target_n": best_params[2],
        },
        "best_score": best_score,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Fast parameter search for ours_normalized scheduler")
    
    parser.add_argument("--method", type=str, default="bayesian",
                        choices=["bayesian", "coarse-to-fine"],
                        help="Search method: 'bayesian' (requires optuna) or 'coarse-to-fine'")
    parser.add_argument("--lam", type=float, default=160,
                        help="Base lambda value for Poisson arrival rate")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of trials for Bayesian optimization")
    parser.add_argument("--run-seconds", type=int, default=30,
                        help="Duration of each run in seconds")
    parser.add_argument("--run-seconds-coarse", type=int, default=10,
                        help="Shorter duration for coarse search stage")
    parser.add_argument("--slo-ms", type=float, default=20.0,
                        help="Total latency SLO in milliseconds")
    parser.add_argument("--slo-quantile", type=str, default="p95_ms",
                        choices=["mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms"])
    parser.add_argument("--warmup-tasks", type=int, default=50,
                        help="Number of warmup tasks (reduced for faster search)")
    parser.add_argument("--profile-dir", type=str, default="saves")
    parser.add_argument("--logs-dir", type=str, default="logs_param_search")
    parser.add_argument("--verbose", action="store_true",
                        help="Show subprocess output")
    
    args = parser.parse_args()
    
    os.makedirs(args.logs_dir, exist_ok=True)
    
    if args.method == "bayesian":
        if not OPTUNA_AVAILABLE:
            print("Optuna not available. Falling back to coarse-to-fine search.")
            args.method = "coarse-to-fine"
        else:
            results = bayesian_search(args)
    
    if args.method == "coarse-to-fine":
        results = coarse_to_fine_search(args)
    
    # Save results
    output_path = os.path.join(args.logs_dir, f"param_search_{args.method}_results.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_path}")
    
    # Also save as JSON for easy reading
    json_path = os.path.join(args.logs_dir, f"param_search_{args.method}_best.json")
    with open(json_path, "w") as f:
        json.dump({
            "best_params": results.get("best_params", {}),
            "best_score": results.get("best_score", None),
        }, f, indent=2)
    print(f"Best params saved to: {json_path}")


if __name__ == "__main__":
    main()
    # use command delete ./logs/
    subprocess.run(["rm", "-rf", "./logs/"])
    subprocess.run(["rm", "-rf", "./figures/"])