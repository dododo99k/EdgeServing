"""
run_ours_param_search_dev.py

Parameter search for algorithm_ours over W_SLO / W_ACC / SLO_PENALTY_TARGET.
We fix lambda to a list (default: 80, 90, 100) and run multi_model_dev.py
with --scheduler ours. Each run appends an output-tag folder named by the
current parameter set.
"""

import argparse
import os
import subprocess
import pickle

import numpy as np


SERVING_SCRIPT = "multi_model_dev.py"


def parse_float_list(raw: str):
    return [float(x) for x in raw.split(",") if x.strip()]


def _fmt_float(val: float) -> str:
    return f"{val:g}"


def make_output_tag(w_slo: float, w_acc: float, slo_penalty_target: float) -> str:
    return (
        f"W_SLO_{_fmt_float(w_slo)}"
        f"_W_ACC_{_fmt_float(w_acc)}"
        f"_SLO_PENALTY_TARGET_{_fmt_float(slo_penalty_target)}"
    )


def run_one_experiment(
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
):
    """
    Run multi_model_dev.py once with scheduler=ours and the given parameters.
    Returns the path to the renamed diagnostic pickle and the output tag.
    """
    lam_label = f"{lam:g}"
    output_tag = make_output_tag(w_slo_n, w_acc_n, slo_penalty_target_n)
    diag_dir = os.path.join(logs_dir, f"lam152_{lam_label}", output_tag)
    os.makedirs(diag_dir, exist_ok=True)

    base_diag_name = "multi_model_diag_ours_normalized.pkl"
    base_diag_path = os.path.join(diag_dir, base_diag_name)

    if os.path.exists(base_diag_path):
        os.remove(base_diag_path)

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
        "--output-tag",
        output_tag,
    ]

    env = os.environ.copy()
    env["W_SLO_N"] = str(w_slo_n)
    env["W_ACC_N"] = str(w_acc_n)
    env["SLO_PENALTY_TARGET_N"] = str(slo_penalty_target_n)

    print(f"\n=== Running ours @ λ={lam} | {output_tag} ===")
    print("Command:", " ".join(cmd))

    subprocess.run(cmd, check=True, env=env)

    if not os.path.exists(base_diag_path):
        raise FileNotFoundError(
            f"Expected diag file not found: {base_diag_path}. "
            f"Did {SERVING_SCRIPT} run successfully?"
        )

    diag_with_lam = os.path.join(
        diag_dir,
        f"multi_model_diag_ours_lam{lam_label}.pkl",
    )
    if os.path.exists(diag_with_lam):
        os.remove(diag_with_lam)
    os.rename(base_diag_path, diag_with_lam)

    return diag_with_lam, output_tag


def compute_metrics_from_diag(diag_path: str):
    """
    Compute global p95 latency and drop ratio from a diag pickle.
    """
    with open(diag_path, "rb") as f:
        payload = pickle.load(f)

    total_time_by_model_exit = payload["total_time_by_model_exit"]
    dropped_wait_times_by_model = payload["dropped_wait_times_by_model"]

    exit_points = payload.get("exit_points")
    if not exit_points:
        observed = set()
        for exit_dict in total_time_by_model_exit.values():
            observed.update(exit_dict.keys())
        preferred = ["layer1", "layer2", "layer3", "final"]
        exit_points = [e for e in preferred if e in observed] + sorted(
            [e for e in observed if e not in preferred]
        )

    # Use global absolute depth map for fair comparison across different exit configurations
    GLOBAL_DEPTH_MAP = {"layer1": 1, "layer2": 2, "layer3": 3, "final": 4}
    depth_map = {e: GLOBAL_DEPTH_MAP.get(e, len(GLOBAL_DEPTH_MAP) + 1) for e in exit_points}
    max_depth = max(depth_map.values()) if depth_map else 0

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

    total_all = 0
    sum_depth_all = 0.0
    for _, exit_dict in total_time_by_model_exit.items():
        for e in exit_points:
            c = int(len(exit_dict.get(e, [])))
            total_all += c
            sum_depth_all += float(depth_map[e]) * c

    avg_exit_all = (sum_depth_all / total_all) if total_all > 0 else float("nan")

    return {
        "p95_ms": float(p95_ms),
        "drop_ratio": float(drop_ratio),
        "num_completed": int(num_completed),
        "num_dropped": int(num_dropped),
        "exit_points": list(exit_points),
        "max_exit_depth": int(max_depth),
        "avg_exit_all": float(avg_exit_all),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lambdas",
        type=str,
        default="110",
        help="Comma-separated list of base lambda values.",
    )
    parser.add_argument(
        "--w-slo-n-list",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5",
        help="Comma-separated list of W_SLO values.",
    )
    parser.add_argument(
        "--w-acc-n-list",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5",
        help="Comma-separated list of W_ACC values.",
    )
    parser.add_argument(
        "--slo-penalty-target-n-list",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5",
        help="Comma-separated list of SLO_PENALTY_TARGET values.",
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
        default="logs_baseline",
        help="Directory where diag and experiment results are stored.",
    )
    args = parser.parse_args()

    lambda_list = parse_float_list(args.lambdas)
    w_slo_n_list = parse_float_list(args.w_slo_n_list)
    w_acc_n_list = parse_float_list(args.w_acc_n_list)
    slo_penalty_target_n_list = parse_float_list(args.slo_penalty_target_n_list)

    results = {}

    for lam in lambda_list:
        results[lam] = {}
        for w_slo_n in w_slo_n_list:
            for w_acc_n in w_acc_n_list:
                for slo_penalty_target_n in slo_penalty_target_n_list:
                    diag_path, output_tag = run_one_experiment(
                        lam=lam,
                        run_seconds=args.run_seconds,
                        slo_ms=args.slo_ms,
                        slo_quantile=args.slo_quantile,
                        warmup_tasks=args.warmup_tasks,
                        profile_dir=args.profile_dir,
                        logs_dir=args.logs_dir,
                        w_slo_n=w_slo_n,
                        w_acc_n=w_acc_n,
                        slo_penalty_target_n=slo_penalty_target_n,
                    )

                    metrics = compute_metrics_from_diag(diag_path)
                    metrics.update(
                        {
                            "w_slo_n": float(w_slo_n),
                            "w_acc_n": float(w_acc_n),
                            "slo_penalty_target_n": float(slo_penalty_target_n),
                            "output_tag": output_tag,
                            "diag_path": diag_path,
                        }
                    )
                    results[lam][output_tag] = metrics

                    print(
                        f"[RESULT] λ={lam} | {output_tag}: "
                        f"p95={metrics['p95_ms']:.2f} ms, "
                        f"drop_ratio={metrics['drop_ratio']:.3f}, "
                        f"completed={metrics['num_completed']}, "
                        f"dropped={metrics['num_dropped']}"
                    )

    os.makedirs(args.logs_dir, exist_ok=True)
    exp_path = os.path.join(args.logs_dir, "ours_normalized_param_search.pkl")
    with open(exp_path, "wb") as f:
        pickle.dump(
            {
                "lambda_list": lambda_list,
                "w_slo_n_list": w_slo_n_list,
                "w_acc_n_list": w_acc_n_list,
                "slo_penalty_target_n_list": slo_penalty_target_n_list,
                "results": results,
                "slo_ms": args.slo_ms,
                "slo_quantile": args.slo_quantile,
                "warmup_tasks": args.warmup_tasks,
            },
            f,
        )
    print(f"\nExperiment data saved to: {exp_path}")


if __name__ == "__main__":
    main()
