import os,pickle
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

from run_multi_model_intensity_dev import compute_metrics_from_diag

def _set_x_ticks_40(ax, lambda_list):
    if not lambda_list:
        return
    data_min = min(lambda_list)
    data_max = max(lambda_list)
    # Calculate tick positions that are multiples of 40
    tick_min = int(np.floor(data_min / 40.0) * 40)
    tick_max = int(np.ceil(data_max / 40.0) * 40)
    if tick_max < tick_min:
        tick_max = tick_min
    ticks = np.arange(tick_min, tick_max + 1, 40)
    ax.set_xticks(ticks)
    # Set xlim based on actual data range
    ax.set_xlim(data_min - 10, data_max + 10)

def _legend_label(name: str) -> str:
    if name == "ours_normalized":
        return "our"
    return name


def plot_results(

    lambda_list,
    results,
    out_dir: str,
):
    """
    Plot comparison between schedulers.

    results structure:
        results[scheduler][lam] = {
            "p95_ms": ...,
            "drop_ratio": ...,
            "num_completed": ...,
            "num_dropped": ...,
        }
    """
    os.makedirs(out_dir, exist_ok=True)

    schedulers = sorted(results.keys())  # ["all_final", "early_exit"], etc.

    # Plot p95 latency
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title("Impact of Scheduling Policy on P95 Latency", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Traffic Intensity λ (req/s)", fontsize=12)
    ax1.set_ylabel("p95 total latency (ms)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)

    for scheduler in schedulers:
        p95_vals = []
        valid_lams = []
        for lam in lambda_list:
            if lam not in results[scheduler]:
                continue
            m = results[scheduler][lam]
            p95_vals.append(m["p95_ms"])
            valid_lams.append(lam)

        if not valid_lams:
            continue

        lam_arr = np.array(valid_lams, dtype=np.float64)
        p95_arr = np.array(p95_vals, dtype=np.float64)

        lw = 3 if scheduler == "ours_normalized" else 1.5
        ax1.plot(
            lam_arr,
            p95_arr,
            marker="o",
            linestyle="-",
            linewidth=lw,
            label=_legend_label(scheduler),
        )

    ax1.legend(loc="upper left", fontsize=14)
    _set_x_ticks_40(ax1, lambda_list)
    fig1.tight_layout()
    out_path1 = os.path.join(out_dir, "main_multi_model_p95_latency_compare.png")
    plt.savefig(out_path1, dpi=150)
    print(f"p95 latency figure saved to: {out_path1}")
    plt.close(fig1)

    # Plot drop ratio
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.set_title("Impact of Scheduling Policy on Drop Ratio", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Traffic Intensity λ (req/s)", fontsize=12)
    ax2.set_ylabel("Drop ratio", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)

    for scheduler in schedulers:
        drop_vals = []
        valid_lams = []
        for lam in lambda_list:
            if lam not in results[scheduler]:
                continue
            m = results[scheduler][lam]
            drop_vals.append(m["drop_ratio"])
            valid_lams.append(lam)
        
        if not valid_lams:
            continue

        lam_arr = np.array(valid_lams, dtype=np.float64)
        drop_arr = np.array(drop_vals, dtype=np.float64)

        lw = 3 if scheduler == "ours_normalized" else 1.5
        ax2.plot(
            lam_arr,
            drop_arr,
            marker="s",
            linestyle="-",
            linewidth=lw,
            label=_legend_label(scheduler),
        )

    ax2.legend(loc="upper left", fontsize=14)
    ax2.set_ylim(0.0, 0.5)
    _set_x_ticks_40(ax2, lambda_list)
    fig2.tight_layout()
    out_path2 = os.path.join(out_dir, "main_multi_model_drop_ratio_compare.png")
    plt.savefig(out_path2, dpi=150)
    print(f"Drop ratio figure saved to: {out_path2}")
    plt.close(fig2)

    # Plot drop ratio without symphony for clearer comparison
    schedulers_temp = [s for s in schedulers if s != "symphony"]
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.set_title("Scheduling Policy Impact on Drop Ratio (w/o Symphony)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Traffic Intensity λ (req/s)", fontsize=12)
    ax3.set_ylabel("Drop ratio", fontsize=12)
    ax3.grid(True, linestyle="-", alpha=0.5)
    for scheduler in schedulers_temp:
        drop_vals = []
        valid_lams = []
        for lam in lambda_list:
            if lam not in results[scheduler]:
                continue
            m = results[scheduler][lam]
            drop_vals.append(m["drop_ratio"])
            valid_lams.append(lam)
        
        if not valid_lams:
            continue

        lam_arr = np.array(valid_lams, dtype=np.float64)
        drop_arr = np.array(drop_vals, dtype=np.float64)

        lw = 3 if scheduler == "ours_normalized" else 1.5
        ax3.plot(
            lam_arr,
            drop_arr,
            marker="s",
            linestyle="-",
            linewidth=lw,
            label=_legend_label(scheduler),
        )

    ax3.legend(loc="upper left")
    ax3.set_ylim(0.0, 0.012)
    _set_x_ticks_40(ax3, lambda_list)
    fig3.tight_layout()
    out_path3 = os.path.join(out_dir, "main_multi_model_drop_ratio_without_symphony_compare.png")
    plt.savefig(out_path3, dpi=150)
    print(f"Drop ratio figure saved to: {out_path3}")
    plt.close(fig3)

    # Separate figure: average early-exit depth vs traffic intensity
    plot_avg_exit_comparison(lambda_list, results, out_dir=out_dir)


def plot_avg_exit_comparison(lambda_list, results, out_dir: str):
    """
    Plot average early-exit depth (completed tasks only) vs traffic intensity, comparing schedulers.

    This is saved as a separate figure to avoid overloading the latency/drop plot.
    """
    os.makedirs(out_dir, exist_ok=True)

    schedulers = sorted(results.keys())
    if not schedulers or not lambda_list:
        return

    # Pull exit_points / max_depth from the first available datapoint
    first_metric = None
    for s in schedulers:
        for lam in lambda_list:
            if lam in results[s]:
                first_metric = results[s][lam]
                break
        if first_metric is not None:
            break

    if first_metric is None:
        return

    exit_points = first_metric.get("exit_points", ["layer1", "layer2", "layer3", "final"])
    max_depth = int(first_metric.get("max_exit_depth", len(exit_points))) if exit_points else 0

    # Collect model names observed across schedulers/lambdas
    model_names = set()
    for s in schedulers:
        for lam in lambda_list:
            m = results[s].get(lam, {})
            for mn in (m.get("avg_exit_by_model") or {}).keys():
                model_names.add(mn)

    def _model_sort_key(name: str):
        # Prefer numeric ordering (e.g., ResNet50 < ResNet101 < ResNet152)
        m = re.search(r"(\d+)", str(name))
        return (0, int(m.group(1))) if m else (1, str(name))

    model_order = sorted(model_names, key=_model_sort_key)

    targets = ["ALL_MODELS"] + model_order
    if not targets:
        return

    for target in targets:
        fig, ax = plt.subplots(figsize=(10, 4))

        for s in schedulers:
            vals = []
            valid_lams = []
            for lam in lambda_list:
                metric = results[s].get(lam, {})
                if target == "ALL_MODELS":
                    v = metric.get("avg_exit_all", float("nan"))
                else:
                    v = (metric.get("avg_exit_by_model") or {}).get(target, float("nan"))
                if not np.isnan(v):
                    vals.append(v)
                    valid_lams.append(lam)

            if not valid_lams:
                continue
            x = np.array(valid_lams, dtype=np.float64)
            y = np.array(vals, dtype=np.float64)
            lw = 3 if s == "ours_normalized" else 1.5
            ax.plot(x, y, marker="o", linestyle="-", linewidth=lw, label=_legend_label(s))

        title_str = "All Models" if target == "ALL_MODELS" else target
        ax.set_title(f"Impact of Scheduling Policy on Average Exit Depth ({title_str})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Traffic Intensity λ (req/s)", fontsize=12)
        ax.set_ylabel("Avg early-exit depth", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        _set_x_ticks_40(ax, lambda_list)

        if max_depth > 0 and exit_points:
            ax.set_ylim(0.8, max_depth + 0.2)
            ax.set_yticks(list(range(1, max_depth + 1)))
            # Show exit labels on the y-axis for interpretability
            if len(exit_points) == max_depth:
                ax.set_yticklabels(exit_points)
        
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=14)
        fig.tight_layout()

        fname = f"avg_exit_compare_{target}.png"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Avg early-exit figure saved to: {out_path}")
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot multi-model intensity experiment results")
    parser.add_argument(
        "--lambdas",
        type=str,
        default="40,80,120,160,200",
        help="Comma-separated list of lambda values.",
    )
    parser.add_argument(
        "--schedulers",
        type=str,
        default="early_exit,all_early,all_final,all_final_round_robin,symphony,ours_normalized",
        help="Comma-separated list of schedulers to plot.",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs_baseline",
        help="Base directory for logs (e.g., 'logs' or 'logs_all_exits_3r50_2r101_1r152').",
    )
    parser.add_argument(
        "--slo-ms",
        type=float,
        default=15.0,
        help="SLO threshold in milliseconds.",
    )
    parser.add_argument(
        "--slo-quantile",
        type=str,
        default="p95_ms",
        help="SLO quantile key.",
    )
    parser.add_argument(
        "--warmup-tasks",
        type=int,
        default=100,
        help="Number of warmup tasks.",
    )
    args = parser.parse_args()

    schedulers = [s.strip() for s in args.schedulers.split(",") if s.strip()]
    lambda_list = [int(x) for x in args.lambdas.split(",") if x.strip()]
    slo_ms = args.slo_ms
    slo_quantile = args.slo_quantile
    warmup_tasks = args.warmup_tasks
    logs_base_dir = args.logs_dir
    
    # Derive figures_dir from logs_dir
    if "logs" in logs_base_dir:
        figures_base_dir = logs_base_dir.replace("logs", "figures")
    else:
        figures_base_dir = "figures_" + logs_base_dir
    
    # Read from logs directory
    results = {sch: {} for sch in schedulers}
    for scheduler in schedulers:
        for lam in lambda_list:
            diag_path = os.path.join(logs_base_dir, f"lam152_{lam}", f"multi_model_diag_{scheduler}_lam{lam}.pkl")
            if not os.path.exists(diag_path):
                print(f"[WARNING] Diag file not found: {diag_path}")
                continue
            metrics = compute_metrics_from_diag(diag_path)
            results[scheduler][lam] = metrics
    
    # Filter out schedulers with no results
    schedulers = [s for s in schedulers if results[s]]
    if not schedulers:
        print("[ERROR] No results found for any scheduler.")
        exit(1)
    
    # Save experiment data
    os.makedirs(logs_base_dir, exist_ok=True)
    exp_path = os.path.join(logs_base_dir, "multi_model_scheduler_experiment.pkl")
    with open(exp_path, "wb") as f:
        pickle.dump(
            {
                "lambda_list": lambda_list,
                "results": results,
                "slo_ms": slo_ms,
                "slo_quantile": slo_quantile,
                "warmup_tasks": warmup_tasks,
            },
            f,
        )
    print(f"Experiment data saved to: {exp_path}")
    
    # Plot results
    plot_results(lambda_list, results, out_dir=figures_base_dir)
