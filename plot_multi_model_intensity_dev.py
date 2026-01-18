import os,pickle
import re
import numpy as np
import matplotlib.pyplot as plt

from run_multi_model_intensity_dev import compute_metrics_from_diag

def _set_x_ticks_20(ax, lambda_list):
    if not lambda_list:
        return
    lam_min = int(np.floor(min(lambda_list) / 20.0) * 20)
    lam_max = int(np.ceil(max(lambda_list) / 20.0) * 20)
    if lam_max < lam_min:
        lam_max = lam_min
    ticks = np.arange(lam_min, lam_max + 1, 20)
    ax.set_xticks(ticks)

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
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set_title("Multi-model: p95 latency vs traffic intensity")
    ax1.set_xlabel("Traffic intensity λ per model (req/s)")
    ax1.set_ylabel("p95 total latency (ms)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    for scheduler in schedulers:
        p95_vals = []
        for lam in lambda_list:
            m = results[scheduler][lam]
            p95_vals.append(m["p95_ms"])

        lam_arr = np.array(lambda_list, dtype=np.float64)
        p95_arr = np.array(p95_vals, dtype=np.float64)

        ax1.plot(
            lam_arr,
            p95_arr,
            marker="o",
            linestyle="-",
            label=_legend_label(scheduler),
        )

    ax1.legend(loc="upper left")
    _set_x_ticks_20(ax1, lambda_list)
    fig1.tight_layout()
    out_path1 = os.path.join(out_dir, "main_multi_model_p95_latency_compare.png")
    plt.savefig(out_path1, dpi=150)
    print(f"p95 latency figure saved to: {out_path1}")
    plt.close(fig1)

    # Plot drop ratio
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.set_title("Multi-model: drop ratio vs traffic intensity")
    ax2.set_xlabel("Traffic intensity λ per model (req/s)")
    ax2.set_ylabel("Drop ratio")
    ax2.grid(True, linestyle="--", alpha=0.5)

    for scheduler in schedulers:
        drop_vals = []
        for lam in lambda_list:
            m = results[scheduler][lam]
            drop_vals.append(m["drop_ratio"])

        lam_arr = np.array(lambda_list, dtype=np.float64)
        drop_arr = np.array(drop_vals, dtype=np.float64)

        ax2.plot(
            lam_arr,
            drop_arr,
            marker="s",
            linestyle="--",
            label=_legend_label(scheduler),
        )

    ax2.legend(loc="upper left")
    ax2.set_ylim(0.0, 0.5)
    _set_x_ticks_20(ax2, lambda_list)
    fig2.tight_layout()
    out_path2 = os.path.join(out_dir, "main_multi_model_drop_ratio_compare.png")
    plt.savefig(out_path2, dpi=150)
    print(f"Drop ratio figure saved to: {out_path2}")
    plt.close(fig2)

    # Plot drop ratio without symphony for clearer comparison
    schedulers_temp = [s for s in schedulers if s != "symphony"]
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.set_title("Multi-model: drop ratio without symphony vs traffic intensity")
    ax3.set_xlabel("Traffic intensity λ per model (req/s)")
    ax3.set_ylabel("Drop ratio")
    ax3.grid(True, linestyle="--", alpha=0.5)
    for scheduler in schedulers_temp:
        drop_vals = []
        for lam in lambda_list:
            m = results[scheduler][lam]
            drop_vals.append(m["drop_ratio"])

        lam_arr = np.array(lambda_list, dtype=np.float64)
        drop_arr = np.array(drop_vals, dtype=np.float64)

        ax3.plot(
            lam_arr,
            drop_arr,
            marker="s",
            linestyle="--",
            label=_legend_label(scheduler),
        )

    ax3.legend(loc="upper left")
    ax3.set_ylim(0.0, 0.01)
    _set_x_ticks_20(ax3, lambda_list)
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

    lam_arr = np.array(lambda_list, dtype=np.float64)

    for target in targets:
        fig, ax = plt.subplots(figsize=(7, 3))

        for s in schedulers:
            vals = []
            for lam in lambda_list:
                metric = results[s].get(lam, {})
                if target == "ALL_MODELS":
                    vals.append(metric.get("avg_exit_all", float("nan")))
                else:
                    vals.append((metric.get("avg_exit_by_model") or {}).get(target, float("nan")))

            y = np.array(vals, dtype=np.float64)
            ax.plot(lam_arr, y, marker="o", label=_legend_label(s))

        title_str = "ALL_MODELS (weighted)" if target == "ALL_MODELS" else target
        ax.set_title(f"Avg early-exit depth vs intensity ({title_str})")
        ax.set_xlabel("Traffic intensity λ (req/s)")
        ax.set_ylabel("Avg early-exit depth")
        ax.grid(True, linestyle="--", alpha=0.5)
        _set_x_ticks_20(ax, lambda_list)

        if max_depth > 0 and exit_points:
            ax.set_ylim(0.8, max_depth + 0.2)
            ax.set_yticks(list(range(1, max_depth + 1)))
            # Show exit labels on the y-axis for interpretability
            if len(exit_points) == max_depth:
                ax.set_yticklabels(exit_points)
        
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        fig.tight_layout()

        fname = f"avg_exit_compare_{target}.png"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Avg early-exit figure saved to: {out_path}")
        plt.close(fig)

if __name__ == "__main__":
    schedulers = ["early_exit", "all_early", "all_final", "all_final_round_robin", "symphony", "ours_normalized"]
    lambda_list = [20,40,60,80,100,120,140,160,180,200]
    slo_ms = 20
    slo_quantile = 'p95_ms'
    warmup_tasks = 100
    
    results = {sch: {} for sch in schedulers}
    for scheduler in schedulers:
        for lam in lambda_list:
            diag_path = f"logs/lam152_{lam}/multi_model_diag_{scheduler}_lam{lam}.pkl"
            metrics = compute_metrics_from_diag(diag_path)
            results[scheduler][lam] = metrics
    
    
    # Save experiment data
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    exp_path = os.path.join(logs_dir, "multi_model_scheduler_experiment.pkl")
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
    
    # pickle_path = "logs/multi_model_scheduler_experiment.pkl"
    # with open(pickle_path, "rb") as f:
    #     data = pickle.load(f)
    # lambda_list = data.get("lambda_list", [])
    # results = data.get("results", {})
    plot_results(lambda_list, results, out_dir="figures")
    # plot_avg_exit_comparison(lambda_list, results, out_dir="figures")
