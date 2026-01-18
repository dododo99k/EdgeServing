"""
plot_early_exit_boxplots.py

Load profiling data produced by early_exit_resnet_bench.py and visualize
latency distributions using boxplots.

For each model:
  - x-axis: batch size (1..10)
  - at each batch size: 5 boxplots (one per early-exit: "stem","layer1","layer2","layer3","final")
    all vertically aligned at the same x (batch size)
  - y-axis: latency (ms)
  - for each early-exit we overlay a line (curve) of mean latency vs batch size

Assumes profiling pickles are named:
  early_exit_latency_ResNet50.pkl
  early_exit_latency_ResNet101.pkl
  early_exit_latency_ResNet152.pkl
"""

import os
import pickle
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


def load_model_profile(
    model_name: str,
    pickle_dir: str = ".",
) -> Dict[str, Any]:
    """
    Load profiling payload for a given model.

    Expected file name:
        early_exit_latency_<model_name>.pkl

    Returns the payload dict with keys:
        - device
        - batch_sizes
        - warmup
        - iters
        - exit_points
        - model_name
        - results: dict[exit_id][batch_size] -> stats dict
    """
    fname = f"early_exit_latency_{model_name}.pkl"
    path = os.path.join(pickle_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Profiling file not found: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    return payload


def plot_boxplots_for_model(
    model_name: str,
    pickle_dir: str = ".",
    batch_sizes: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot latency boxplots for a single model.

    - x-axis: batch size
    - at each batch size: multiple colored boxplots (one per early-exit),
      all centered at that batch size (no horizontal offsets)
    - y-axis: latency (ms)
    - additionally, plot mean latency vs batch size for each early-exit as a line.

    Parameters:
      model_name : e.g. "ResNet50"
      pickle_dir : directory containing pickle files
      batch_sizes : which batch sizes to plot; if None, use all in the file
      save_path : if given, save the figure to this path
      show      : if True, call plt.show() at the end
    """
    payload = load_model_profile(model_name, pickle_dir=pickle_dir)
    recorded_batch_sizes = payload["batch_sizes"]
    exit_points = payload["exit_points"]
    results = payload["results"]  # exit_id -> batch_size -> stats

    if batch_sizes is None:
        batch_sizes = recorded_batch_sizes
    else:
        batch_sizes = [bs for bs in batch_sizes if bs in recorded_batch_sizes]

    if not batch_sizes:
        raise ValueError("No valid batch sizes to plot.")

    n_bs = len(batch_sizes)
    n_exits = len(exit_points)

    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * n_bs, 6))

    box_width = 0.5  # all boxes share same x; width controls overlap
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_exits, 3)))

    # For legend: we will use the mean lines
    legend_handles = []
    legend_labels = []

    for ei, exit_id in enumerate(exit_points):
        color = colors[ei % len(colors)]

        means = []
        bs_with_data = []

        for bs in batch_sizes:
            stats = results.get(exit_id, {}).get(bs, None)
            if stats is None:
                continue

            arr = np.array(stats["latencies_ms"])
            if arr.size == 0:
                continue

            # Boxplot centered at x = bs
            bp = ax.boxplot(
                arr,
                positions=[bs],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
            )

            # Color box + median
            for box in bp["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.35)
                box.set_edgecolor(color)
            for median in bp["medians"]:
                median.set_color(color)
                median.set_linewidth(1.5)

            # For line plot
            means.append(arr.mean())
            bs_with_data.append(bs)

        if bs_with_data:
            line, = ax.plot(
                bs_with_data,
                means,
                color=color,
                marker="o",
                linewidth=1.5,
            )
            legend_handles.append(line)
            legend_labels.append(exit_id)

    # Axes labels and ticks
    ax.set_title(f"{model_name} latency by batch size and early-exit")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])

    if legend_handles:
        ax.legend(legend_handles, legend_labels, title="Exit", loc="upper left")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_all_models(
    model_names: List[str] = None,
    pickle_dir: str = ".",
    batch_sizes: Optional[List[int]] = None,
    save_dir: Optional[str] = None,
):
    """
    Convenience function to plot boxplots for multiple models.

    Parameters:
      model_names : list of model names, e.g. ["ResNet50","ResNet101","ResNet152"]
      pickle_dir  : directory with pickle files
      batch_sizes : which batch sizes to plot (subset of profiled ones)
      save_dir    : if given, saves each figure as:
                    <save_dir>/boxplot_<model_name>.png
    """
    if model_names is None:
        model_names = ["ResNet50", "ResNet101", "ResNet152"]

    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for model_name in model_names:
        save_path = None
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"boxplot_{model_name}.png")

        print(f"Plotting boxplots for {model_name}...")
        plot_boxplots_for_model(
            model_name=model_name,
            pickle_dir=pickle_dir,
            batch_sizes=batch_sizes,
            save_path=save_path,
            show=(save_dir is None),
        )


if __name__ == "__main__":
    # Example usage:
    #   1) Run early_exit_resnet_bench.py to generate the pickle files.
    #   2) Run this script:
    #
    #      python plot_early_exit_boxplots.py
    #
    # It will show one figure per model.
    plot_all_models(
        model_names=["ResNet50", "ResNet101", "ResNet152"],
        pickle_dir="saves",
        batch_sizes=None,   # or e.g. [1, 4, 8]
        save_dir="figures",      # or a directory path to save PNGs
    )
