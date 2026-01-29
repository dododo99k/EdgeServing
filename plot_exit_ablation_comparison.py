"""
plot_exit_ablation_comparison.py

Compare ours_normalized performance with different exit point configurations.

Reads from:
  - logs_layer*_final/ (exit ablation experiments)
  - logs_all_exits_3r50_2r101_1r152/ (baseline with all exits)

Generates comparison plots for p95 latency, drop ratio, and average exit depth.
"""

import os
import pickle
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

from run_ours_normalized_exit_ablation import compute_metrics_from_diag


def _set_x_ticks_40(ax, lambda_list):
    """Set x-axis ticks at intervals of 40."""
    if not lambda_list:
        return
    data_min = min(lambda_list)
    data_max = max(lambda_list)
    tick_min = int(np.floor(data_min / 40.0) * 40)
    tick_max = int(np.ceil(data_max / 40.0) * 40)
    if tick_max < tick_min:
        tick_max = tick_min
    ticks = np.arange(tick_min, tick_max + 1, 40)
    ax.set_xticks(ticks)
    ax.set_xlim(data_min - 10, data_max + 10)


def load_exit_ablation_results(logs_dir="logs_exit_point_ablation"):
    """
    Load all exit ablation results from logs_exit_point_ablation/ directory.

    Structure: logs_exit_point_ablation/lam152_*/multi_model_diag_ours_normalized_exits_*.pkl

    Returns
    -------
    dict
        {exit_config_label: {lam: metrics}}
    """
    results = {}

    if not os.path.exists(logs_dir):
        print(f"Directory not found: {logs_dir}")
        return results

    # Find all lambda subdirectories
    lam_dirs = glob.glob(os.path.join(logs_dir, "lam152_*"))

    for lam_dir in lam_dirs:
        if not os.path.isdir(lam_dir):
            continue

        # Extract lambda value
        lam_match = re.search(r"lam152_(\d+(?:\.\d+)?)", lam_dir)
        if not lam_match:
            continue
        lam = float(lam_match.group(1))

        # Find all exit ablation pkl files
        pkl_files = glob.glob(os.path.join(lam_dir, "*_exits_*.pkl"))

        for pkl_file in pkl_files:
            # Extract exit config from filename
            # e.g., "multi_model_diag_ours_normalized_exits_layer1_final.pkl" -> "layer1_final"
            fname = os.path.basename(pkl_file)
            exit_match = re.search(r"_exits_(.+)\.pkl$", fname)
            if not exit_match:
                continue
            exit_label = exit_match.group(1)

            try:
                metrics = compute_metrics_from_diag(pkl_file)

                if exit_label not in results:
                    results[exit_label] = {}
                results[exit_label][lam] = metrics
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
                continue

    for exit_label in results:
        print(f"Loaded {len(results[exit_label])} lambda points for {exit_label}")
    
    return results


def load_baseline_results(logs_dir="logs_all_exits_3r50_2r101_1r152", scheduler="ours_normalized"):
    """
    Load baseline results (all exits) from standard logs directory.
    
    Returns
    -------
    dict
        {lam: metrics}
    """
    results = {}
    
    if not os.path.exists(logs_dir):
        print(f"Warning: Baseline directory not found: {logs_dir}")
        return results
    
    # Find all lambda subdirectories
    lam_dirs = glob.glob(os.path.join(logs_dir, "lam152_*"))
    
    for lam_dir in lam_dirs:
        # Extract lambda value
        lam_match = re.search(r"lam152_(\d+(?:\.\d+)?)", lam_dir)
        if not lam_match:
            continue
        lam = float(lam_match.group(1))
        
        # Find the diagnostic pickle file for the scheduler
        pkl_name = f"multi_model_diag_{scheduler}_lam{int(lam)}.pkl"
        diag_path = os.path.join(lam_dir, pkl_name)
        
        if not os.path.exists(diag_path):
            print(f"Warning: Baseline file not found: {diag_path}")
            continue
        
        try:
            metrics = compute_metrics_from_diag(diag_path)
            results[lam] = metrics
        except Exception as e:
            print(f"Error loading {diag_path}: {e}")
            continue
    
    print(f"Loaded {len(results)} lambda points for baseline (all exits)")
    return results


def plot_comparison(results_dict, lambda_list, out_dir="figures_exit_point_ablation"):
    """
    Generate comparison plots for different exit configurations.
    
    Parameters
    ----------
    results_dict : dict
        {config_label: {lam: metrics}}
    lambda_list : list
        List of lambda values to plot
    out_dir : str
        Output directory for figures
    """
    os.makedirs(out_dir, exist_ok=True)
    
    configs = sorted(results_dict.keys())
    
    if not configs:
        print("No configurations to plot!")
        return
    
    # Prepare data
    data = {}
    for config in configs:
        lambdas = []
        p95_vals = []
        drop_vals = []
        avg_exit_vals = []
        
        for lam in lambda_list:
            if lam in results_dict[config]:
                m = results_dict[config][lam]
                lambdas.append(lam)
                p95_vals.append(m["p95_ms"])
                drop_vals.append(m["drop_ratio"])
                avg_exit_vals.append(m["avg_exit_all"])
        
        data[config] = {
            "lambdas": np.array(lambdas),
            "p95": np.array(p95_vals),
            "drop": np.array(drop_vals),
            "avg_exit": np.array(avg_exit_vals),
        }
    
    # Sort configs for consistent legend ordering
    # Put "all_exits" or similar baseline last
    def sort_key(cfg):
        if "all" in cfg.lower() or cfg.startswith("layer1_layer2_layer3"):
            return (1, cfg)
        return (0, cfg)
    
    configs_sorted = sorted(configs, key=sort_key)
    
    # Plot 1: P95 Latency
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title("Impact of Exit Point Configuration on P95 Latency", fontsize=18, fontweight='bold')
    ax1.set_xlabel("Traffic Intensity λ (req/s)", fontsize=16)
    ax1.set_ylabel("P95 total latency (ms)", fontsize=16)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    for config in configs_sorted:
        if len(data[config]["lambdas"]) == 0:
            continue
        label = config.replace("_", "+")
        if config == "all_exits_baseline":
            label = "all exits"
        ax1.plot(
            data[config]["lambdas"],
            data[config]["p95"],
            marker="o",
            linestyle="-",
            label=label,
            linewidth=2,
        )
    
    ax1.legend(loc="best", fontsize=14)
    _set_x_ticks_40(ax1, lambda_list)
    fig1.tight_layout()
    out_path1 = os.path.join(out_dir, "exit_ablation_p95_latency_compare.png")
    plt.savefig(out_path1, dpi=300)
    print(f"P95 latency figure saved to: {out_path1}")
    plt.close(fig1)
    
    # Plot 2: Drop Ratio
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.set_title("Impact of Exit Point Configuration on Drop Ratio", fontsize=18, fontweight='bold')
    ax2.set_xlabel("Traffic Intensity λ (req/s)", fontsize=16)
    ax2.set_ylabel("Drop ratio", fontsize=16)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    for config in configs_sorted:
        if len(data[config]["lambdas"]) == 0:
            continue
        label = config.replace("_", "+")
        if config == "all_exits_baseline":
            label = "all exits"
        ax2.plot(
            data[config]["lambdas"],
            data[config]["drop"],
            marker="s",
            linestyle="-",
            label=label,
            linewidth=2,
        )
    
    ax2.legend(loc="best", fontsize=14)
    ax2.set_ylim(0.0, 0.003)
    _set_x_ticks_40(ax2, lambda_list)
    fig2.tight_layout()
    out_path2 = os.path.join(out_dir, "exit_ablation_drop_ratio_compare.png")
    plt.savefig(out_path2, dpi=300)
    print(f"Drop ratio figure saved to: {out_path2}")
    plt.close(fig2)
    
    # Plot 3: Average Exit Depth
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.set_title("Impact of Exit Point Configuration on Average Exit Depth", fontsize=18, fontweight='bold')
    ax3.set_xlabel("Traffic Intensity λ (req/s)", fontsize=16)
    ax3.set_ylabel("Average exit depth", fontsize=16)
    ax3.grid(True, linestyle="--", alpha=0.5)
    
    for config in configs_sorted:
        if len(data[config]["lambdas"]) == 0:
            continue
        label = config.replace("_", "+")
        if config == "all_exits_baseline":
            label = "all exits"
        ax3.plot(
            data[config]["lambdas"],
            data[config]["avg_exit"],
            marker="^",
            linestyle="-",
            label=label,
            linewidth=2,
        )
    
    ax3.legend(loc="best", fontsize=8)
    _set_x_ticks_40(ax3, lambda_list)
    fig3.tight_layout()
    out_path3 = os.path.join(out_dir, "exit_ablation_avg_exit_depth_compare.png")
    plt.savefig(out_path3, dpi=300)
    print(f"Average exit depth figure saved to: {out_path3}")
    plt.close(fig3)


def main():
    print("="*60)
    print("Exit Point Ablation Comparison Plotter")
    print("="*60)
    
    # Load exit ablation results
    print("\nLoading exit ablation results from logs_exit_point_ablation/ directory...")
    ablation_results = load_exit_ablation_results("logs_exit_point_ablation")
    
    # Load baseline results (all exits)
    print("\nLoading baseline results (all exits)...")
    baseline_results = load_baseline_results("logs_all_exits_3r50_2r101_1r152", "ours_normalized")
    
    # Combine results
    all_results = ablation_results.copy()
    if baseline_results:
        all_results["all_exits_baseline"] = baseline_results
    
    if not all_results:
        print("\nError: No results loaded!")
        return
    
    # Determine lambda list from available data
    all_lambdas = set()
    for config_results in all_results.values():
        all_lambdas.update(config_results.keys())
    
    lambda_list = sorted(all_lambdas)
    
    print(f"\nFound {len(all_results)} configurations")
    print(f"Lambda values: {lambda_list}")
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison(all_results, lambda_list, out_dir="figures_exit_point_ablation")
    
    # Save combined results
    output_pkl = "exit_ablation_comparison_results.pkl"
    with open(output_pkl, "wb") as f:
        pickle.dump({
            "lambda_list": lambda_list,
            "results": all_results,
        }, f)
    print(f"\nCombined results saved to: {output_pkl}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
