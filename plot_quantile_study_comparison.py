"""
plot_quantile_study_comparison.py

Generate comparison plots for quantile study.

This script loads results from logs_quantile_study/ directory and creates
comparison plots showing how different quantile_key settings affect:
  - P95 latency vs traffic intensity
  - Drop ratio vs traffic intensity
  - Average exit depth vs traffic intensity
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_quantile_study_results(base_dir="logs_quantile_study"):
    """
    Load quantile study results.
    
    Returns
    -------
    dict or None
        Results data structure
    """
    pkl_path = os.path.join(base_dir, "quantile_study_results.pkl")
    
    if not os.path.exists(pkl_path):
        print(f"Results file not found: {pkl_path}")
        return None
    
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded results from {pkl_path}")
        return data
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None


def plot_comparison(data, output_dir="figures_quantile_study"):
    """
    Generate comparison plots for different quantile keys.
    
    Parameters
    ----------
    data : dict
        Loaded results data
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not data:
        print("No data to plot!")
        return
    
    quantile_keys = data.get("quantile_keys", [])
    results = data.get("results", {})
    
    if not quantile_keys or not results:
        print("No results to plot!")
        return
    
    # Prepare data
    plot_data = {}
    
    for quantile in quantile_keys:
        if quantile not in results:
            continue
            
        quantile_results = results[quantile]
        
        # Extract lambda152 values and metrics
        temp_data = []
        for lam_key in quantile_results.keys():
            metrics = quantile_results[lam_key]
            
            # Extract lambda152 from string like "120.0_80.0_40.0"
            lam_parts = lam_key.split('_')
            lam152 = float(lam_parts[-1])
            
            temp_data.append({
                'lam152': lam152,
                'p95': metrics["p95_ms"],
                'drop': metrics["drop_ratio"],
                'avg_exit': metrics["avg_exit_all"]
            })
        
        # Sort by lambda152 value
        temp_data.sort(key=lambda x: x['lam152'])
        
        # Extract sorted lists
        plot_data[quantile] = {
            "lambdas": [d['lam152'] for d in temp_data],
            "p95": [d['p95'] for d in temp_data],
            "drop": [d['drop'] for d in temp_data],
            "avg_exit": [d['avg_exit'] for d in temp_data],
        }
    
    # Sort quantiles for consistent plotting
    sorted_quantiles = sorted(plot_data.keys())

    # Collect all unique lambda values from data
    all_lambdas = set()
    for quantile in sorted_quantiles:
        all_lambdas.update(plot_data[quantile]["lambdas"])
    lambda_ticks = sorted(all_lambdas)

    # Calculate x-axis limits with padding
    if lambda_ticks:
        lambda_min, lambda_max = min(lambda_ticks), max(lambda_ticks)
        padding = (lambda_max - lambda_min) * 0.05 if lambda_max > lambda_min else 1
        xlim_min, xlim_max = lambda_min - padding, lambda_max + padding
    else:
        xlim_min, xlim_max = 0, 100

    # Plot 1: P95 Latency
    plt.figure(figsize=(12, 6))
    for quantile in sorted_quantiles:
        label = quantile.replace("_ms", "").replace("_", " ").title()
        plt.plot(plot_data[quantile]["lambdas"], plot_data[quantile]["p95"], 
                marker='o', label=label, linestyle='-', 
                linewidth=2, alpha=0.8, markersize=6)
    
    plt.xlabel("Traffic Intensity λ (req/s)", fontsize=16)
    plt.ylabel("P95 Total Latency (ms)", fontsize=16)
    plt.title("Impact of Quantile Setting on P95 Latency", fontsize=18, fontweight='bold')
    plt.xticks(lambda_ticks)
    plt.xlim(xlim_min, xlim_max)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quantile_study_p95_latency_compare.png"), dpi=300)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'quantile_study_p95_latency_compare.png')}")
    
    # Plot 2: Drop Ratio
    plt.figure(figsize=(12, 6))
    for quantile in sorted_quantiles:
        label = quantile.replace("_ms", "").replace("_", " ").title()
        plt.plot(plot_data[quantile]["lambdas"], plot_data[quantile]["drop"], 
                marker='s', label=label, linestyle='-',
                linewidth=2, alpha=0.8, markersize=6)
    
    plt.xlabel("Traffic Intensity λ (req/s)", fontsize=16)
    plt.ylabel("Drop Ratio", fontsize=16)
    plt.title("Impact of Quantile Setting on Drop Ratio", fontsize=18, fontweight='bold')
    plt.xticks(lambda_ticks)
    plt.xlim(xlim_min, xlim_max)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quantile_study_drop_ratio_compare.png"), dpi=300)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'quantile_study_drop_ratio_compare.png')}")
    
    # Plot 3: Average Exit Depth
    plt.figure(figsize=(12, 6))
    for quantile in sorted_quantiles:
        label = quantile.replace("_ms", "").replace("_", " ").title()
        plt.plot(plot_data[quantile]["lambdas"], plot_data[quantile]["avg_exit"], 
                marker='^', label=label, linestyle='-',
                linewidth=2, alpha=0.8, markersize=6)
    
    plt.xlabel("Traffic Intensity λ (req/s)", fontsize=16)
    plt.ylabel("Average Exit Depth", fontsize=16)
    plt.title("Impact of Quantile Setting on Average Exit Depth", fontsize=18, fontweight='bold')
    plt.xticks(lambda_ticks)
    plt.xlim(xlim_min, xlim_max)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quantile_study_avg_exit_depth_compare.png"), dpi=300)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'quantile_study_avg_exit_depth_compare.png')}")
    
    print(f"\nAll plots saved to: {output_dir}")


def main():
    print("Loading quantile study results...")
    data = load_quantile_study_results()
    
    if not data:
        print("No quantile study results found!")
        return
    
    quantile_keys = data.get("quantile_keys", [])
    print(f"\nFound results for quantiles: {quantile_keys}")
    
    print("\nGenerating comparison plots...")
    plot_comparison(data)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
