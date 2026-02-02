"""
plot_model_combination_comparison.py

Generate comparison plots for model combination study.

This script loads results from logs_diff_model_combination/ directory and creates
comparison plots showing how different model combinations affect:
  - P95 latency vs traffic intensity
  - Drop ratio vs traffic intensity
  - Average exit depth vs traffic intensity
"""

import os
import pickle
import glob
import re
import numpy as np
import matplotlib.pyplot as plt


def load_model_combination_results(base_dir="logs_diff_model_combination"):
    """
    Load all model combination results from logs_diff_model_combination/ directory.

    Returns
    -------
    dict
        {combo_label: {lam: metrics}} structure
        where combo_label is like "1r50_1r101_1r152"
    """
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return {}

    results = {}

    # Find all combo_*_results.pkl files directly in base_dir
    pkl_files = glob.glob(os.path.join(base_dir, "combo_*_results.pkl"))

    if not pkl_files:
        print(f"No model combination results files found in {base_dir}")
        return {}

    for pkl_path in pkl_files:
        # Extract combination from filename (e.g., combo_1r50_2r101_1r152_results.pkl)
        file_name = os.path.basename(pkl_path)
        match = re.search(r'combo_(\d+r50_\d+r101_\d+r152)_results\.pkl', file_name)
        if not match:
            continue

        combo_label = match.group(1)

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            results[combo_label] = {
                "data": data["results"],
                "num_r50": data["num_r50"],
                "num_r101": data["num_r101"],
                "num_r152": data["num_r152"],
            }
            print(f"Loaded results for combination {combo_label} from {pkl_path}")
        except Exception as e:
            print(f"Error loading {pkl_path}: {e}")

    return results


def plot_comparison(results, output_dir="figures_diff_model_combination"):
    """
    Generate comparison plots for different model combinations.
    
    Parameters
    ----------
    results : dict
        {combo_label: {"data": {lam: metrics}, "num_r50": int, ...}} structure
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        print("No results to plot!")
        return
    
    # Prepare data
    data = {}
    combo_names = {}
    for combo_label, combo_info in results.items():
        # First extract all lambda152 values and corresponding metrics
        temp_data = []
        for lam_key in combo_info["data"].keys():
            metrics = combo_info["data"][lam_key]
            
            # Extract lambda152 from string like "120.0_80.0_40.0"
            # Use the rightmost value (lambda152)
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
        lambdas = [d['lam152'] for d in temp_data]
        p95_list = [d['p95'] for d in temp_data]
        drop_list = [d['drop'] for d in temp_data]
        avg_exit_list = [d['avg_exit'] for d in temp_data]
        
        data[combo_label] = {
            "lambdas": lambdas,
            "p95": p95_list,
            "drop": drop_list,
            "avg_exit": avg_exit_list,
        }
        
        # Create human-readable name
        num_r50 = combo_info["num_r50"]
        num_r101 = combo_info["num_r101"]
        num_r152 = combo_info["num_r152"]
        
        # Build name parts, ignoring models with 0 instances
        name_parts = []
        if num_r50 > 0:
            name_parts.append(f"{num_r50}×ResNet50")
        if num_r101 > 0:
            name_parts.append(f"{num_r101}×ResNet101")
        if num_r152 > 0:
            name_parts.append(f"{num_r152}×ResNet152")
        
        combo_names[combo_label] = " + ".join(name_parts)
    
    # Custom sorting: balanced (1-1-1) first, then mixed, then single model
    def combo_sort_key(combo_label):
        """
        Sort combos by priority:
        1. Balanced combinations (equal instances, like 1×R50 + 1×R101 + 1×R152)
        2. Mixed combinations (two or more model types)
        3. Single model combinations (only one model type, like 3×R152)
        """
        num_r50 = results[combo_label]["num_r50"]
        num_r101 = results[combo_label]["num_r101"]
        num_r152 = results[combo_label]["num_r152"]
        
        # Count how many model types are used
        non_zero = sum([1 for x in [num_r50, num_r101, num_r152] if x > 0])
        
        # Check if balanced (all equal and non-zero)
        if non_zero == 3 and num_r50 == num_r101 == num_r152:
            priority = 0  # Balanced - highest priority
        elif non_zero == 1:
            priority = 2  # Single model - lowest priority
        else:
            priority = 1  # Mixed - middle priority
        
        # Within same priority, sort by (num_r50, num_r101, num_r152)
        return (priority, num_r50, num_r101, num_r152)
    
    sorted_combos = sorted(data.keys(), key=combo_sort_key)

    # Collect all unique lambda values from data
    all_lambdas = set()
    for combo in sorted_combos:
        all_lambdas.update(data[combo]["lambdas"])
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
    for combo in sorted_combos:
        label = combo_names[combo]
        plt.plot(data[combo]["lambdas"], data[combo]["p95"],
                marker='o', label=label, linestyle='-',
                linewidth=2, alpha=1.0, markersize=6)

    plt.xlabel("Traffic Intensity λ (req/s)", fontsize=16)
    plt.ylabel("P95 Total Latency (ms)", fontsize=16)
    plt.title("Impact of Model Combination on P95 Latency", fontsize=18, fontweight='bold')
    plt.xticks(lambda_ticks)
    plt.xlim(xlim_min, xlim_max)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_combination_p95_latency_compare.png"), dpi=300)
    plt.close()
    
    # Plot 2: Drop Ratio
    plt.figure(figsize=(12, 6))
    for combo in sorted_combos:
        label = combo_names[combo]
        plt.plot(data[combo]["lambdas"], data[combo]["drop"],
                marker='s', label=label, linestyle='-',
                linewidth=2, alpha=1.0, markersize=6)

    plt.xlabel("Traffic Intensity λ (req/s)", fontsize=16)
    plt.ylabel("Drop Ratio", fontsize=16)
    plt.title("Impact of Model Combination on Drop Ratio", fontsize=18, fontweight='bold')
    plt.xticks(lambda_ticks)
    plt.xlim(xlim_min, xlim_max)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_combination_drop_ratio_compare.png"), dpi=300)
    plt.close()
    
    # Plot 3: Average Exit Depth
    plt.figure(figsize=(12, 6))
    for combo in sorted_combos:
        label = combo_names[combo]
        plt.plot(data[combo]["lambdas"], data[combo]["avg_exit"],
                marker='^', label=label, linestyle='-',
                linewidth=2, alpha=1.0, markersize=6)

    plt.xlabel("Traffic Intensity λ (req/s)", fontsize=16)
    plt.ylabel("Average Exit Depth", fontsize=16)
    plt.title("Impact of Model Combination on Average Exit Depth", fontsize=18, fontweight='bold')
    plt.xticks(lambda_ticks)
    plt.xlim(xlim_min, xlim_max)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_combination_avg_exit_depth_compare.png"), dpi=300)
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}")


def main():
    print("Loading model combination results...")
    results = load_model_combination_results()
    
    if not results:
        print("No model combination results found!")
        return
    
    print(f"\nFound results for combinations: {sorted(results.keys())}")
    
    print("\nGenerating comparison plots...")
    plot_comparison(results)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
