"""
plot_slo_ablation_comparison.py

Generate comparison plots for SLO ablation study.

This script loads results from logs_slo_ablation/ directory and creates
comparison plots showing how different SLO thresholds affect:
  - P95 latency vs traffic intensity
  - Violate ratio vs traffic intensity
  - Average exit depth vs traffic intensity
"""

import os
import pickle
import glob
import re
import numpy as np
import matplotlib.pyplot as plt


def load_slo_ablation_results(base_dir="logs_slo_ablation"):
    """
    Load all SLO ablation results from logs_slo_ablation/ directory.

    Returns
    -------
    dict
        {slo_ms: {lam: metrics}} structure
    """
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return {}

    results = {}

    # Find all slo_*ms_results.pkl files directly in base_dir
    pkl_files = glob.glob(os.path.join(base_dir, "slo_*ms_results.pkl"))

    if not pkl_files:
        print(f"No SLO results files found in {base_dir}")
        return {}

    for pkl_path in pkl_files:
        # Extract SLO value from filename (e.g., slo_10ms_results.pkl -> 10, slo_15p5ms_results.pkl -> 15.5)
        file_name = os.path.basename(pkl_path)
        match = re.search(r'slo_(\d+(?:p\d+)?)ms_results\.pkl', file_name)
        if not match:
            continue

        slo_label = match.group(1).replace("p", ".")
        try:
            slo_ms = float(slo_label)
        except ValueError:
            continue

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            results[slo_ms] = data["results"]
            print(f"Loaded results for SLO={slo_ms}ms from {pkl_path}")
        except Exception as e:
            print(f"Error loading {pkl_path}: {e}")

    return results


def load_baseline_results(base_dir="logs_baseline"):
    """
    Load baseline results (all exits, SLO=20ms).
    
    Returns
    -------
    dict
        {lam: metrics} structure
    """
    if not os.path.exists(base_dir):
        print(f"Baseline directory not found: {base_dir}")
        return {}
    
    results = {}
    
    # Find all lambda directories
    lam_dirs = glob.glob(os.path.join(base_dir, "lam152_*"))
    
    for lam_dir in lam_dirs:
        # Extract lambda value
        dir_name = os.path.basename(lam_dir)
        match = re.search(r'lam152_(\d+)', dir_name)
        if not match:
            continue
        
        try:
            lam = float(match.group(1))
        except ValueError:
            continue
        
        # Find the diagnostic file
        diag_files = glob.glob(os.path.join(lam_dir, "multi_model_diag_ours_normalized.pkl"))
        if not diag_files:
            continue
        
        diag_path = diag_files[0]
        
        try:
            with open(diag_path, "rb") as f:
                payload = pickle.load(f)
            
            # Compute metrics
            total_time_by_model_exit = payload["total_time_by_model_exit"]
            slo_violations_by_model = payload["slo_violations_by_model"]
            
            # Flatten total_times
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
            num_violated = sum(v["violations"] for v in slo_violations_by_model.values())
            total = num_completed + num_violated
            violate_ratio = float(num_violated) / total if total > 0 else 0.0
            
            # Average exit depth - use global absolute depth for fair comparison
            exit_points = payload.get("exit_points", ["layer1", "layer2", "layer3", "final"])
            GLOBAL_DEPTH_MAP = {"layer1": 1, "layer2": 2, "layer3": 3, "final": 4}
            depth_map = {e: GLOBAL_DEPTH_MAP.get(e, len(GLOBAL_DEPTH_MAP) + 1) for e in exit_points}
            
            total_all = 0
            sum_depth_all = 0.0
            for model_name, exit_dict in total_time_by_model_exit.items():
                for e in exit_points:
                    times = exit_dict.get(e, [])
                    c = len(times)
                    total_all += c
                    sum_depth_all += depth_map[e] * c
            
            avg_exit_all = (sum_depth_all / total_all) if total_all > 0 else float("nan")
            
            results[lam] = {
                "p95_ms": p95_ms,
                "violate_ratio": violate_ratio,
                "num_completed": num_completed,
                "num_violated": num_violated,
                "avg_exit_all": avg_exit_all,
            }
            
        except Exception as e:
            print(f"Error loading baseline {diag_path}: {e}")
    
    if results:
        print(f"Loaded baseline results (SLO=20ms, all exits) for {len(results)} lambda values")
    
    return results


def plot_comparison(results, output_dir="figures_slo_ablation"):
    """
    Generate comparison plots for different SLO values.
    
    Parameters
    ----------
    results : dict
        {slo_ms: {lam: metrics}} structure
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort SLO values for consistent plotting
    slo_values = sorted(results.keys())
    
    # Prepare data
    data = {}
    for slo_ms in slo_values:
        lambdas = []
        p95_list = []
        violate_list = []
        avg_exit_list = []
        
        for lam in sorted(results[slo_ms].keys()):
            metrics = results[slo_ms][lam]
            lambdas.append(lam)
            p95_list.append(metrics["p95_ms"])
            violate_list.append(metrics["violate_ratio"])
            avg_exit_list.append(metrics["avg_exit_all"])
        
        data[slo_ms] = {
            "lambdas": lambdas,
            "p95": p95_list,
            "violate": violate_list,
            "avg_exit": avg_exit_list,
        }

    # Collect all unique lambda values from data
    all_lambdas = set()
    for slo_ms in slo_values:
        all_lambdas.update(data[slo_ms]["lambdas"])
    lambda_ticks = sorted(all_lambdas)

    # Calculate x-axis limits with padding
    if lambda_ticks:
        lambda_min, lambda_max = min(lambda_ticks), max(lambda_ticks)
        padding = (lambda_max - lambda_min) * 0.05 if lambda_max > lambda_min else 1
        xlim_min, xlim_max = lambda_min - padding, lambda_max + padding
    else:
        xlim_min, xlim_max = 0, 100

    # Plot 1: P95 Latency
    plt.figure(figsize=(12, 5))
    numeric_keys = sorted([k for k in data.keys() if isinstance(k, (int, float))])
    for key in numeric_keys:
        label = f"SLO={key}ms"
        plt.plot(data[key]["lambdas"], data[key]["p95"], 
                marker='o', label=label, linestyle='-', 
                linewidth=2, alpha=1.0, markersize=6)
    
    # Add SLO reference lines
    for slo_ms in slo_values:
        if slo_ms <= 20:  # Only show reference lines within y-axis range
            plt.axhline(y=slo_ms, color='gray', linestyle=':', alpha=0.3, linewidth=1)
            if data[slo_ms]["lambdas"]:
                plt.text(max(data[slo_ms]["lambdas"]) * 0.95, slo_ms + 0.5, 
                        f'{slo_ms}ms', fontsize=9, color='gray', alpha=0.7)
    
    plt.xlabel("Traffic Intensity λ (req/s)", fontsize=16)
    plt.ylabel("P95 Total Latency (ms)", fontsize=16)
    plt.title("Impact of SLO Threshold on P95 Latency", fontsize=18, fontweight='bold')
    plt.xticks(lambda_ticks)
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(0, 70)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "slo_ablation_p95_latency_compare.png"), dpi=300)
    plt.close()
    
    # Plot 2: Violate Ratio
    plt.figure(figsize=(12, 5))
    numeric_keys = sorted([k for k in data.keys() if isinstance(k, (int, float))])
    for key in numeric_keys:
        label = f"SLO={key}ms"
        plt.plot(data[key]["lambdas"], data[key]["violate"], 
                marker='s', label=label, linestyle='-',
                linewidth=2, alpha=1.0, markersize=6)
    
    plt.xlabel("Traffic Intensity λ (req/s)", fontsize=16)
    plt.ylabel("Violate Ratio", fontsize=16)
    plt.title("Impact of SLO Threshold on Violate Ratio", fontsize=18, fontweight='bold')
    plt.xticks(lambda_ticks)
    plt.xlim(xlim_min, xlim_max)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "slo_ablation_violate_ratio_compare.png"), dpi=300)
    plt.close()
    
    # Plot 3: Average Exit Depth
    plt.figure(figsize=(12, 7))
    numeric_keys = sorted([k for k in data.keys() if isinstance(k, (int, float))])
    for key in numeric_keys:
        label = f"SLO={key}ms"
        plt.plot(data[key]["lambdas"], data[key]["avg_exit"], 
                marker='^', label=label, linestyle='-',
                linewidth=2, alpha=1.0, markersize=6)
    
    plt.xlabel("Traffic Intensity λ (req/s)", fontsize=16)
    plt.ylabel("Average Exit Depth", fontsize=16)
    plt.title("Impact of SLO Threshold on Average Exit Depth", fontsize=18, fontweight='bold')
    plt.xticks(lambda_ticks)
    plt.xlim(xlim_min, xlim_max)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "slo_ablation_avg_exit_depth_compare.png"), dpi=300)
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}")


def main():
    print("Loading SLO ablation results...")
    results = load_slo_ablation_results()
    
    if not results:
        print("No SLO ablation results found!")
        return
    
    print(f"\nFound results for SLO values: {sorted(results.keys())}")
    
    print("\nGenerating comparison plots...")
    plot_comparison(results)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
