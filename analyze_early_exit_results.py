"""
analyze_early_exit_results.py

Analyze and visualize the performance of early exit points for trained models.
"""

import os
import json
import argparse
from typing import Dict

import torch
import matplotlib.pyplot as plt
import numpy as np

from early_exit_resnets import EarlyExitResNet
from train_early_exit_resnets import (
    build_cifar100_early_exit_resnets,
    get_cifar100_loaders,
    evaluate
)


def load_checkpoint_and_evaluate(
    checkpoint_path: str,
    model: EarlyExitResNet,
    testloader,
    device: torch.device,
    exit_points: tuple,
) -> Dict[str, float]:
    """
    Load a checkpoint and evaluate the model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nLoaded checkpoint from: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")

    # Evaluate
    accuracies = evaluate(model, testloader, device, exit_points)

    return accuracies


def plot_exit_accuracies(
    results: Dict[str, Dict[str, float]],
    save_path: str = "./results/exit_accuracies.png"
):
    """
    Plot bar chart comparing accuracies across models and exit points.
    """
    exit_points = list(next(iter(results.values())).keys())
    models = list(results.keys())

    x = np.arange(len(exit_points))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_name in enumerate(models):
        accuracies = [results[model_name][exit_id] for exit_id in exit_points]
        ax.bar(x + i * width, accuracies, width, label=model_name)

    ax.set_xlabel('Exit Point', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Early Exit Point Accuracies on CIFAR-10', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(exit_points)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved accuracy plot to: {save_path}")
    plt.close()


def plot_accuracy_vs_params(
    results: Dict[str, Dict[str, float]],
    models_dict: Dict[str, EarlyExitResNet],
    save_path: str = "./results/accuracy_vs_params.png"
):
    """
    Plot accuracy vs number of parameters for each exit point.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    exit_points = list(next(iter(results.values())).keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(models_dict)))

    for (model_name, accuracies), color in zip(results.items(), colors):
        if model_name in models_dict:
            model = models_dict[model_name]
            params = []
            accs = []

            for exit_id in exit_points:
                param_count = model.param_count_for_exit(exit_id) / 1e6  # in millions
                params.append(param_count)
                accs.append(accuracies[exit_id])

            ax.plot(params, accs, 'o-', label=model_name, color=color, linewidth=2, markersize=8)

            # Annotate each point with exit name
            for i, exit_id in enumerate(exit_points):
                ax.annotate(exit_id, (params[i], accs[i]),
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=8, alpha=0.7)

    ax.set_xlabel('Parameters (Millions)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Model Size for Early Exits', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy vs params plot to: {save_path}")
    plt.close()


def print_results_table(results: Dict[str, Dict[str, float]]):
    """
    Print a formatted table of results.
    """
    print("\n" + "="*80)
    print("EARLY EXIT PERFORMANCE SUMMARY")
    print("="*80)

    exit_points = list(next(iter(results.values())).keys())

    # Header
    header = f"{'Model':<15}"
    for exit_id in exit_points:
        header += f"{exit_id:>12}"
    print(header)
    print("-" * 80)

    # Data rows
    for model_name, accuracies in results.items():
        row = f"{model_name:<15}"
        for exit_id in exit_points:
            row += f"{accuracies[exit_id]:>11.2f}%"
        print(row)

    print("="*80)


def save_detailed_results(
    results: Dict[str, Dict[str, float]],
    models_dict: Dict[str, EarlyExitResNet],
    save_path: str = "./results/detailed_results.json"
):
    """
    Save detailed results including parameter counts.
    """
    detailed_results = {}

    for model_name, accuracies in results.items():
        if model_name in models_dict:
            model = models_dict[model_name]
            detailed_results[model_name] = {}

            for exit_id, accuracy in accuracies.items():
                param_count = model.param_count_for_exit(exit_id)
                detailed_results[model_name][exit_id] = {
                    "accuracy": accuracy,
                    "params": param_count,
                    "params_millions": param_count / 1e6
                }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print(f"\nSaved detailed results to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze EarlyExitResNet results')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='directory containing checkpoints')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='dataloader workers')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['resnet50', 'resnet101', 'resnet152'],
                       choices=['resnet50', 'resnet101', 'resnet152'],
                       help='which models to analyze')

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    print("\nLoading CIFAR-100 test dataset...")
    _, testloader = get_cifar100_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Build models
    exit_points = ("layer1", "layer2", "layer3", "final")
    m50, m101, m152 = build_cifar100_early_exit_resnets(
        device,
        exit_points=exit_points,
        use_pretrained=False
    )

    models_dict = {
        'ResNet50': m50,
        'ResNet101': m101,
        'ResNet152': m152
    }

    # Filter based on args
    models_to_analyze = {}
    if 'resnet50' in args.models:
        models_to_analyze['ResNet50'] = m50
    if 'resnet101' in args.models:
        models_to_analyze['ResNet101'] = m101
    if 'resnet152' in args.models:
        models_to_analyze['ResNet152'] = m152

    # Evaluate each model
    results = {}

    for model_name, model in models_to_analyze.items():
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{model_name}_best.pth")

        if not os.path.exists(checkpoint_path):
            print(f"\nWarning: Checkpoint not found for {model_name}: {checkpoint_path}")
            print("Skipping this model...")
            continue

        accuracies = load_checkpoint_and_evaluate(
            checkpoint_path, model, testloader, device, exit_points
        )

        results[model_name] = accuracies

        print(f"\n{model_name} - Exit Point Accuracies:")
        for exit_id, acc in accuracies.items():
            param_count = model.param_count_for_exit(exit_id) / 1e6
            print(f"  {exit_id:8s}: {acc:6.2f}%  ({param_count:7.2f}M params)")

    if not results:
        print("\nNo results to analyze. Please train models first.")
        return

    # Print summary table
    print_results_table(results)

    # Save detailed results
    save_detailed_results(results, models_to_analyze)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_exit_accuracies(results)
    plot_accuracy_vs_params(results, models_to_analyze)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
