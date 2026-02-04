"""
train_early_exit_resnets.py

Train EarlyExitResNet50/101/152 on CIFAR-100 dataset and save performance metrics
for each exit point.
"""

import os
import json
import time
from typing import Dict, List, Tuple
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import (
    resnet50, resnet101, resnet152,
    ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
)

from early_exit_resnets import EarlyExitResNet


def get_cifar100_loaders(batch_size: int = 128, num_workers: int = 4):
    """
    Get CIFAR-100 train and test data loaders with standard augmentation.
    """
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # No augmentation for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return trainloader, testloader


def adapt_resnet_for_cifar100(base_model: nn.Module) -> nn.Module:
    """
    Adapt a ResNet model for CIFAR-100:
    - Change first conv layer from 7x7 stride 2 to 3x3 stride 1
    - Remove maxpool layer
    - Change fc layer output to 100 classes
    """
    # Replace first conv: 7x7 kernel, stride 2 -> 3x3 kernel, stride 1
    base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Replace maxpool with identity (we'll handle this in forward)
    base_model.maxpool = nn.Identity()

    # Change final fc layer to 100 classes (CIFAR-100)
    in_features = base_model.fc.in_features
    base_model.fc = nn.Linear(in_features, 100)

    return base_model


def build_cifar100_early_exit_resnets(
    device: torch.device,
    exit_points: Tuple[str, ...] = ("layer1", "layer2", "layer3", "final"),
    use_pretrained: bool = False,
) -> Tuple[EarlyExitResNet, EarlyExitResNet, EarlyExitResNet]:
    """
    Build EarlyExitResNet models adapted for CIFAR-100.
    """
    print("Building ResNet50/101/152 for CIFAR-100 with early exits...")

    if use_pretrained:
        base50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        base101 = resnet101(weights=ResNet101_Weights.DEFAULT)
        base152 = resnet152(weights=ResNet152_Weights.DEFAULT)
    else:
        base50 = resnet50(weights=None)
        base101 = resnet101(weights=None)
        base152 = resnet152(weights=None)

    # Adapt for CIFAR-100
    base50 = adapt_resnet_for_cifar100(base50)
    base101 = adapt_resnet_for_cifar100(base101)
    base152 = adapt_resnet_for_cifar100(base152)

    # Wrap with early exit heads (num_classes will be 10 after adaptation)
    m50 = EarlyExitResNet(base50, exit_points=exit_points).to(device)
    m101 = EarlyExitResNet(base101, exit_points=exit_points).to(device)
    m152 = EarlyExitResNet(base152, exit_points=exit_points).to(device)

    return m50, m101, m152


def train_epoch(
    model: EarlyExitResNet,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    exit_points: Tuple[str, ...],
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch with joint loss from all exit points.
    Returns dict of losses per exit point.
    """
    model.train()

    exit_losses = {exit_id: 0.0 for exit_id in exit_points}
    total_samples = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Forward through all exits
        outputs = model(inputs, return_all=True)

        # Compute loss for each exit
        total_loss = 0.0
        for exit_id in exit_points:
            if exit_id in outputs:
                loss = criterion(outputs[exit_id], targets)
                exit_losses[exit_id] += loss.item() * inputs.size(0)
                total_loss += loss

        # Backward and optimize
        total_loss.backward()
        optimizer.step()

        total_samples += inputs.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch [{batch_idx+1}/{len(trainloader)}], "
                  f"Total Loss: {total_loss.item():.4f}")

    # Average losses
    for exit_id in exit_losses:
        exit_losses[exit_id] /= total_samples

    return exit_losses


def evaluate(
    model: EarlyExitResNet,
    testloader: DataLoader,
    device: torch.device,
    exit_points: Tuple[str, ...],
) -> Dict[str, float]:
    """
    Evaluate accuracy for each exit point.
    Returns dict of accuracies per exit point.
    """
    model.eval()

    exit_correct = {exit_id: 0 for exit_id in exit_points}
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Get predictions from all exits
            outputs = model(inputs, return_all=True)

            for exit_id in exit_points:
                if exit_id in outputs:
                    _, predicted = outputs[exit_id].max(1)
                    exit_correct[exit_id] += predicted.eq(targets).sum().item()

            total_samples += targets.size(0)

    # Calculate accuracies
    exit_accuracies = {}
    for exit_id in exit_points:
        exit_accuracies[exit_id] = 100.0 * exit_correct[exit_id] / total_samples

    return exit_accuracies


def train_model(
    model_name: str,
    model: EarlyExitResNet,
    trainloader: DataLoader,
    testloader: DataLoader,
    device: torch.device,
    exit_points: Tuple[str, ...],
    epochs: int = 100,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    save_dir: str = "./checkpoints",
):
    """
    Train a single EarlyExitResNet model.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # Learning rate scheduler: decay at 50% and 75% of total epochs
    milestones = [int(epochs * 0.5), int(epochs * 0.75)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # Track training history
    history = {
        "train_losses": [],
        "test_accuracies": [],
        "best_accuracies": {exit_id: 0.0 for exit_id in exit_points},
    }

    best_final_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{epochs}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Train
        train_losses = train_epoch(
            model, trainloader, criterion, optimizer, device, exit_points, epoch
        )

        # Evaluate
        test_accs = evaluate(model, testloader, device, exit_points)

        epoch_time = time.time() - epoch_start

        # Print results
        print(f"\nEpoch {epoch+1} completed in {epoch_time:.2f}s")
        print("Train Losses:")
        for exit_id, loss in train_losses.items():
            print(f"  {exit_id:8s}: {loss:.4f}")
        print("Test Accuracies:")
        for exit_id, acc in test_accs.items():
            print(f"  {exit_id:8s}: {acc:.2f}%")
            if acc > history["best_accuracies"][exit_id]:
                history["best_accuracies"][exit_id] = acc

        # Save history
        history["train_losses"].append(train_losses)
        history["test_accuracies"].append(test_accs)

        # Save best model based on final exit accuracy
        if test_accs["final"] > best_final_acc:
            best_final_acc = test_accs["final"]
            checkpoint_path = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracies': test_accs,
                'best_accuracies': history["best_accuracies"],
            }, checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")

        scheduler.step()

    # Save final checkpoint
    final_checkpoint_path = os.path.join(save_dir, f"{model_name}_final.pth")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracies': test_accs,
        'history': history,
    }, final_checkpoint_path)
    print(f"\nSaved final checkpoint to {final_checkpoint_path}")

    # Final evaluation
    print(f"\n{model_name} - Best Test Accuracies:")
    for exit_id, acc in history["best_accuracies"].items():
        print(f"  {exit_id:8s}: {acc:.2f}%")

    return history


def save_results_summary(
    results: Dict[str, Dict],
    save_path: str = "./results/early_exit_results.json"
):
    """
    Save training results summary to JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert to JSON-serializable format
    json_results = {}
    for model_name, history in results.items():
        json_results[model_name] = {
            "best_accuracies": history["best_accuracies"],
            "final_accuracies": history["test_accuracies"][-1],
        }

    with open(save_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nSaved results summary to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train EarlyExitResNets on CIFAR-10')
    parser.add_argument('--batch-size', type=int, default=128, help='training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--num-workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='checkpoint directory')
    parser.add_argument('--models', type=str, nargs='+', default=['resnet50', 'resnet101', 'resnet152'],
                        choices=['resnet50', 'resnet101', 'resnet152'],
                        help='which models to train')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='use ImageNet pretrained weights as initialization')

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data loaders
    print("\nLoading CIFAR-100 dataset...")
    trainloader, testloader = get_cifar100_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Train samples: {len(trainloader.dataset)}")
    print(f"Test samples: {len(testloader.dataset)}")

    # Build models
    exit_points = ("layer1", "layer2", "layer3", "final")
    m50, m101, m152 = build_cifar100_early_exit_resnets(
        device,
        exit_points=exit_points,
        use_pretrained=args.use_pretrained
    )

    models = {}
    if 'resnet50' in args.models:
        models['ResNet50'] = m50
    if 'resnet101' in args.models:
        models['ResNet101'] = m101
    if 'resnet152' in args.models:
        models['ResNet152'] = m152

    # Training parameters
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Momentum: {args.momentum}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Models: {list(models.keys())}")
    print(f"  Exit Points: {exit_points}")
    print(f"  Use Pretrained: {args.use_pretrained}")

    # Train each model
    all_results = {}

    for model_name, model in models.items():
        history = train_model(
            model_name=model_name,
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            device=device,
            exit_points=exit_points,
            epochs=args.epochs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            save_dir=args.save_dir,
        )
        all_results[model_name] = history

    # Save results summary
    save_results_summary(all_results)

    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED - FINAL SUMMARY")
    print("="*60)
    for model_name, history in all_results.items():
        print(f"\n{model_name}:")
        print("  Best Accuracies:")
        for exit_id, acc in history["best_accuracies"].items():
            print(f"    {exit_id:8s}: {acc:.2f}%")


if __name__ == "__main__":
    main()
