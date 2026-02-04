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


def train_backbone_only(
    model: EarlyExitResNet,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Train only the backbone and final classifier (no early exits).
    """
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Only use final exit
        outputs = model(inputs, exit_id="final")
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch [{batch_idx+1}/{len(trainloader)}], Loss: {loss.item():.4f}")

    return total_loss / total_samples


def train_early_exits_only(
    model: EarlyExitResNet,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    exit_points: Tuple[str, ...],
    epoch: int,
) -> Dict[str, float]:
    """
    Train only early exit heads with frozen backbone.
    """
    model.train()

    # Freeze backbone
    for param in model.base.parameters():
        param.requires_grad = False

    # Unfreeze exit heads
    for param in model.exit_heads.parameters():
        param.requires_grad = True

    exit_losses = {exit_id: 0.0 for exit_id in exit_points if exit_id != "final"}
    total_samples = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Forward through all exits
        outputs = model(inputs, return_all=True)

        # Compute loss only for early exits (not final)
        total_loss = 0.0
        for exit_id in exit_points:
            if exit_id != "final" and exit_id in outputs:
                loss = criterion(outputs[exit_id], targets)
                exit_losses[exit_id] += loss.item() * inputs.size(0)
                total_loss += loss

        if total_loss > 0:
            total_loss.backward()
            optimizer.step()

        total_samples += inputs.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch [{batch_idx+1}/{len(trainloader)}], Total Loss: {total_loss.item():.4f}")

    # Average losses
    for exit_id in exit_losses:
        exit_losses[exit_id] /= total_samples

    return exit_losses


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
    stage1_epochs: int = None,
):
    """
    Train a single EarlyExitResNet model in two stages:
    Stage 1: Train backbone + final classifier only
    Stage 2: Freeze backbone, train early exit heads only
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} (Two-Stage Training)")
    print(f"{'='*60}")

    os.makedirs(save_dir, exist_ok=True)

    # If stage1_epochs not specified, use 60% of total epochs for stage 1
    if stage1_epochs is None:
        stage1_epochs = int(epochs * 0.6)
    stage2_epochs = epochs - stage1_epochs

    print(f"\nStage 1 (Backbone + Final): {stage1_epochs} epochs")
    print(f"Stage 2 (Early Exits): {stage2_epochs} epochs\n")

    criterion = nn.CrossEntropyLoss()

    # =========================
    # STAGE 1: Train backbone + final classifier
    # =========================
    print(f"\n{'='*60}")
    print(f"STAGE 1: Training Backbone + Final Classifier")
    print(f"{'='*60}")

    optimizer_stage1 = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    milestones_stage1 = [int(stage1_epochs * 0.5), int(stage1_epochs * 0.75)]
    scheduler_stage1 = optim.lr_scheduler.MultiStepLR(optimizer_stage1, milestones=milestones_stage1, gamma=0.1)

    # Track training history
    history = {
        "stage1_train_losses": [],
        "stage1_test_accuracies": [],
        "stage2_train_losses": [],
        "stage2_test_accuracies": [],
        "best_accuracies": {exit_id: 0.0 for exit_id in exit_points},
    }

    best_final_acc = 0.0

    # Stage 1 Training Loop
    for epoch in range(stage1_epochs):
        epoch_start = time.time()
        print(f"\n[Stage 1] Epoch {epoch+1}/{stage1_epochs}, LR: {scheduler_stage1.get_last_lr()[0]:.6f}")

        # Train backbone + final only
        train_loss = train_backbone_only(
            model, trainloader, criterion, optimizer_stage1, device, epoch
        )

        # Evaluate all exits
        test_accs = evaluate(model, testloader, device, exit_points)
        epoch_time = time.time() - epoch_start

        # Print results
        print(f"\n[Stage 1] Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"Train Loss (final): {train_loss:.4f}")
        print("Test Accuracies:")
        for exit_id, acc in test_accs.items():
            print(f"  {exit_id:8s}: {acc:.2f}%")
            if acc > history["best_accuracies"][exit_id]:
                history["best_accuracies"][exit_id] = acc

        # Save history
        history["stage1_train_losses"].append({"final": train_loss})
        history["stage1_test_accuracies"].append(test_accs)

        # Save best model based on final exit accuracy
        if test_accs["final"] > best_final_acc:
            best_final_acc = test_accs["final"]
            checkpoint_path = os.path.join(save_dir, f"{model_name}_stage1_best.pth")
            torch.save({
                'stage': 1,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_stage1.state_dict(),
                'test_accuracies': test_accs,
            }, checkpoint_path)
            print(f"Saved Stage 1 best checkpoint to {checkpoint_path}")

        scheduler_stage1.step()

    print(f"\n{'='*60}")
    print(f"Stage 1 Complete - Best Final Accuracy: {best_final_acc:.2f}%")
    print(f"{'='*60}")

    # =========================
    # STAGE 2: Train early exit heads only
    # =========================
    if stage2_epochs > 0 and len([e for e in exit_points if e != "final"]) > 0:
        print(f"\n{'='*60}")
        print(f"STAGE 2: Training Early Exit Heads (Backbone Frozen)")
        print(f"{'='*60}")

        # Only optimize early exit head parameters
        optimizer_stage2 = optim.SGD(
            [p for p in model.exit_heads.parameters() if p.requires_grad],
            lr=lr * 0.1,  # Use lower learning rate for stage 2
            momentum=momentum,
            weight_decay=weight_decay
        )

        milestones_stage2 = [int(stage2_epochs * 0.5), int(stage2_epochs * 0.75)]
        scheduler_stage2 = optim.lr_scheduler.MultiStepLR(optimizer_stage2, milestones=milestones_stage2, gamma=0.1)

        # Stage 2 Training Loop
        for epoch in range(stage2_epochs):
            epoch_start = time.time()
            print(f"\n[Stage 2] Epoch {epoch+1}/{stage2_epochs}, LR: {scheduler_stage2.get_last_lr()[0]:.6f}")

            # Train early exits only
            train_losses = train_early_exits_only(
                model, trainloader, criterion, optimizer_stage2, device, exit_points, epoch
            )

            # Evaluate all exits
            test_accs = evaluate(model, testloader, device, exit_points)
            epoch_time = time.time() - epoch_start

            # Print results
            print(f"\n[Stage 2] Epoch {epoch+1} completed in {epoch_time:.2f}s")
            print("Train Losses (early exits):")
            for exit_id, loss in train_losses.items():
                print(f"  {exit_id:8s}: {loss:.4f}")
            print("Test Accuracies:")
            for exit_id, acc in test_accs.items():
                print(f"  {exit_id:8s}: {acc:.2f}%")
                if acc > history["best_accuracies"][exit_id]:
                    history["best_accuracies"][exit_id] = acc

            # Save history
            history["stage2_train_losses"].append(train_losses)
            history["stage2_test_accuracies"].append(test_accs)

            scheduler_stage2.step()

        print(f"\n{'='*60}")
        print(f"Stage 2 Complete")
        print(f"{'='*60}")

    # Unfreeze all parameters for final checkpoint
    for param in model.parameters():
        param.requires_grad = True

    # Save final checkpoint
    final_checkpoint_path = os.path.join(save_dir, f"{model_name}_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_accuracies': history["best_accuracies"],
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
        result_entry = {
            "best_accuracies": history["best_accuracies"],
        }

        # Get final accuracies from last epoch of stage 2 if available, otherwise stage 1
        if history["stage2_test_accuracies"]:
            result_entry["final_accuracies"] = history["stage2_test_accuracies"][-1]
        elif history["stage1_test_accuracies"]:
            result_entry["final_accuracies"] = history["stage1_test_accuracies"][-1]

        json_results[model_name] = result_entry

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
    parser.add_argument('--stage1-epochs', type=int, default=None,
                        help='epochs for stage 1 (backbone training). If None, uses 60%% of total epochs')

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
    stage1_epochs = args.stage1_epochs if args.stage1_epochs is not None else int(args.epochs * 0.6)
    stage2_epochs = args.epochs - stage1_epochs

    print(f"\nTraining Configuration:")
    print(f"  Total Epochs: {args.epochs}")
    print(f"  Stage 1 Epochs (Backbone): {stage1_epochs}")
    print(f"  Stage 2 Epochs (Early Exits): {stage2_epochs}")
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
            stage1_epochs=stage1_epochs,
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
