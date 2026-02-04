#!/bin/bash
# Quick start script for training Early Exit ResNets on CIFAR-10

set -e  # Exit on error

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate default

echo "========================================"
echo "Early Exit ResNet Training on CIFAR-10"
echo "========================================"

# Parse command line arguments
MODEL="all"
EPOCHS=200
BATCH_SIZE=128
USE_PRETRAINED=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --pretrained)
            USE_PRETRAINED="--use-pretrained"
            shift
            ;;
        --all)
            MODEL="all"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model resnet50|resnet101|resnet152|all] [--epochs 100] [--batch-size 128] [--pretrained]"
            exit 1
            ;;
    esac
done

# Determine which models to train
if [ "$MODEL" == "all" ]; then
    MODELS="resnet50 resnet101 resnet152"
else
    MODELS="$MODEL"
fi

echo ""
echo "Configuration:"
echo "  Models: $MODELS"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Use Pretrained: $([ -n "$USE_PRETRAINED" ] && echo "Yes" || echo "No")"
echo ""

# Create necessary directories
mkdir -p checkpoints
mkdir -p results
mkdir -p data

# Step 1: Train models
echo "========================================"
echo "Step 1: Training Models"
echo "========================================"

python train_early_exit_resnets.py \
    --models $MODELS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr 0.1 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --num-workers 4 \
    --save-dir ./checkpoints \
    $USE_PRETRAINED

echo ""
echo "Training completed!"
echo ""

# Step 2: Analyze results
echo "========================================"
echo "Step 2: Analyzing Results"
echo "========================================"

python analyze_early_exit_results.py \
    --checkpoint-dir ./checkpoints \
    --models $MODELS \
    --batch-size $BATCH_SIZE \
    --num-workers 4

echo ""
echo "========================================"
echo "All Done!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Checkpoints: ./checkpoints/"
echo "  - Results: ./results/"
echo "  - Plots: ./results/*.png"
echo ""
