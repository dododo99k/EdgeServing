# Early Exit ResNet Training on CIFAR-10

这个项目包含了训练和评估Early Exit ResNet模型（ResNet50/101/152）在CIFAR-10数据集上的完整代码。

## 文件说明

- `early_exit_resnets.py`: 定义EarlyExitResNet类和基础功能
- `train_early_exit_resnets.py`: 训练脚本
- `analyze_early_exit_results.py`: 结果分析和可视化脚本

## 快速开始

### 1. 训练单个模型（ResNet50）

```bash
python train_early_exit_resnets.py \
    --models resnet50 \
    --batch-size 128 \
    --epochs 100 \
    --lr 0.1
```

### 2. 训练所有模型（ResNet50/101/152）

```bash
python train_early_exit_resnets.py \
    --models resnet50 resnet101 resnet152 \
    --batch-size 128 \
    --epochs 100 \
    --lr 0.1 \
    --num-workers 4
```

### 3. 使用预训练权重初始化（迁移学习）

```bash
python train_early_exit_resnets.py \
    --models resnet50 \
    --use-pretrained \
    --batch-size 128 \
    --epochs 50 \
    --lr 0.01
```

### 4. 分析训练结果

```bash
python analyze_early_exit_results.py \
    --checkpoint-dir ./checkpoints \
    --models resnet50 resnet101 resnet152
```

## 命令行参数

### train_early_exit_resnets.py

- `--batch-size`: 训练批次大小（默认: 128）
- `--epochs`: 训练轮数（默认: 100）
- `--lr`: 学习率（默认: 0.1）
- `--momentum`: SGD动量（默认: 0.9）
- `--weight-decay`: 权重衰减（默认: 5e-4）
- `--num-workers`: 数据加载器工作进程数（默认: 4）
- `--save-dir`: 检查点保存目录（默认: ./checkpoints）
- `--models`: 要训练的模型列表（可选: resnet50, resnet101, resnet152）
- `--use-pretrained`: 使用ImageNet预训练权重作为初始化

### analyze_early_exit_results.py

- `--checkpoint-dir`: 检查点目录（默认: ./checkpoints）
- `--batch-size`: 评估批次大小（默认: 128）
- `--num-workers`: 数据加载器工作进程数（默认: 4）
- `--models`: 要分析的模型列表（可选: resnet50, resnet101, resnet152）

## Early Exit点说明

每个模型包含4个exit点：

1. **layer1**: 在ResNet的第一个stage之后
2. **layer2**: 在ResNet的第二个stage之后
3. **layer3**: 在ResNet的第三个stage之后
4. **final**: 原始分类器（完整模型）

## 输出文件

### 训练阶段

训练会生成以下文件：

- `./checkpoints/<ModelName>_best.pth`: 最佳检查点（基于final exit准确率）
- `./checkpoints/<ModelName>_final.pth`: 最终检查点（包含完整训练历史）
- `./results/early_exit_results.json`: 所有模型的性能摘要

### 分析阶段

分析会生成以下文件：

- `./results/detailed_results.json`: 详细结果（包含参数量和准确率）
- `./results/exit_accuracies.png`: 各个exit点准确率对比图
- `./results/accuracy_vs_params.png`: 准确率vs参数量曲线图

## CIFAR-10适配说明

原始ResNet设计用于ImageNet（224x224图像），我们对CIFAR-10（32x32图像）做了以下适配：

1. 将第一个卷积层从7x7 stride 2改为3x3 stride 1
2. 移除maxpool层
3. 将输出层从1000类改为10类

## 训练策略

- **优化器**: SGD with momentum (0.9)
- **学习率调度**: MultiStepLR（在50%和75%训练进度时降低10倍）
- **数据增强**: Random crop + Random horizontal flip
- **损失函数**: 联合训练所有exit点的交叉熵损失

## 示例输出

训练完成后，你会看到类似这样的输出：

```
ResNet50:
  Best Accuracies:
    layer1  : 85.32%
    layer2  : 89.67%
    layer3  : 91.23%
    final   : 92.15%
```

## 注意事项

1. 确保有足够的GPU显存（ResNet152需要较大显存）
2. CIFAR-10数据集会自动下载到`./data`目录
3. 训练时间取决于模型大小和硬件配置
4. 建议先用ResNet50测试，确认流程正常后再训练更大的模型
