"""
early_exit_resnet_bench.py

Build ResNet50 / ResNet101 / ResNet152 with multiple early-exit points
and benchmark latency + parameter size for each exit under isolated GPU.

Early exits:
    - "stem"   (after conv1/bn1/relu/maxpool)
    - "layer1"
    - "layer2"
    - "layer3"
    - "final" (original classifier)

We benchmark, for each model and each exit:
    - batch_size in {1, 2, ..., 10}
    - per-inference latency distribution (ms)
    - summary stats: mean, p50, p90, p95, p99

For each model we save one pickle:
    early_exit_latency_<ModelName>.pkl
"""

from typing import Iterable, Tuple, List, Dict

import time
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,
    resnet101,
    resnet152,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)


class EarlyExitResNet(nn.Module):
    """
    Wrap torchvision ResNet (50/101/152) with early-exit classifiers.

    Early exits supported:
        - "stem"   : after conv1/bn1/relu/maxpool
        - "layer1" : after layer1
        - "layer2" : after layer2
        - "layer3" : after layer3
        - "final"  : original avgpool + fc
    """

    def __init__(
        self,
        base_model: nn.Module,
        exit_points: Iterable[str] = ("stem", "layer1", "layer2", "layer3", "final"),
    ):
        super().__init__()
        self.base = base_model
        self.exit_points = tuple(exit_points)

        num_classes = base_model.fc.out_features  # ImageNet: 1000

        self.exit_heads = nn.ModuleDict()

        # "stem" exit: after maxpool, channels = conv1.out_channels (64)
        if "stem" in self.exit_points:
            c_stem = self.base.conv1.out_channels
            self.exit_heads["stem"] = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(c_stem, num_classes),
            )

        # For Bottleneck-based ResNet:
        # layer1: 256 channels, layer2: 512, layer3: 1024
        if "layer1" in self.exit_points:
            c1 = self.base.layer1[-1].conv3.out_channels
            self.exit_heads["layer1"] = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(c1, num_classes),
            )

        if "layer2" in self.exit_points:
            c2 = self.base.layer2[-1].conv3.out_channels
            self.exit_heads["layer2"] = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(c2, num_classes),
            )

        if "layer3" in self.exit_points:
            c3 = self.base.layer3[-1].conv3.out_channels
            self.exit_heads["layer3"] = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(c3, num_classes),
            )

    def forward(
        self,
        x: torch.Tensor,
        exit_id: str = "final",
        return_all: bool = False,
    ):
        """
        x: [B, 3, 224, 224]
        exit_id: "stem" | "layer1" | "layer2" | "layer3" | "final"
        return_all: if True, compute all exits and return dict.
        """
        outputs = {}

        # Stem
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        if "stem" in self.exit_heads:
            logits_stem = self.exit_heads["stem"](x)
            outputs["stem"] = logits_stem
            if not return_all and exit_id == "stem":
                return logits_stem

        # layer1
        x = self.base.layer1(x)
        if "layer1" in self.exit_heads:
            logits1 = self.exit_heads["layer1"](x)
            outputs["layer1"] = logits1
            if not return_all and exit_id == "layer1":
                return logits1

        # layer2
        x = self.base.layer2(x)
        if "layer2" in self.exit_heads:
            logits2 = self.exit_heads["layer2"](x)
            outputs["layer2"] = logits2
            if not return_all and exit_id == "layer2":
                return logits2

        # layer3
        x = self.base.layer3(x)
        if "layer3" in self.exit_heads:
            logits3 = self.exit_heads["layer3"](x)
            outputs["layer3"] = logits3
            if not return_all and exit_id == "layer3":
                return logits3

        # layer4 + final
        x = self.base.layer4(x)
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        logits_final = self.base.fc(x)
        outputs["final"] = logits_final

        if not return_all:
            return outputs.get(exit_id, logits_final)
        else:
            return outputs

    # --------- parameter count per exit ---------

    def _count_params(self, modules: List[nn.Module]) -> int:
        total = 0
        for m in modules:
            for p in m.parameters():
                total += p.numel()
        return total

    def param_count_for_exit(self, exit_id: str) -> int:
        """
        Count parameters *actually used* to compute up to this exit.
        """
        if exit_id == "stem":
            mods = [
                self.base.conv1,
                self.base.bn1,
                self.exit_heads["stem"],
            ]
        elif exit_id == "layer1":
            mods = [
                self.base.conv1,
                self.base.bn1,
                self.base.layer1,
                self.exit_heads["layer1"],
            ]
        elif exit_id == "layer2":
            mods = [
                self.base.conv1,
                self.base.bn1,
                self.base.layer1,
                self.base.layer2,
                self.exit_heads["layer2"],
            ]
        elif exit_id == "layer3":
            mods = [
                self.base.conv1,
                self.base.bn1,
                self.base.layer1,
                self.base.layer2,
                self.base.layer3,
                self.exit_heads["layer3"],
            ]
        elif exit_id == "final":
            mods = [self.base]
        else:
            raise ValueError(f"Unknown exit_id: {exit_id}")

        return self._count_params(mods)


def build_early_exit_resnets(
    device: torch.device,
    exit_points=("stem", "layer1", "layer2", "layer3", "final"),
) -> Tuple[EarlyExitResNet, EarlyExitResNet, EarlyExitResNet]:
    print("Loading pretrained ResNet50/101/152 with early exits...")

    base50 = resnet50(weights=ResNet50_Weights.DEFAULT)
    base101 = resnet101(weights=ResNet101_Weights.DEFAULT)
    base152 = resnet152(weights=ResNet152_Weights.DEFAULT)

    m50 = EarlyExitResNet(base50, exit_points=exit_points).to(device)
    m101 = EarlyExitResNet(base101, exit_points=exit_points).to(device)
    m152 = EarlyExitResNet(base152, exit_points=exit_points).to(device)

    m50.eval()
    m101.eval()
    m152.eval()

    return m50, m101, m152


def benchmark_latency(
    model: EarlyExitResNet,
    device: torch.device,
    exit_id: str,
    batch_size: int,
    warmup: int,
    iters: int,
) -> List[float]:
    """
    Benchmark per-inference latency (ms) for a given exit_id and batch size.
    Returns a list of latencies in ms, length = iters.
    """
    x_cpu = torch.randn(batch_size, 3, 224, 224)
    # x = x_cpu.to(device)

    # Warm-up
    # x = torch.randn(batch_size, 3, 224, 224, device=device)

    with torch.no_grad():
        # Warm-up
        for _ in range(warmup):
            x = x_cpu.to(device)
            _ = model(x, exit_id=exit_id)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        latencies_ms: List[float] = []
        for _ in range(iters):
            start = time.perf_counter()
            x = x_cpu.to(device)
            _ = model(x, exit_id=exit_id)
            # _ = result.cpu()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.perf_counter()
            lat_ms = (end - start) * 1000.0
            latencies_ms.append(lat_ms)

    return latencies_ms


def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs("./saves", exist_ok=True)


    exit_points = ("stem", "layer1", "layer2", "layer3", "final")
    models: Dict[str, EarlyExitResNet] = {}
    models["ResNet50"], models["ResNet101"], models["ResNet152"] = build_early_exit_resnets(
        device, exit_points=exit_points
    )

    batch_sizes = list(range(1, 11))  # 1..10
    warmup = 20
    iters = 200

    print("\nBenchmarking latency and parameter size per exit (isolated GPU):")
    print(f"(batch_sizes = {batch_sizes}, warmup = {warmup}, iters = {iters})\n")

    for model_name, model in models.items():
        print(f"\n===== {model_name} =====")

        # model_results[exit_id][batch_size] = {...}
        model_results: Dict[str, Dict[int, Dict[str, object]]] = {}

        for exit_id in exit_points:
            exit_results: Dict[int, Dict[str, object]] = {}

            # Parameter count does not depend on batch size
            params = model.param_count_for_exit(exit_id)
            params_million = params / 1e6

            for bs in batch_sizes:
                latencies_ms = benchmark_latency(
                    model=model,
                    device=device,
                    exit_id=exit_id,
                    batch_size=bs,
                    warmup=warmup,
                    iters=iters,
                )

                arr = np.array(latencies_ms)
                mean_ms = arr.mean()
                p50 = np.percentile(arr, 50)
                p90 = np.percentile(arr, 90)
                p95 = np.percentile(arr, 95)
                p99 = np.percentile(arr, 99)

                print(
                    f"bs={bs:2d}  exit={exit_id:6}  "
                    f"params={params_million:7.3f}M  "
                    f"mean={mean_ms:7.3f} ms  "
                    f"p50={p50:7.3f} ms  "
                    f"p90={p90:7.3f} ms  "
                    f"p95={p95:7.3f} ms  "
                    f"p99={p99:7.3f} ms"
                )

                exit_results[bs] = {
                    "params": params,
                    "latencies_ms": latencies_ms,
                    "mean_ms": float(mean_ms),
                    "p50_ms": float(p50),
                    "p90_ms": float(p90),
                    "p95_ms": float(p95),
                    "p99_ms": float(p99),
                }

            model_results[exit_id] = exit_results

        # Save this model’s results to its own pickle file
        payload = {
            "device": str(device),
            "batch_sizes": batch_sizes,
            "warmup": warmup,
            "iters": iters,
            "exit_points": exit_points,
            "model_name": model_name,
            "results": model_results,  # exit_id -> batch_size -> stats dict
        }
        pickle_path = f"saves/early_exit_latency_{model_name}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"\nSaved latency data for {model_name} to: {pickle_path}")

    print("\nAll models benchmarked. Done.")


if __name__ == "__main__":
    main()
