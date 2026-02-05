"""Benchmark script comparing MPS vs CPU performance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src import parse_cli_args
from src.benchmark import Benchmark
from src.data import create_dataloaders
from src.device import get_device_info, is_mps_available
from src.model import create_model


def main():
    """Run device benchmark."""
    # Parse arguments (unused but kept for consistency with other scripts)
    _ = parse_cli_args()

    print("=" * 70)
    print("PyTorch Device Benchmark - MPS vs CPU")
    print("=" * 70)

    # Print device info
    print("\nDevice Information:")
    device_info = get_device_info()
    print(f"  MPS Available: {device_info['mps_available']}")
    print(f"  MPS Supported: {device_info['mps_supported']}")
    print(f"  CUDA Available: {device_info['cuda_available']}")
    print(f"  PyTorch Version: {device_info['pytorch_version']}")

    if not is_mps_available():
        print("\n⚠️  MPS not available. Benchmarking CPU only.")
        devices = ["cpu"]
    else:
        devices = ["mps", "cpu"]

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, _, _ = create_dataloaders(
        dataset_size=500,
        batch_size=32,
        num_workers=0,
    )

    # Define model factory
    def model_fn():
        # Note: device parameter is ignored; benchmark moves model to target device
        return create_model(
            input_dim=10,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
        )

    # Define optimizer factory
    def optimizer_fn(model):
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Run benchmarks
    print("\n" + "=" * 70)
    print("Running Benchmarks...")
    print("=" * 70)

    results = Benchmark.compare_devices(
        model_fn=model_fn,
        dataloader=train_loader,
        criterion=criterion,
        optimizer_fn=optimizer_fn,
        devices=devices,
        dtypes=["fp32"],  # MPS has limitations with fp16/bf16
        num_steps=50,
        warmup_steps=10,
    )

    # Print comparison
    Benchmark.print_comparison(results)

    # Print recommendations
    print("\n" + "=" * 70)
    print("Recommendations:")
    print("=" * 70)

    if is_mps_available():
        mps_result = results.get("mps_fp32", {})
        cpu_result = results.get("cpu_fp32", {})

        if mps_result and cpu_result:
            speedup = (
                mps_result["throughput_samples_per_sec"] / cpu_result["throughput_samples_per_sec"]
            )
            if speedup > 1:
                print(f"✓ MPS is {speedup:.2f}x faster than CPU for this workload")
                print("  → Use MPS for training")
            else:
                print(f"⚠️  CPU is {1/speedup:.2f}x faster than MPS for this workload")
                print("  → Use CPU for this configuration")

    print("\nNotes:")
    print("- Batch size affects performance significantly")
    print("- Try increasing batch_size for better GPU utilization")
    print("- MPS works best with large models and large batches")
    print("- If MPS becomes unstable, fall back to CPU")


if __name__ == "__main__":
    main()
