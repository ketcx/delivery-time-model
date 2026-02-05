"""Benchmarking utilities for performance comparison."""

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.device import get_dtype_from_string


class Benchmark:
    """Benchmark runner for training performance."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        dtype: str = "fp32",
        warmup_steps: int = 10,
    ):
        """Initialize benchmark.

        Args:
            model: Model to benchmark
            dataloader: Data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device (mps, cpu, cuda)
            dtype: Data type (fp32, fp16, bf16)
            warmup_steps: Number of warmup steps to ignore
        """
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.dtype = get_dtype_from_string(dtype)
        self.warmup_steps = warmup_steps
        self.model.to(device)

    def run(
        self,
        num_steps: int = 100,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run benchmark.

        Args:
            num_steps: Number of steps to benchmark
            verbose: Print progress

        Returns:
            Dictionary with benchmark results
        """
        self.model.train()

        # Create iterator that cycles through dataloader
        dataloader_iter = iter(self.dataloader)

        step_times = []
        total_samples = 0

        if verbose:
            print(f"Warming up for {self.warmup_steps} steps...")

        for step in range(num_steps + self.warmup_steps):
            # Get batch
            try:
                x, y = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.dataloader)
                x, y = next(dataloader_iter)

            x, y = x.to(self.device), y.to(self.device)

            # Benchmark step
            step_start = time.perf_counter()

            # Forward pass
            with torch.autocast(
                device_type=self.device,
                dtype=self.dtype,
                enabled=(self.dtype != torch.float32),
            ):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            step_time = time.perf_counter() - step_start
            total_samples += x.size(0)

            # Skip warmup steps
            if step >= self.warmup_steps:
                step_times.append(step_time)

                if verbose and (step - self.warmup_steps + 1) % 10 == 0:
                    avg_time = sum(step_times) / len(step_times)
                    throughput = x.size(0) / avg_time
                    print(
                        f"Step {step - self.warmup_steps + 1}/{num_steps} | "
                        f"Time: {avg_time*1000:.2f}ms | "
                        f"Throughput: {throughput:.2f} samples/s"
                    )

        # Calculate statistics
        step_times_sorted = sorted(step_times)
        results = {
            "device": self.device,
            "dtype": str(self.dtype).split(".")[-1],
            "total_steps": len(step_times),
            "total_samples": total_samples,
            "avg_step_time_ms": (sum(step_times) / len(step_times)) * 1000,
            "median_step_time_ms": step_times_sorted[len(step_times) // 2] * 1000,
            "min_step_time_ms": min(step_times) * 1000,
            "max_step_time_ms": max(step_times) * 1000,
            "throughput_samples_per_sec": total_samples / sum(step_times),
        }

        return results

    @staticmethod
    def compare_devices(
        model_fn: Any,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer_fn: Any,
        devices: Optional[List[str]] = None,
        dtypes: Optional[List[str]] = None,
        num_steps: int = 50,
        warmup_steps: int = 10,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare benchmark across multiple devices and dtypes.

        Args:
            model_fn: Function that returns a fresh model
            dataloader: Data loader
            criterion: Loss function
            optimizer_fn: Function that returns optimizer for a model
            devices: List of devices to benchmark (default: [mps, cpu])
            dtypes: List of dtypes to benchmark (default: [fp32])
            num_steps: Number of steps per benchmark
            warmup_steps: Warmup steps

        Returns:
            Dictionary with benchmark results for each config
        """
        if devices is None:
            devices = ["mps", "cpu"]
        if dtypes is None:
            dtypes = ["fp32"]

        # Filter devices to available ones
        import torch

        if not torch.backends.mps.is_available():
            devices = [d for d in devices if d != "mps"]

        results = {}

        for device in devices:
            for dtype in dtypes:
                print(f"\nBenchmarking {device} with {dtype}...")

                # Create fresh model and optimizer
                model = model_fn()
                optimizer = optimizer_fn(model)

                # Run benchmark
                benchmark = Benchmark(
                    model=model,
                    dataloader=dataloader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    dtype=dtype,
                    warmup_steps=warmup_steps,
                )

                result_key = f"{device}_{dtype}"
                results[result_key] = benchmark.run(
                    num_steps=num_steps,
                    verbose=True,
                )

                # Clean up
                del model, optimizer
                if device == "cuda":
                    torch.cuda.empty_cache()

        return results

    @staticmethod
    def print_comparison(results: Dict[str, Dict[str, float]]) -> None:
        """Print comparison table.

        Args:
            results: Results from compare_devices
        """
        print("\n" + "=" * 100)
        print("BENCHMARK COMPARISON")
        print("=" * 100)

        # Header
        print(
            f"{'Config':<20} | "
            f"{'Avg Step (ms)':<15} | "
            f"{'Median (ms)':<15} | "
            f"{'Throughput':<20}"
        )
        print("-" * 100)

        # Results sorted by throughput (descending)
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]["throughput_samples_per_sec"],
            reverse=True,
        )

        for config, metrics in sorted_results:
            print(
                f"{config:<20} | "
                f"{metrics['avg_step_time_ms']:>13.2f} | "
                f"{metrics['median_step_time_ms']:>13.2f} | "
                f"{metrics['throughput_samples_per_sec']:>18.2f} samples/s"
            )

        print("=" * 100)
