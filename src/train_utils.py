"""Training loop utilities."""

import math
import time
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.run_manager import RunManager


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: str,
    epoch: int,
    run_manager: RunManager,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (step-based, called after each batch)
        criterion: Loss function
        device: Device (cpu, mps, cuda)
        epoch: Current epoch number
        run_manager: Run manager for logging

    Returns:
        Dictionary with epoch metrics

    Note:
        This function expects a step-based scheduler (e.g., from
        get_cosine_schedule_with_warmup or get_linear_schedule_with_warmup)
        that is updated after every batch. For epoch-based schedulers,
        call scheduler.step() after this function returns.
    """
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(dataloader):
        # Move to device
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits = model(x)
        loss = criterion(logits, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients (optional but good practice)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()
        scheduler.step()

        # Accumulate metrics
        epoch_loss += loss.item()
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean()
            epoch_acc += acc.item()

        num_batches += 1

        # Log every save_interval batches
        if (batch_idx + 1) % run_manager.config.logging.save_interval == 0:
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            lr = optimizer.param_groups[0]["lr"]
            run_manager.logger.info(
                f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | LR: {lr:.2e}"
            )

    epoch_time = time.time() - start_time
    avg_loss = epoch_loss / num_batches
    avg_acc = epoch_acc / num_batches

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "time_seconds": epoch_time,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on dataloader.

    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        criterion: Loss function
        device: Device (cpu, mps, cuda)

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    num_batches = 0

    for x, y in dataloader:
        # Move to device
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits = model(x)
        loss = criterion(logits, y)

        # Accumulate metrics
        eval_loss += loss.item()
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        eval_acc += acc.item()

        num_batches += 1

    avg_loss = eval_loss / num_batches
    avg_acc = eval_acc / num_batches

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
    }


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create a learning rate scheduler with warmup.

    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps

    Returns:
        Learning rate scheduler
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create a cosine learning rate scheduler with warmup.

    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles

    Returns:
        Learning rate scheduler
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0,
            0.5
            * (
                1.0
                + math.cos(
                    math.pi * float(num_cycles) * 2.0 * progress
                )
            ),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
