"""Checkpoint management utilities."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """Manages model checkpointing and resuming."""

    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        epoch: int,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        metrics: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> Path:
        """Save checkpoint.

        Args:
            epoch: Current epoch
            step: Current step
            model: Model to save
            optimizer: Optimizer state
            scheduler: Optional scheduler state
            metrics: Optional metrics dictionary
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics or {},
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)

        return checkpoint_path

    def load(
        self,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint and restore state.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to restore
            optimizer: Optimizer to restore
            scheduler: Optional scheduler to restore

        Returns:
            Dictionary with checkpoint metadata (epoch, step, metrics)

        Security Note:
            Uses weights_only=False to support loading optimizer and scheduler states.
            Only load checkpoints from trusted sources. For untrusted checkpoints,
            consider using weights_only=True and loading only model weights separately.
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return {
            "epoch": checkpoint["epoch"],
            "step": checkpoint["step"],
            "metrics": checkpoint.get("metrics", {}),
        }

    def load_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load best checkpoint if it exists.

        Args:
            model: Model to restore
            optimizer: Optimizer to restore
            scheduler: Optional scheduler to restore

        Returns:
            Checkpoint metadata or None if best checkpoint doesn't exist
        """
        best_path = self.checkpoint_dir / "best.pt"
        if not best_path.exists():
            return None

        return self.load(best_path, model, optimizer, scheduler)

    def list_checkpoints(self) -> list[Path]:
        """List all checkpoint files.

        Returns:
            List of checkpoint paths sorted by epoch
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        return sorted(checkpoints)

    def cleanup_old_checkpoints(self, keep_last_n: int = 3) -> None:
        """Keep only last N checkpoints to save space.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) > keep_last_n:
            for ckpt in checkpoints[:-keep_last_n]:
                ckpt.unlink()
