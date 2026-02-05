"""Tests for src.checkpoint module."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn

from src.checkpoint import CheckpointManager
from src.model import SimpleMLP


class TestCheckpointManager:
    """Test CheckpointManager functionality."""

    @pytest.fixture
    def temp_checkpoint_dir(self) -> Generator[Path, None, None]:
        """Create a temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_model(self) -> nn.Module:
        """Create a sample model for testing."""
        return SimpleMLP(input_dim=10, hidden_size=64, num_classes=2)

    @pytest.fixture
    def sample_optimizer(self, sample_model: nn.Module) -> torch.optim.Optimizer:
        """Create a sample optimizer."""
        return torch.optim.AdamW(sample_model.parameters(), lr=1e-3)

    def test_checkpoint_manager_creation(self, temp_checkpoint_dir: Path) -> None:
        """Test CheckpointManager can be instantiated."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        assert manager is not None
        assert manager.checkpoint_dir == temp_checkpoint_dir

    def test_save_checkpoint(
        self,
        temp_checkpoint_dir: Path,
        sample_model: nn.Module,
        sample_optimizer: torch.optim.Optimizer,
    ) -> None:
        """Test saving a checkpoint."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        manager.save(
            epoch=0,
            step=100,
            model=sample_model,
            optimizer=sample_optimizer,
            scheduler=None,
            metrics={"loss": 0.5},
            is_best=False,
        )

        # Verify checkpoint file was created
        checkpoint_files = list(temp_checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(checkpoint_files) > 0

    def test_save_best_checkpoint(
        self,
        temp_checkpoint_dir: Path,
        sample_model: nn.Module,
        sample_optimizer: torch.optim.Optimizer,
    ) -> None:
        """Test saving best checkpoint creates best.pt file."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        manager.save(
            epoch=0,
            step=100,
            model=sample_model,
            optimizer=sample_optimizer,
            scheduler=None,
            metrics={"loss": 0.5},
            is_best=True,
        )

        # Verify best.pt was created
        best_path = temp_checkpoint_dir / "best.pt"
        assert best_path.exists()

    def test_load_checkpoint(
        self,
        temp_checkpoint_dir: Path,
        sample_model: nn.Module,
        sample_optimizer: torch.optim.Optimizer,
    ) -> None:
        """Test loading a checkpoint."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        # Save checkpoint
        manager.save(
            epoch=5,
            step=1000,
            model=sample_model,
            optimizer=sample_optimizer,
            scheduler=None,
            metrics={"loss": 0.3, "accuracy": 0.95},
            is_best=False,
        )

        # Load checkpoint
        checkpoint_files = list(temp_checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(checkpoint_files) > 0

        info = manager.load(
            checkpoint_path=checkpoint_files[0],
            model=sample_model,
            optimizer=sample_optimizer,
            scheduler=None,
        )

        assert info["epoch"] == 5
        assert info["step"] == 1000
        assert info["metrics"]["loss"] == 0.3

    def test_checkpoint_preserves_model_state(
        self,
        temp_checkpoint_dir: Path,
        sample_model: nn.Module,
        sample_optimizer: torch.optim.Optimizer,
    ) -> None:
        """Test that checkpoint preserves model state correctly."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        # Get original weights
        original_weights = {
            name: param.clone()
            for name, param in sample_model.named_parameters()
        }

        # Save checkpoint
        manager.save(
            epoch=0,
            step=100,
            model=sample_model,
            optimizer=sample_optimizer,
            scheduler=None,
            metrics={},
            is_best=False,
        )

        # Create a new model and load checkpoint
        new_model = SimpleMLP(input_dim=10, hidden_size=64, num_classes=2)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)

        checkpoint_files = list(temp_checkpoint_dir.glob("checkpoint_*.pt"))
        manager.load(
            checkpoint_path=checkpoint_files[0],
            model=new_model,
            optimizer=new_optimizer,
            scheduler=None,
        )

        # Verify weights match
        for name, param in new_model.named_parameters():
            assert torch.allclose(param, original_weights[name])
