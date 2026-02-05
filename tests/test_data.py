"""Tests for src.data module - simplified."""

import torch
from torch.utils.data import DataLoader

from src.data import DummyDataset, create_dataloaders


class TestDummyDataset:
    """Test DummyDataset class."""

    def test_dataset_creation(self) -> None:
        """Test DummyDataset instantiation."""
        dataset = DummyDataset(size=100, input_dim=10, num_classes=2)
        assert dataset is not None

    def test_dataset_length(self) -> None:
        """Test dataset __len__ method."""
        size = 100
        dataset = DummyDataset(size=size, input_dim=10, num_classes=2)
        assert len(dataset) == size

    def test_dataset_getitem(self) -> None:
        """Test dataset __getitem__ method."""
        dataset = DummyDataset(size=100, input_dim=10, num_classes=2)
        x, y = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (10,)
        assert y.shape == ()  # scalar

    def test_dataset_different_sizes(self) -> None:
        """Test dataset with different input dimensions."""
        for input_dim in [5, 10, 20]:
            dataset = DummyDataset(size=50, input_dim=input_dim, num_classes=2)
            x, y = dataset[0]
            assert x.shape == (input_dim,)

    def test_dataset_different_num_classes(self) -> None:
        """Test dataset with different number of classes."""
        for num_classes in [2, 5, 10]:
            dataset = DummyDataset(size=50, input_dim=10, num_classes=num_classes)
            x, y = dataset[0]
            assert 0 <= y.item() < num_classes


class TestCreateDataloaders:
    """Test create_dataloaders function."""

    def test_create_dataloaders_returns_tuple(self) -> None:
        """Test that create_dataloaders returns a tuple."""
        loaders = create_dataloaders(dataset_size=100, batch_size=32, num_workers=0)
        assert isinstance(loaders, tuple)
        assert len(loaders) == 3  # train, val, test

    def test_create_dataloaders_returns_dataloader_objects(self) -> None:
        """Test that create_dataloaders returns DataLoader objects."""
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_size=100, batch_size=32, num_workers=0
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    def test_dataloader_batch_shape(self) -> None:
        """Test that DataLoader produces correct batch shapes."""
        train_loader, _, _ = create_dataloaders(
            dataset_size=100, batch_size=32, num_workers=0, input_dim=10
        )

        batch_x, batch_y = next(iter(train_loader))

        assert batch_x.shape[0] <= 32  # batch size
        assert batch_x.shape[1] == 10  # input dimension
        assert batch_y.shape[0] == batch_x.shape[0]

    def test_dataloader_with_different_batch_sizes(self) -> None:
        """Test DataLoader with different batch sizes."""
        for batch_size in [8, 16, 32, 64]:
            train_loader, _, _ = create_dataloaders(
                dataset_size=100, batch_size=batch_size, num_workers=0
            )

            batch_x, _ = next(iter(train_loader))
            assert batch_x.shape[0] <= batch_size

    def test_dataloader_num_workers_zero(self) -> None:
        """Test that num_workers=0 works (important for macOS)."""
        train_loader, _, _ = create_dataloaders(
            dataset_size=50, batch_size=16, num_workers=0
        )

        # Should iterate without errors
        for batch_x, batch_y in train_loader:
            assert batch_x is not None
            assert batch_y is not None
            break  # Just test one batch
