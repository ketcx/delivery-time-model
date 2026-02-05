"""PyTorch data utilities."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class DeliveryDataset(Dataset):
    """Delivery time dataset with feature engineering.
    
    Features:
    - distance_miles (normalized)
    - time_of_day_hours (normalized)
    - is_weekend (binary)
    - rush_hour (engineered binary feature)
    
    Target:
    - delivery_time_minutes
    """

    def __init__(self, csv_path: str | Path):
        """Initialize delivery dataset from CSV.

        Args:
            csv_path: Path to CSV file with delivery data
        """
        # Load data
        df = pd.read_csv(csv_path)
        
        # Prepare features and targets
        self.features, self.targets = self._prepare_data(df)
        
    def _rush_hour_feature(
        self, 
        hours_tensor: torch.Tensor, 
        weekends_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Engineers rush hour feature.
        
        Rush hour is defined as:
        - Morning: 8:00-10:00 AM (8.0-10.0)
        - Evening: 4:00-7:00 PM (16.0-19.0)
        - Only on weekdays (is_weekend == 0)
        
        Args:
            hours_tensor: Delivery time of day (hours)
            weekends_tensor: Weekend indicator (0=weekday, 1=weekend)
            
        Returns:
            Binary tensor indicating rush hour (1) or not (0)
        """
        # Define rush hour conditions
        is_morning_rush = (hours_tensor >= 8.0) & (hours_tensor < 10.0)
        is_evening_rush = (hours_tensor >= 16.0) & (hours_tensor < 19.0)
        is_weekday = (weekends_tensor == 0)
        
        # Combine: weekday AND (morning rush OR evening rush)
        is_rush_hour_mask = is_weekday & (is_morning_rush | is_evening_rush)
        
        return is_rush_hour_mask.float()
    
    def _prepare_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts DataFrame into prepared PyTorch tensors.
        
        Args:
            df: DataFrame with columns: distance_miles, time_of_day_hours,
                is_weekend, delivery_time_minutes
                
        Returns:
            Tuple of (features, targets) tensors
        """
        # Convert to tensor
        full_tensor = torch.tensor(df.values, dtype=torch.float32)
        
        # Slice columns
        raw_distances = full_tensor[:, 0]
        raw_hours = full_tensor[:, 1]
        raw_weekends = full_tensor[:, 2]
        raw_targets = full_tensor[:, 3]
        
        # Engineer rush hour feature
        is_rush_hour_feature = self._rush_hour_feature(raw_hours, raw_weekends)
        
        # Reshape to column vectors
        distances_col = raw_distances.unsqueeze(1)
        hours_col = raw_hours.unsqueeze(1)
        weekends_col = raw_weekends.unsqueeze(1)
        rush_hour_col = is_rush_hour_feature.unsqueeze(1)
        
        # Normalize continuous features
        dist_mean, dist_std = distances_col.mean(), distances_col.std()
        hours_mean, hours_std = hours_col.mean(), hours_col.std()
        
        distances_norm = (distances_col - dist_mean) / dist_std
        hours_norm = (hours_col - hours_mean) / hours_std
        
        # Combine all features
        features = torch.cat([
            distances_norm,
            hours_norm,
            weekends_col,
            rush_hour_col
        ], dim=1)
        
        # Prepare targets
        targets = raw_targets.unsqueeze(1)
        
        return features, targets

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, target)
        """
        return self.features[idx], self.targets[idx]


def create_delivery_dataloaders(
    csv_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders for delivery data.

    Args:
        csv_path: Path to CSV file with delivery data
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        split: (train, val, test) split ratios

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create dataset
    dataset = DeliveryDataset(csv_path)

    # Split indices
    train_size = int(len(dataset) * split[0])
    val_size = int(len(dataset) * split[1])

    indices = torch.randperm(len(dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create subsets
    from torch.utils.data import Subset

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


class DummyDataset(Dataset):
    """Dummy dataset for testing.

    Generates random samples with targets for demonstration.
    """

    def __init__(self, size: int = 1000, input_dim: int = 10, num_classes: int = 2):
        """Initialize dummy dataset.

        Args:
            size: Number of samples
            input_dim: Input feature dimension
            num_classes: Number of classes
        """
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Generate dummy data
        self.X = np.random.randn(size, input_dim).astype(np.float32)
        self.y = np.random.randint(0, num_classes, size).astype(np.int64)

    def __len__(self) -> int:
        """Return dataset size."""
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input tensor, target tensor)
        """
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def create_dataloaders(
    dataset_size: int = 1000,
    input_dim: int = 10,
    num_classes: int = 2,
    batch_size: int = 32,
    num_workers: int = 0,
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders.

    Args:
        dataset_size: Total dataset size
        input_dim: Input feature dimension
        num_classes: Number of classes
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        split: (train, val, test) split ratios

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    dataset = DummyDataset(
        size=dataset_size, input_dim=input_dim, num_classes=num_classes
    )

    # Split indices
    train_size = int(len(dataset) * split[0])
    val_size = int(len(dataset) * split[1])

    indices = torch.randperm(len(dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create subsets
    from torch.utils.data import Subset

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
