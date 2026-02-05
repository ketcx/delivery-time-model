"""PyTorch model definitions."""

from typing import Any

import torch
import torch.nn as nn


class DeliveryTimeModel(nn.Module):
    """Neural Network for predicting delivery times.

    Architecture: 4 inputs → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1)
    
    This model predicts delivery time in minutes based on:
    - Normalized distance (miles)
    - Normalized time of day (hours)
    - Weekend flag (0 or 1)
    - Rush hour flag (0 or 1)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
    ):
        """Initialize DeliveryTimeModel.

        Args:
            input_dim: Input feature dimension (default: 4)
            hidden_size_1: First hidden layer size (default: 64)
            hidden_size_2: Second hidden layer size (default: 32)
        """
        super().__init__()

        # Build the network architecture
        self.model = nn.Sequential(
            # Input layer: 4 features → 64 neurons
            nn.Linear(input_dim, hidden_size_1),
            nn.ReLU(),
            # Hidden layer: 64 → 32 neurons
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            # Output layer: 32 → 1 (delivery time prediction)
            nn.Linear(hidden_size_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 4)

        Returns:
            Output tensor of shape (batch_size, 1) - predicted delivery time
        """
        return self.model(x)


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for classification.

    Architecture: input → linear → relu → dropout → linear → output
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize SimpleMLP.

        Args:
            input_dim: Input feature dimension
            hidden_size: Hidden layer size
            num_layers: Number of hidden layers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()

        layers: list[Any] = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_size, num_classes))

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # type: ignore
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)


def create_delivery_model(
    input_dim: int = 4,
    hidden_size_1: int = 64,
    hidden_size_2: int = 32,
    device: str = "cpu",
) -> DeliveryTimeModel:
    """Create and initialize delivery time prediction model.

    Args:
        input_dim: Input feature dimension (default: 4)
        hidden_size_1: First hidden layer size (default: 64)
        hidden_size_2: Second hidden layer size (default: 32)
        device: Device to place model on (cpu, mps, cuda)

    Returns:
        Model instance on specified device
    """
    model = DeliveryTimeModel(
        input_dim=input_dim,
        hidden_size_1=hidden_size_1,
        hidden_size_2=hidden_size_2,
    )
    return model.to(device)


def create_model(
    input_dim: int = 10,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_classes: int = 2,
    dropout: float = 0.1,
    device: str = "cpu",
) -> SimpleMLP:
    """Create and initialize model.

    Args:
        input_dim: Input feature dimension
        hidden_size: Hidden layer size
        num_layers: Number of hidden layers
        num_classes: Number of output classes
        dropout: Dropout probability
        device: Device to place model on (cpu, mps, cuda)

    Returns:
        Model instance on specified device
    """
    model = SimpleMLP(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )
    model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
