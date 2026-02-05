"""Tests for src.model module."""

import torch
import torch.nn as nn

from src.model import SimpleMLP, count_parameters, create_model


class TestSimpleMLP:
    """Test SimpleMLP architecture."""

    def test_model_creation(self) -> None:
        """Test that SimpleMLP can be instantiated."""
        model = SimpleMLP(input_dim=10, hidden_size=128, num_classes=2)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_model_forward_pass(self) -> None:
        """Test forward pass produces correct output shape."""
        model = SimpleMLP(input_dim=10, hidden_size=128, num_classes=2)
        batch = torch.randn(32, 10)
        output = model(batch)

        assert output.shape == (32, 2)
        assert output.dtype == torch.float32

    def test_model_device_movement(self) -> None:
        """Test model can be moved to different devices."""
        model = SimpleMLP(input_dim=10, hidden_size=128, num_classes=2)

        # Move to CPU
        model.to("cpu")
        assert next(model.parameters()).device.type == "cpu"

    def test_count_parameters(self) -> None:
        """Test parameter counting."""
        model = SimpleMLP(input_dim=10, hidden_size=128, num_classes=2)
        params = count_parameters(model)

        assert params > 0
        assert isinstance(params, int)

    def test_different_hidden_sizes(self) -> None:
        """Test model with different hidden layer sizes."""
        for hidden_size in [64, 128, 256]:
            model = SimpleMLP(input_dim=10, hidden_size=hidden_size, num_classes=2)
            batch = torch.randn(16, 10)
            output = model(batch)
            assert output.shape == (16, 2)


class TestCreateModel:
    """Test create_model factory function."""

    def test_create_model_default(self) -> None:
        """Test create_model with default arguments."""
        model = create_model()
        assert model is not None
        assert isinstance(model, SimpleMLP)

    def test_create_model_custom_dims(self) -> None:
        """Test create_model with custom dimensions."""
        model = create_model(input_dim=20, hidden_size=256, num_classes=5)
        batch = torch.randn(8, 20)
        output = model(batch)
        assert output.shape == (8, 5)

    def test_create_model_produces_trainable_params(self) -> None:
        """Test that created model has trainable parameters."""
        model = create_model()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0
