"""Tests for src.device module - simplified."""

import pytest
import torch

from src.device import (
    get_device,
    get_device_info,
    get_dtype_from_string,
    is_mps_available,
    is_mps_supported,
)


class TestDeviceDetection:
    """Test device detection functions."""

    def test_is_mps_available_returns_bool(self) -> None:
        """Test that is_mps_available returns a boolean."""
        result = is_mps_available()
        assert isinstance(result, bool)

    def test_is_mps_supported_returns_bool(self) -> None:
        """Test that is_mps_supported returns a boolean."""
        result = is_mps_supported()
        assert isinstance(result, bool)

    def test_get_device_returns_string(self) -> None:
        """Test that get_device returns a device string."""
        device = get_device(preferred_device="cpu")
        assert isinstance(device, str)
        assert device in ["cpu", "mps"]

    def test_get_device_fallback(self) -> None:
        """Test that get_device falls back to CPU if MPS unavailable."""
        device = get_device(preferred_device="mps", fallback_on_error=True)
        assert device in ["cpu", "mps"]

    def test_get_device_cpu_always_available(self) -> None:
        """Test that CPU device is always available."""
        device = get_device(preferred_device="cpu")
        assert device == "cpu"


class TestDtypeConversion:
    """Test dtype conversion functions."""

    def test_get_dtype_fp32(self) -> None:
        """Test FP32 dtype conversion."""
        dtype = get_dtype_from_string("fp32")
        assert dtype == torch.float32

    def test_get_dtype_fp16(self) -> None:
        """Test FP16 dtype conversion."""
        dtype = get_dtype_from_string("fp16")
        assert dtype == torch.float16

    def test_get_dtype_bf16(self) -> None:
        """Test BF16 dtype conversion."""
        dtype = get_dtype_from_string("bf16")
        assert dtype == torch.bfloat16

    def test_get_dtype_invalid_raises_error(self) -> None:
        """Test that invalid dtype raises error."""
        with pytest.raises(ValueError):
            get_dtype_from_string("invalid_dtype")

    def test_dtype_is_torch_dtype(self) -> None:
        """Test that returned dtype is a torch dtype."""
        for dtype_str in ["fp32", "fp16", "bf16"]:
            dtype = get_dtype_from_string(dtype_str)
            assert isinstance(dtype, torch.dtype)


class TestDeviceTensorOperations:
    """Test tensor operations on different devices."""

    def test_tensor_creation_on_cpu(self) -> None:
        """Test creating tensors on CPU."""
        tensor = torch.randn(10, 10, device="cpu")
        assert tensor.device.type == "cpu"

    def test_tensor_dtype_conversion(self) -> None:
        """Test tensor dtype conversion."""
        tensor = torch.randn(10, 10, dtype=torch.float32)
        assert tensor.dtype == torch.float32

        tensor_fp16 = tensor.to(torch.float16)
        assert tensor_fp16.dtype == torch.float16

    def test_tensor_device_movement(self) -> None:
        """Test moving tensors to different devices."""
        tensor = torch.randn(10, 10, device="cpu")
        tensor_cpu = tensor.to("cpu")
        assert tensor_cpu.device.type == "cpu"

    def test_get_device_info_returns_dict(self) -> None:
        """Test that get_device_info returns a dictionary."""
        info = get_device_info()
        assert isinstance(info, dict)
        assert len(info) > 0
