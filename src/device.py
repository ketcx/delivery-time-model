"""Device utilities for Mac M1/MPS support and fallback."""

import torch


def is_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available.

    Returns:
        True if MPS is available and supported, False otherwise
    """
    try:
        return torch.backends.mps.is_available()
    except AttributeError:
        return False


def is_mps_supported() -> bool:
    """Check if MPS is supported by PyTorch.

    Returns:
        True if MPS is supported, False otherwise
    """
    try:
        return torch.backends.mps.is_built()
    except AttributeError:
        return False


def get_device(
    preferred_device: str = "mps",
    fallback_on_error: bool = True,
) -> str:
    """Get appropriate device with fallback support.

    Args:
        preferred_device: Preferred device (mps, cpu, cuda)
        fallback_on_error: If True, fall back to CPU if preferred device fails

    Returns:
        Device string (mps, cpu, or cuda)
    """
    if preferred_device == "mps":
        if is_mps_available():
            return "mps"
        elif fallback_on_error:
            return "cpu"
        else:
            raise RuntimeError("MPS not available and fallback disabled")

    elif preferred_device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        elif fallback_on_error:
            return "cpu"
        else:
            raise RuntimeError("CUDA not available and fallback disabled")

    return "cpu"


def get_device_info() -> dict:
    """Get detailed device information.

    Returns:
        Dictionary with device details
    """
    info = {
        "mps_available": is_mps_available(),
        "mps_supported": is_mps_supported(),
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "device_count": {
            "cuda": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    }

    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)

    return info


def get_dtype_from_string(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype.

    Args:
        dtype_str: String representation (fp32, fp16, bf16)

    Returns:
        torch.dtype

    Raises:
        ValueError: If dtype string is not recognized
    """
    dtype_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }

    dtype_str_lower = dtype_str.lower()
    if dtype_str_lower not in dtype_map:
        raise ValueError(
            f"Unknown dtype: {dtype_str}. " f"Supported: {', '.join(dtype_map.keys())}"
        )

    return dtype_map[dtype_str_lower]


def check_mps_limitations(device: str, dtype: torch.dtype, logger=None) -> None:
    """Check and warn about MPS limitations.

    Args:
        device: Device being used (mps, cpu, cuda)
        dtype: Data type being used
        logger: Optional logger for warnings
    """
    if device != "mps":
        return

    warnings = []

    # MPS limitations
    if dtype == torch.bfloat16:
        warnings.append(
            "BF16 (bfloat16) has limited support on MPS. "
            "Consider using FP16 or FP32."
        )

    if warnings and logger:
        for warning in warnings:
            logger.warning(f"MPS Limitation: {warning}")


class DeviceContext:
    """Context manager for device operations with fallback support."""

    def __init__(
        self,
        preferred_device: str = "mps",
        dtype: str = "fp32",
        fallback_on_error: bool = True,
    ):
        """Initialize device context.

        Args:
            preferred_device: Preferred device (mps, cpu, cuda)
            dtype: Data type (fp32, fp16, bf16)
            fallback_on_error: Enable fallback to CPU on error
        """
        self.preferred_device = preferred_device
        self.dtype = get_dtype_from_string(dtype)
        self.fallback_on_error = fallback_on_error
        self.device = get_device(preferred_device, fallback_on_error)
        self.original_device = None
        self.original_dtype = None

    def __enter__(self):
        """Enter context."""
        self.original_device = (
            torch.get_default_device() if hasattr(torch, "get_default_device") else None
        )
        torch.set_default_dtype(self.dtype)
        return self.device

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if self.original_dtype:
            torch.set_default_dtype(self.original_dtype)
        return False


def print_device_stats(device: str) -> None:
    """Print device statistics.

    Args:
        device: Device to get stats for
    """
    print(f"Device: {device}")

    if device == "mps":
        print("  Metal Performance Shaders (Apple Silicon)")
    elif device == "cuda":
        print(f"  CUDA - {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM Available: {vram_gb:.2f} GB")
    else:
        print("  CPU")
