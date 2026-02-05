"""System information collection for reproducibility."""

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class DeviceInfo:
    """Device information."""

    type: str  # mps, cpu, cuda
    name: str
    available_memory_gb: float


@dataclass
class SystemInfo:
    """System information for reproducibility."""

    python_version: str
    torch_version: str
    platform: str
    architecture: str
    device: DeviceInfo
    git_commit: str
    git_branch: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["device"] = asdict(self.device)
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str) -> None:
        """Save system info to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())


def get_device_info(device_type: str) -> DeviceInfo:
    """Get device information."""
    try:
        import torch

        if device_type == "mps":
            if torch.backends.mps.is_available():
                # macOS with MPS
                return DeviceInfo(
                    type="mps",
                    name=platform.processor(),
                    available_memory_gb=0.0,  # MPS doesn't expose memory directly
                )
            else:
                # Fallback to CPU
                return DeviceInfo(
                    type="cpu",
                    name=platform.processor(),
                    available_memory_gb=0.0,
                )
        else:
            return DeviceInfo(
                type="cpu",
                name=platform.processor(),
                available_memory_gb=0.0,
            )
    except ImportError:
        return DeviceInfo(
            type="cpu",
            name=platform.processor(),
            available_memory_gb=0.0,
        )


def get_git_info() -> tuple[str, str]:
    """Get current git commit hash and branch."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        branch = "unknown"

    return commit, branch


def get_system_info(device_type: str) -> SystemInfo:
    """Collect system information."""
    import torch

    commit, branch = get_git_info()
    device_info = get_device_info(device_type)

    return SystemInfo(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        platform=platform.system(),
        architecture=platform.machine(),
        device=device_info,
        git_commit=commit,
        git_branch=branch,
    )
