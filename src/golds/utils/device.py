"""Device detection utilities."""

from __future__ import annotations

import torch


def get_device(device: str = "auto") -> str:
    """Get the appropriate device for training.

    Args:
        device: Device specification. 'auto' detects the best available device.
            Priority: cuda > mps > cpu.

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def get_device_info() -> dict[str, str | bool | int]:
    """Get information about available devices.

    Returns:
        Dictionary with device information
    """
    info: dict[str, str | bool | int] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda or "unknown"

    if torch.backends.mps.is_available():
        info["mps_device_name"] = "Apple Silicon GPU"

    return info
