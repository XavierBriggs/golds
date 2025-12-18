"""Device detection utilities."""

from __future__ import annotations

import torch


def get_device(device: str = "auto") -> str:
    """Get the appropriate device for training.

    Args:
        device: Device specification. 'auto' will detect CUDA availability.

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def get_device_info() -> dict[str, str | bool | int]:
    """Get information about available devices.

    Returns:
        Dictionary with device information
    """
    info: dict[str, str | bool | int] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda or "unknown"

    return info
