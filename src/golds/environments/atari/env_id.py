"""Atari env-id utilities.

In Gymnasium, Atari env ids can appear in two common forms (when `ale-py` is installed):
- Legacy Gym-style ids (e.g. `SpaceInvadersNoFrameskip-v4`)
- ALE v5 ids (e.g. `ALE/SpaceInvaders-v5`)

This module provides a tiny compatibility layer for callers that omit the
version suffix (e.g. `SpaceInvadersNoFrameskip`).
"""

from __future__ import annotations

import re

_LEGACY_ATARI_ID_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9]+)(?P<noframeskip>NoFrameskip)?(?:-v(?P<version>\d+))?$"
)


def resolve_atari_env_id(env_id: str) -> str:
    """Normalize an Atari env id.

    Args:
        env_id: Either an `ALE/...` id or a legacy Gym-style id.

    Returns:
        The normalized env id.
    """
    if env_id.startswith("ALE/"):
        return env_id

    match = _LEGACY_ATARI_ID_RE.match(env_id)
    if not match:
        return env_id

    # If a version is already present, keep as-is.
    if match.group("version") is not None:
        return env_id

    # Default to the common SB3-compatible env version.
    return f"{env_id}-v4"
