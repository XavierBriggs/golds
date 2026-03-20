"""Learning rate and clip range schedule functions for PPO training."""

from __future__ import annotations

import math
from collections.abc import Callable


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear decay from initial_value to 0.

    Compatible with SB3's schedule interface where the argument is
    progress_remaining (1.0 at start, 0.0 at end).

    Args:
        initial_value: Starting value.

    Returns:
        Schedule function.
    """

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


def cosine_schedule(initial_value: float) -> Callable[[float], float]:
    """Cosine annealing from initial_value to 0.

    Uses cosine decay: value = initial * (1 + cos(pi * (1 - progress))) / 2

    Args:
        initial_value: Starting value.

    Returns:
        Schedule function.
    """

    def schedule(progress_remaining: float) -> float:
        return initial_value * (1 + math.cos(math.pi * (1 - progress_remaining))) / 2

    return schedule
