"""Custom wrappers for retro environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class PlatformerRewardWrapper(gym.Wrapper):
    """Potential-based reward shaping using x-position progress.

    Adds a shaped reward component: scale * (x_new - x_old) to encourage
    rightward progress in platformer games. Uses info dict keys from
    stable-retro's data.json (e.g. ``info['x']`` for Sonic,
    ``info['xscrollLo'] + info['xscrollHi'] * 256`` for Mario).

    The shaping is potential-based (PBRS) so it preserves optimal policies.
    """

    # Mapping from game prefix to the callable that extracts x-position from info.
    _EXTRACTORS: dict[str, Any] = {
        "Sonic": lambda info: info.get("x", 0),
        "SuperMarioBros": lambda info: info.get("xscrollLo", 0)
        + info.get("xscrollHi", 0) * 256,
    }

    def __init__(self, env: gym.Env, scale: float = 0.1, game: str = "") -> None:
        """Initialize the wrapper.

        Args:
            env: Environment to wrap.
            scale: Multiplier for (x_new - x_old). Use 0 to disable.
            game: Retro game ID used to select the x-position extractor.
        """
        super().__init__(env)
        self.scale = scale
        self._x_old: float = 0.0
        self._extractor = self._resolve_extractor(game)

    @classmethod
    def _resolve_extractor(cls, game: str):
        """Pick the right info-key extractor based on the game name."""
        for prefix, fn in cls._EXTRACTORS.items():
            if game.startswith(prefix):
                return fn
        # Fallback: try common key 'x', then 0
        return lambda info: info.get("x", 0)

    def step(self, action: Any) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_new = float(self._extractor(info))
        shaped = self.scale * (x_new - self._x_old)
        self._x_old = x_new

        info["raw_reward"] = reward
        info["shaped_reward"] = shaped
        return obs, reward + shaped, terminated, truncated, info

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        self._x_old = float(self._extractor(info))
        return obs, info


class TimeLimitWrapper(gym.Wrapper):
    """Truncate episodes after a fixed number of steps.

    Useful for fighting games (e.g. MK2) where rounds can loop indefinitely.
    Sets ``truncated=True`` when the step budget is exhausted.
    """

    def __init__(self, env: gym.Env, max_steps: int = 10000) -> None:
        super().__init__(env)
        self.max_steps = max_steps
        self._step_count = 0

    def step(self, action: Any) -> tuple:
        self._step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._step_count >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> tuple:
        self._step_count = 0
        return self.env.reset(**kwargs)
