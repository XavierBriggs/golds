"""Custom wrappers for retro environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------------------
# Action set definitions
# ---------------------------------------------------------------------------

# Each entry is (human_name, list_of_button_names_to_press).
# Button names match env.unwrapped.buttons (e.g. "A", "B", "RIGHT").
ACTION_SETS: dict[str, list[tuple[str, list[str]]]] = {
    "platformer": [
        ("NOOP", []),
        ("Left", ["LEFT"]),
        ("Right", ["RIGHT"]),
        ("Down", ["DOWN"]),
        ("Right+B", ["RIGHT", "B"]),
        ("Down+B", ["DOWN", "B"]),
        ("A", ["A"]),
        ("Right+A", ["RIGHT", "A"]),
        ("Right+A+B", ["RIGHT", "A", "B"]),
    ],
    "fighter": [
        ("NOOP", []),
        ("Left", ["LEFT"]),
        ("Right", ["RIGHT"]),
        ("Up", ["UP"]),
        ("Down", ["DOWN"]),
        ("A", ["A"]),
        ("B", ["B"]),
        ("Down+A", ["DOWN", "A"]),
        ("Down+B", ["DOWN", "B"]),
        ("Left+A", ["LEFT", "A"]),
        ("Right+A", ["RIGHT", "A"]),
        ("Right+B", ["RIGHT", "B"]),
    ],
    "puzzle": [
        ("NOOP", []),
        ("Left", ["LEFT"]),
        ("Right", ["RIGHT"]),
        ("Down", ["DOWN"]),
        ("A", ["A"]),
    ],
}


class DiscreteActionWrapper(gym.Wrapper):
    """Maps a small Discrete action space to MultiBinary button presses.

    Retro games expose ~126 filtered button combos. This wrapper reduces the
    action space to 5-12 meaningful actions per genre, dramatically improving
    sample efficiency.
    """

    def __init__(self, env: gym.Env, action_set: str = "platformer") -> None:
        super().__init__(env)
        if action_set not in ACTION_SETS:
            raise ValueError(f"Unknown action_set: {action_set}. Choose from {list(ACTION_SETS)}")

        buttons = env.unwrapped.buttons
        n_buttons = len(buttons)
        combos = ACTION_SETS[action_set]

        # Build lookup table: action_index → MultiBinary array
        self._lookup = np.zeros((len(combos), n_buttons), dtype=np.int8)
        for i, (_name, btn_names) in enumerate(combos):
            for btn in btn_names:
                if btn in buttons:
                    self._lookup[i, buttons.index(btn)] = 1

        self.action_space = gym.spaces.Discrete(len(combos))

    def step(self, action: int) -> tuple:
        return self.env.step(self._lookup[action])

    def reset(self, **kwargs) -> tuple:
        return self.env.reset(**kwargs)


class StickyActionWrapper(gym.Wrapper):
    """Repeat the previous action with probability stickprob.

    Breaks determinism in retro games so the agent cannot memorize
    frame-perfect action sequences. Standard practice: stickprob=0.25.
    """

    def __init__(self, env: gym.Env, stickprob: float = 0.25) -> None:
        super().__init__(env)
        self.stickprob = stickprob
        self._last_action = None
        self._rng = np.random.default_rng()

    def step(self, action: Any) -> tuple:
        if self._last_action is not None and self._rng.random() < self.stickprob:
            action = self._last_action
        self._last_action = action
        return self.env.step(action)

    def reset(self, **kwargs) -> tuple:
        self._last_action = None
        seed = kwargs.get("seed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self.env.reset(**kwargs)


class MultiLevelWrapper(gym.Wrapper):
    """Randomly select a level on each episode reset.

    On ``reset()``, picks a random level from the configured list and sets
    the retro emulator's ``initial_state`` so the subsequent reset loads
    the correct level.
    """

    def __init__(self, env: gym.Env, levels: list[str]) -> None:
        super().__init__(env)
        if not levels:
            raise ValueError("levels must be a non-empty list")
        self.levels = levels
        self._rng = np.random.default_rng()

    def reset(self, **kwargs) -> tuple:
        level = self._rng.choice(self.levels)
        inner = self.env.unwrapped
        inner.load_state(level)
        inner.initial_state = inner.em.get_state()
        return self.env.reset(**kwargs)


class PlatformerRewardWrapper(gym.Wrapper):
    """Potential-based reward shaping with multiple signals.

    Signals (all optional, controlled by constructor args):
    - X-position progress: ``scale * (x_new - x_old)``
    - Death penalty: applied on ``terminated=True`` (not truncation)
    - Collectible bonus: delta in rings/coins from info dict
    - Time penalty: small per-step negative reward

    Uses info dict keys from stable-retro's data.json.
    """

    _EXTRACTORS: dict[str, Any] = {
        "Sonic": lambda info: info.get("x", 0),
        "SuperMarioBros": lambda info: info.get("xscrollLo", 0)
        + info.get("xscrollHi", 0) * 256,
    }

    _COLLECTIBLE_EXTRACTORS: dict[str, Any] = {
        "Sonic": lambda info: info.get("rings", 0),
        "SuperMarioBros": lambda info: info.get("coins", 0),
    }

    def __init__(
        self,
        env: gym.Env,
        scale: float = 0.1,
        game: str = "",
        death_penalty: float = 0.0,
        collectible_reward_scale: float = 0.0,
        time_penalty: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.scale = scale
        self.death_penalty = death_penalty
        self.collectible_reward_scale = collectible_reward_scale
        self.time_penalty = time_penalty
        self._x_old: float = 0.0
        self._collectible_old: float = 0.0
        self._extractor = self._resolve_extractor(game)
        self._collectible_extractor = self._resolve_collectible_extractor(game)

    @classmethod
    def _resolve_extractor(cls, game: str):
        for prefix, fn in cls._EXTRACTORS.items():
            if game.startswith(prefix):
                return fn
        return lambda info: info.get("x", 0)

    @classmethod
    def _resolve_collectible_extractor(cls, game: str):
        for prefix, fn in cls._COLLECTIBLE_EXTRACTORS.items():
            if game.startswith(prefix):
                return fn
        return lambda info: 0

    def step(self, action: Any) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # X-position shaping
        x_new = float(self._extractor(info))
        x_shaped = self.scale * (x_new - self._x_old)
        self._x_old = x_new

        # Death penalty (only on true termination, not truncation)
        death = self.death_penalty if terminated and not truncated else 0.0

        # Collectible bonus (delta)
        collect_new = float(self._collectible_extractor(info))
        collect_delta = max(0.0, collect_new - self._collectible_old)
        collect_shaped = self.collectible_reward_scale * collect_delta
        self._collectible_old = collect_new

        # Time penalty
        time_shaped = self.time_penalty

        total_shaped = x_shaped + death + collect_shaped + time_shaped

        info["raw_reward"] = reward
        info["shaped_reward"] = total_shaped
        info["shaped_x_progress"] = x_shaped
        info["shaped_death"] = death
        info["shaped_collectible"] = collect_shaped
        info["shaped_time"] = time_shaped

        return obs, reward + total_shaped, terminated, truncated, info

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        self._x_old = float(self._extractor(info))
        self._collectible_old = float(self._collectible_extractor(info))
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
