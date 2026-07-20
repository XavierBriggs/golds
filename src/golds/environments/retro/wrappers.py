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


_PROGRESS_MODES = ("delta_x", "delta_max_x")


class PlatformerRewardWrapper(gym.Wrapper):
    """Potential-based reward shaping with multiple signals.

    Signals (all optional, controlled by constructor args):
    - X-position progress: either raw delta-x or delta-max(x), see ``progress_mode``
    - Death penalty: applied on ``terminated=True`` (not truncation)
    - Collectible bonus: delta in rings/coins from info dict
    - Time penalty: small per-step negative reward
    - Completion bonus: one-time reward when level completion is first detected

    ``progress_mode`` controls the X-position term:
    - ``"delta_x"`` (default, backward compatible): ``scale * (x_new - x_old)``.
      Punishes backtracking. This is the pre-ADR-004 behavior, kept as the
      default so other retro platformer configs are unaffected.
    - ``"delta_max_x"``: ``scale * max(0.0, x_new - max_x_so_far)``. Only NEW
      furthest-right progress is rewarded; backtracking is never punished and
      re-covering old ground earns nothing. This is the standard Sonic recipe
      (ADR-004 / R3): it fixes the failure mode where the agent gets stuck and
      needs to backtrack past a trap.

    Level completion (ADR-004 / R4) is detected when EITHER:
    - ``x_new >= level_end_x`` (a configurable per-level threshold), or
    - ``info.get(level_end_info_key)`` is truthy, if ``level_end_info_key`` is set
      (e.g. a level/act-change flag, if stable-retro's data.json exposes one).

    ``level_end_x`` defaults to ``None``, meaning threshold-based completion is
    DISABLED, so nothing silently reports false completion. Sonic x-position
    can be large (and can in principle loop/wrap within a level), so this
    threshold must be set from real ROM data once available (see ADR-004
    consequences) rather than guessed.

    ``completion_bonus`` is awarded exactly once, the first step completion is
    detected in an episode.

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
        progress_mode: str = "delta_x",
        level_end_x: float | None = None,
        completion_bonus: float = 0.0,
        level_end_info_key: str | None = None,
        terminate_on_completion: bool = True,
    ) -> None:
        super().__init__(env)
        if progress_mode not in _PROGRESS_MODES:
            raise ValueError(
                f"Unknown progress_mode: {progress_mode!r}. Choose from {_PROGRESS_MODES}"
            )
        self.scale = scale
        self.death_penalty = death_penalty
        self.collectible_reward_scale = collectible_reward_scale
        self.time_penalty = time_penalty
        self.progress_mode = progress_mode
        # NOTE: the exact GHZ Act 1 end-x value is unknown until the ROM is
        # available on ithaca (see docs/inception/spec.md M0). None means
        # threshold-based completion is disabled; do not guess this value.
        self.level_end_x = level_end_x
        self.completion_bonus = completion_bonus
        self.level_end_info_key = level_end_info_key
        self.terminate_on_completion = terminate_on_completion
        self._x_old: float = 0.0
        self._max_x: float = 0.0
        self._collectible_old: float = 0.0
        self._completed: bool = False
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
        if self.progress_mode == "delta_max_x":
            x_shaped = self.scale * max(0.0, x_new - self._max_x)
        else:  # "delta_x" (default, backward compatible)
            x_shaped = self.scale * (x_new - self._x_old)
        self._x_old = x_new
        self._max_x = max(self._max_x, x_new)

        # Death penalty (only on true termination, not truncation)
        death = self.death_penalty if terminated and not truncated else 0.0

        # Collectible bonus (delta)
        collect_new = float(self._collectible_extractor(info))
        collect_delta = max(0.0, collect_new - self._collectible_old)
        collect_shaped = self.collectible_reward_scale * collect_delta
        self._collectible_old = collect_new

        # Time penalty
        time_shaped = self.time_penalty

        # Level completion (R4): threshold OR info-key signal, one-time bonus.
        completion_shaped = 0.0
        just_completed = False
        if not self._completed:
            threshold_hit = self.level_end_x is not None and x_new >= self.level_end_x
            info_key_hit = self.level_end_info_key is not None and bool(
                info.get(self.level_end_info_key)
            )
            if threshold_hit or info_key_hit:
                self._completed = True
                just_completed = True
                completion_shaped = self.completion_bonus

        # Terminate the episode at the signpost: the objective is to COMPLETE
        # the level, not run past it into the next act. This also keeps eval
        # episodes bounded (a completing agent otherwise plays until it dies).
        if just_completed and self.terminate_on_completion:
            terminated = True

        total_shaped = x_shaped + death + collect_shaped + time_shaped + completion_shaped

        info["raw_reward"] = reward
        info["shaped_reward"] = total_shaped
        info["shaped_x_progress"] = x_shaped
        info["shaped_death"] = death
        info["shaped_collectible"] = collect_shaped
        info["shaped_time"] = time_shaped
        info["shaped_completion_bonus"] = completion_shaped
        info["level_complete"] = self._completed

        return obs, reward + total_shaped, terminated, truncated, info

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        self._x_old = float(self._extractor(info))
        self._max_x = self._x_old
        self._collectible_old = float(self._collectible_extractor(info))
        self._completed = False
        return obs, info

    def get_episode_progress(self) -> dict[str, Any]:
        """Return unclipped, raw episode progress (R5).

        Queryable after (or during) an episode so an eval loop can compute a
        completion RATE over N episodes without depending on clipped/shaped
        reward. Reset to a fresh state on every ``reset()``.

        Returns:
            dict with keys:
            - ``completed``: bool, whether level completion was detected this episode.
            - ``max_x``: float, the furthest-right raw x-position reached this episode.
            - ``level_end_x``: float | None, the configured completion threshold
              (None if threshold-based completion is disabled).
        """
        return {
            "completed": self._completed,
            "max_x": self._max_x,
            "level_end_x": self.level_end_x,
        }


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
