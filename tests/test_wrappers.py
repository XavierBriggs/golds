"""Tests for retro environment wrappers."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest  # noqa: I001

# ---------------------------------------------------------------------------
# Helpers — lightweight fake env that behaves like a retro env
# ---------------------------------------------------------------------------


class _FakeRetroEnv(gym.Env):
    """Minimal env that returns configurable info dicts (simulating retro)."""

    def __init__(self, info_sequence: list[dict] | None = None):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(7)
        self._info_seq = info_sequence or []
        self._step_idx = 0

    def reset(self, **kwargs):
        self._step_idx = 0
        info = self._info_seq[0] if self._info_seq else {}
        return np.zeros(self.observation_space.shape, dtype=np.uint8), info

    def step(self, action):
        self._step_idx += 1
        idx = min(self._step_idx, len(self._info_seq) - 1) if self._info_seq else 0
        info = self._info_seq[idx] if self._info_seq else {}
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs, 1.0, False, False, info


# ---------------------------------------------------------------------------
# PlatformerRewardWrapper tests
# ---------------------------------------------------------------------------


class TestPlatformerRewardWrapper:
    def test_shaped_reward_increases_with_forward_progress(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 0}, {"x": 10}, {"x": 25}]
        env = PlatformerRewardWrapper(_FakeRetroEnv(infos), scale=0.1, game="SonicTheHedgehog-Genesis")
        env.reset()

        _, reward1, _, _, info1 = env.step(0)
        # raw_reward=1.0, shaped = 0.1 * (10 - 0) = 1.0
        assert reward1 == pytest.approx(2.0)
        assert info1["raw_reward"] == pytest.approx(1.0)
        assert info1["shaped_reward"] == pytest.approx(1.0)

        _, reward2, _, _, info2 = env.step(0)
        # raw_reward=1.0, shaped = 0.1 * (25 - 10) = 1.5
        assert reward2 == pytest.approx(2.5)

    def test_backward_movement_gives_negative_shaping(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 100}, {"x": 90}]
        env = PlatformerRewardWrapper(_FakeRetroEnv(infos), scale=0.1, game="SonicTheHedgehog-Genesis")
        env.reset()

        _, reward, _, _, _ = env.step(0)
        # raw=1.0, shaped = 0.1 * (90 - 100) = -1.0
        assert reward == pytest.approx(0.0)

    def test_mario_extractor_uses_scroll_registers(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [
            {"xscrollLo": 0, "xscrollHi": 0},
            {"xscrollLo": 0, "xscrollHi": 1},  # 256 pixels
        ]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos), scale=0.01, game="SuperMarioBros-Nes"
        )
        env.reset()

        _, reward, _, _, info = env.step(0)
        # shaped = 0.01 * (256 - 0) = 2.56
        assert info["shaped_reward"] == pytest.approx(2.56)

    def test_zero_scale_disables_shaping(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 0}, {"x": 100}]
        env = PlatformerRewardWrapper(_FakeRetroEnv(infos), scale=0.0, game="SonicTheHedgehog-Genesis")
        env.reset()

        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(1.0)  # Only raw reward

    def test_reset_resets_x_old(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 50}, {"x": 60}]
        env = PlatformerRewardWrapper(_FakeRetroEnv(infos), scale=0.1, game="SonicTheHedgehog-Genesis")
        env.reset()
        assert env._x_old == 50.0

        env.step(0)
        assert env._x_old == 60.0

        env.reset()
        assert env._x_old == 50.0


# ---------------------------------------------------------------------------
# TimeLimitWrapper tests
# ---------------------------------------------------------------------------


class TestTimeLimitWrapper:
    def test_truncates_at_max_steps(self):
        from golds.environments.retro.wrappers import TimeLimitWrapper

        env = TimeLimitWrapper(_FakeRetroEnv(), max_steps=5)
        env.reset()

        for i in range(4):
            _, _, _, truncated, _ = env.step(0)
            assert not truncated, f"Should not truncate at step {i+1}"

        _, _, _, truncated, _ = env.step(0)
        assert truncated, "Should truncate at step 5"

    def test_reset_clears_step_count(self):
        from golds.environments.retro.wrappers import TimeLimitWrapper

        env = TimeLimitWrapper(_FakeRetroEnv(), max_steps=3)
        env.reset()

        for _ in range(3):
            env.step(0)

        env.reset()
        # After reset we should get 3 more steps
        for i in range(2):
            _, _, _, truncated, _ = env.step(0)
            assert not truncated

        _, _, _, truncated, _ = env.step(0)
        assert truncated

    def test_preserves_terminated_flag(self):
        """TimeLimitWrapper should not mask a terminated=True from the inner env."""
        from golds.environments.retro.wrappers import TimeLimitWrapper

        class _TerminatingEnv(_FakeRetroEnv):
            def step(self, action):
                obs, reward, _, truncated, info = super().step(action)
                return obs, reward, True, truncated, info  # always terminated

        env = TimeLimitWrapper(_TerminatingEnv(), max_steps=100)
        env.reset()
        _, _, terminated, _, _ = env.step(0)
        assert terminated
