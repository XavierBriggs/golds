"""Tests for the retro environment maker's FrameSkip wrapper.

Covers Tier-1 stochastic frame skip (research-workspace/REPORT.md): the
Sonic benchmark recipe samples frame skip from {2,3,4} each step instead of
a fixed 4, so the timing noise breaks deterministic freeze/oscillation
failure modes. ROM-free: exercises FrameSkip directly against a fake env.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from golds.environments.retro.maker import FrameSkip


class _CountingEnv(gym.Env):
    """Minimal fake env: obs value == cumulative step count, reward=1.0/step."""

    def __init__(self, done_at: int | None = None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, (4, 4, 1), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self._step_count = 0
        self._done_at = done_at
        self.step_calls = 0

    def reset(self, **kwargs):
        self._step_count = 0
        self.step_calls = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self._step_count += 1
        self.step_calls += 1
        obs = np.full(self.observation_space.shape, self._step_count % 256, dtype=np.uint8)
        terminated = self._done_at is not None and self._step_count >= self._done_at
        return obs, 1.0, terminated, False, {}


class TestFrameSkipFixed:
    """Backward-compatible fixed-skip behavior (default, unchanged)."""

    def test_fixed_skip_sums_reward_and_calls_n_times(self):
        inner = _CountingEnv()
        env = FrameSkip(inner, skip=4)
        env.reset()
        _, reward, terminated, truncated, _ = env.step(0)
        assert inner.step_calls == 4
        assert reward == pytest.approx(4.0)
        assert not terminated
        assert not truncated

    def test_fixed_skip_stops_early_on_done(self):
        inner = _CountingEnv(done_at=2)
        env = FrameSkip(inner, skip=4)
        env.reset()
        _, reward, terminated, _truncated, _ = env.step(0)
        assert inner.step_calls == 2
        assert reward == pytest.approx(2.0)
        assert terminated

    def test_max_pool_observation_over_last_two_frames(self):
        # obs value == step_count (monotonically increasing), so max-pooling
        # the last two frames of a 4-skip call should equal the final frame.
        inner = _CountingEnv()
        env = FrameSkip(inner, skip=4)
        env.reset()
        obs, _, _, _, _ = env.step(0)
        assert obs.max() == 4

    def test_stochastic_defaults_to_false(self):
        inner = _CountingEnv()
        env = FrameSkip(inner)
        assert env.stochastic is False
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert inner.step_calls == 4  # default skip=4, unaffected
        assert reward == pytest.approx(4.0)


class TestStochasticFrameSkip:
    """Stochastic frame skip: sample uniformly from [skip_min, skip_max]."""

    def test_default_range_is_2_to_4(self):
        inner = _CountingEnv()
        env = FrameSkip(inner, stochastic=True)
        assert env.skip_min == 2
        assert env.skip_max == 4

    def test_samples_within_range_and_sums_reward_correctly(self):
        inner = _CountingEnv()
        env = FrameSkip(inner, stochastic=True, skip_min=2, skip_max=4)
        env.reset(seed=123)

        for _ in range(30):
            inner.step_calls = 0
            _, reward, _, _, _ = env.step(0)
            assert 2 <= inner.step_calls <= 4
            assert reward == pytest.approx(float(inner.step_calls))

    def test_uses_full_configured_range_over_many_samples(self):
        inner = _CountingEnv()
        env = FrameSkip(inner, stochastic=True, skip_min=2, skip_max=4)
        env.reset(seed=42)

        seen = set()
        for _ in range(200):
            inner.step_calls = 0
            env.step(0)
            seen.add(inner.step_calls)
        assert seen == {2, 3, 4}

    def test_deterministic_given_same_seed(self):
        inner_a = _CountingEnv()
        env_a = FrameSkip(inner_a, stochastic=True, skip_min=2, skip_max=4)
        env_a.reset(seed=7)
        skips_a = []
        for _ in range(15):
            inner_a.step_calls = 0
            env_a.step(0)
            skips_a.append(inner_a.step_calls)

        inner_b = _CountingEnv()
        env_b = FrameSkip(inner_b, stochastic=True, skip_min=2, skip_max=4)
        env_b.reset(seed=7)
        skips_b = []
        for _ in range(15):
            inner_b.step_calls = 0
            env_b.step(0)
            skips_b.append(inner_b.step_calls)

        assert skips_a == skips_b

    def test_done_mid_skip_stops_early_and_sums_partial_reward(self):
        inner = _CountingEnv(done_at=1)
        env = FrameSkip(inner, stochastic=True, skip_min=2, skip_max=4)
        env.reset(seed=1)
        _, reward, terminated, _truncated, _ = env.step(0)
        assert inner.step_calls == 1
        assert reward == pytest.approx(1.0)
        assert terminated

    def test_custom_range_respected(self):
        inner = _CountingEnv()
        env = FrameSkip(inner, stochastic=True, skip_min=3, skip_max=3)
        env.reset(seed=0)
        for _ in range(10):
            inner.step_calls = 0
            env.step(0)
            assert inner.step_calls == 3
