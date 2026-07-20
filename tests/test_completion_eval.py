"""Tests for the completion-rate evaluation (R11 / north-star goal G6).

These tests are fully synthetic: no real model, ROM, or emulator is
involved. A fake "model" and a fake SB3-style VecEnv are used to exercise
``evaluate_completion_rate`` in isolation, matching the info-dict-based
detection contract:

- ``info["level_complete"]`` is read from the vectorized info dict/list
  returned by ``VecEnv.step()`` (never by reaching into the wrapper object).
- An episode counts as "completed" if ``level_complete`` was True on ANY
  step of that episode, even if the episode continues afterward.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from golds.evaluation.completion import evaluate_completion_rate


class _FakeModel:
    """Stand-in for an SB3 model; action content is irrelevant to the env."""

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        return np.array([0] * _n_envs_from_obs(obs)), None


def _n_envs_from_obs(obs: Any) -> int:
    # obs is a stacked array of shape (n_envs, ...); fall back to 1.
    if hasattr(obs, "shape") and len(obs.shape) > 0:
        return obs.shape[0]
    return 1


class _FakeVecEnv:
    """Minimal SB3-VecEnv-shaped fake, single sub-env (n_envs=1).

    ``episode_specs`` is a list of per-episode step scripts: each entry is a
    list of ``(level_complete, done)`` tuples describing one step's info
    stream. The env auto-resets on ``done`` like a real SB3 VecEnv.
    """

    num_envs = 1

    def __init__(self, episode_specs: list[list[tuple[bool, bool]]]) -> None:
        self._episode_specs = episode_specs
        self._ep_idx = 0
        self._step_idx = 0

    def reset(self) -> np.ndarray:
        self._ep_idx = 0
        self._step_idx = 0
        return np.zeros((1, 4), dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple:
        spec = self._episode_specs[self._ep_idx]
        level_complete, done = spec[self._step_idx]
        info: dict[str, Any] = {"level_complete": level_complete}

        obs = np.zeros((1, 4), dtype=np.float32)
        rewards = np.array([1.0])
        dones = np.array([done])
        infos = [info]

        self._step_idx += 1
        if done:
            self._ep_idx += 1
            self._step_idx = 0

        return obs, rewards, dones, infos


def _episode(n_steps: int, complete_on_step: int | None) -> list[tuple[bool, bool]]:
    """Build a single-episode step script.

    ``complete_on_step`` is the 0-indexed step at which level_complete first
    becomes True (None means never). The episode always runs ``n_steps``
    steps and ``done=True`` only on the last one.
    """
    steps = []
    for i in range(n_steps):
        complete = complete_on_step is not None and i >= complete_on_step
        done = i == n_steps - 1
        steps.append((complete, done))
    return steps


def test_completion_rate_hand_computed_k_of_n() -> None:
    """K of N episodes complete => completion_rate == K/N."""
    # 3 completed (complete_on_step set), 2 not completed (None), N=5.
    specs = [
        _episode(3, complete_on_step=1),
        _episode(3, complete_on_step=None),
        _episode(2, complete_on_step=0),
        _episode(4, complete_on_step=None),
        _episode(3, complete_on_step=2),
    ]
    env = _FakeVecEnv(specs)
    model = _FakeModel()

    result = evaluate_completion_rate(model, env, n_episodes=5, deterministic=True)

    assert result["n_episodes"] == 5
    assert result["n_completed"] == 3
    assert result["completion_rate"] == pytest.approx(3 / 5)


def test_completion_flag_mid_episode_still_counts_after_continuing() -> None:
    """level_complete=True mid-episode, then more steps follow => still completed."""
    # Completion fires on step 0 of a 5-step episode; episode keeps running
    # for 4 more steps past the signpost (matches real wrapper semantics).
    specs = [_episode(5, complete_on_step=0)]
    env = _FakeVecEnv(specs)
    model = _FakeModel()

    result = evaluate_completion_rate(model, env, n_episodes=1, deterministic=True)

    assert result["n_completed"] == 1
    assert result["completion_rate"] == 1.0
    assert result["per_episode"][0]["completed"] is True


def test_zero_completions_yields_zero_rate() -> None:
    specs = [
        _episode(2, complete_on_step=None),
        _episode(3, complete_on_step=None),
        _episode(1, complete_on_step=None),
    ]
    env = _FakeVecEnv(specs)
    model = _FakeModel()

    result = evaluate_completion_rate(model, env, n_episodes=3, deterministic=True)

    assert result["n_completed"] == 0
    assert result["completion_rate"] == 0.0


def test_all_completions_yields_full_rate() -> None:
    specs = [
        _episode(2, complete_on_step=0),
        _episode(3, complete_on_step=1),
        _episode(1, complete_on_step=0),
    ]
    env = _FakeVecEnv(specs)
    model = _FakeModel()

    result = evaluate_completion_rate(model, env, n_episodes=3, deterministic=True)

    assert result["n_completed"] == 3
    assert result["completion_rate"] == 1.0


def test_returns_expected_shape_and_mean_reward() -> None:
    specs = [
        _episode(3, complete_on_step=1),
        _episode(2, complete_on_step=None),
    ]
    env = _FakeVecEnv(specs)
    model = _FakeModel()

    result = evaluate_completion_rate(model, env, n_episodes=2, deterministic=True)

    assert set(
        ["completion_rate", "n_episodes", "n_completed", "mean_reward", "per_episode"]
    ).issubset(result.keys())
    # Each step in the fake env yields reward 1.0.
    assert result["mean_reward"] == pytest.approx((3 + 2) / 2)
    assert len(result["per_episode"]) == 2
    for ep in result["per_episode"]:
        assert "completed" in ep
        assert "reward" in ep


class _VectorizedFakeVecEnv:
    """SB3-style VecEnv with n_envs=2 to exercise info-as-list-of-dicts.

    Both sub-envs run independently; each auto-resets to a fresh episode
    from its own queue when done, matching real VecEnv semantics.
    """

    num_envs = 2

    def __init__(self, queues: list[list[list[tuple[bool, bool]]]]) -> None:
        # queues[env_idx] is a list of episode specs for that sub-env.
        self._queues = queues
        self._ep_idx = [0, 0]
        self._step_idx = [0, 0]

    def reset(self) -> np.ndarray:
        self._ep_idx = [0, 0]
        self._step_idx = [0, 0]
        return np.zeros((2, 4), dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple:
        infos: list[dict[str, Any]] = []
        dones_list: list[bool] = []
        for env_i in range(2):
            spec = self._queues[env_i][self._ep_idx[env_i]]
            level_complete, done = spec[self._step_idx[env_i]]
            infos.append({"level_complete": level_complete})
            dones_list.append(done)

            self._step_idx[env_i] += 1
            if done:
                self._ep_idx[env_i] += 1
                self._step_idx[env_i] = 0

        obs = np.zeros((2, 4), dtype=np.float32)
        rewards = np.array([1.0, 1.0])
        dones = np.array(dones_list)
        return obs, rewards, dones, infos


def test_vecenv_info_as_list_of_dicts_is_handled() -> None:
    """SB3 VecEnv returns infos as a list of per-sub-env dicts; must not choke."""
    # Extra trailing episodes pad each queue so a sub-env that finishes its
    # "real" episodes first still has scripted steps to return while the
    # other sub-env catches up (both sub-envs step in lockstep each call).
    queues = [
        [
            _episode(2, complete_on_step=0),
            _episode(2, complete_on_step=None),
            _episode(5, complete_on_step=None),
        ],
        [
            _episode(3, complete_on_step=None),
            _episode(2, complete_on_step=1),
            _episode(5, complete_on_step=None),
        ],
    ]
    env = _VectorizedFakeVecEnv(queues)
    model = _FakeModel()

    result = evaluate_completion_rate(model, env, n_episodes=4, deterministic=True)

    assert result["n_episodes"] == 4
    # env0: [completed, not completed], env1: [not completed, completed] => 2/4
    assert result["n_completed"] == 2
    assert result["completion_rate"] == pytest.approx(0.5)
