"""Tests for PPOInvariantCallback: live PPO training-health invariant
checks (R10, G5).

All tests are ROM-free: they use synthetic metric values and a fake
(MagicMock) SB3 model, never a real environment or training run.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from golds.config.schema import EnvironmentConfig, ExperimentConfig
from golds.training.invariant_callback import PPOInvariantCallback, PPOInvariantError
from golds.training.trainer import Trainer


def _make_model(name_to_value=None, advantages=None):
    """Build a MagicMock standing in for the SB3 model the callback reads from."""
    model = MagicMock()
    model.logger.name_to_value = name_to_value or {}
    if advantages is not None:
        model.rollout_buffer.advantages = advantages
    else:
        model.rollout_buffer.advantages = np.random.default_rng(0).normal(size=256)
    return model


def _make_callback(**overrides):
    return PPOInvariantCallback(**overrides)


class TestClipFraction:
    """Invariant 1: clip_fraction must stay within the open interval (0, 0.3)."""

    def test_clip_fraction_just_inside_lower_bound_passes(self):
        cb = _make_callback()
        cb.model = _make_model(name_to_value={"train/clip_fraction": 0.001})
        cb.num_timesteps = 1024
        cb._on_rollout_end()
        assert cb.get_violations() == []

    def test_clip_fraction_at_zero_flags(self):
        """Boundary: exactly 0 violates the open lower bound."""
        cb = _make_callback()
        cb.model = _make_model(name_to_value={"train/clip_fraction": 0.0})
        cb.num_timesteps = 1024
        cb._on_rollout_end()
        violations = cb.get_violations()
        assert len(violations) == 1
        assert violations[0]["name"] == "clip_fraction"

    def test_clip_fraction_just_inside_upper_bound_passes(self):
        cb = _make_callback()
        cb.model = _make_model(name_to_value={"train/clip_fraction": 0.299})
        cb.num_timesteps = 1024
        cb._on_rollout_end()
        assert cb.get_violations() == []

    def test_clip_fraction_at_upper_bound_flags(self):
        """Boundary: exactly 0.3 violates the open upper bound."""
        cb = _make_callback()
        cb.model = _make_model(name_to_value={"train/clip_fraction": 0.3})
        cb.num_timesteps = 1024
        cb._on_rollout_end()
        violations = cb.get_violations()
        assert len(violations) == 1
        assert violations[0]["name"] == "clip_fraction"

    def test_clip_fraction_above_upper_bound_flags(self):
        cb = _make_callback()
        cb.model = _make_model(name_to_value={"train/clip_fraction": 0.5})
        cb.num_timesteps = 1024
        cb._on_rollout_end()
        violations = cb.get_violations()
        assert len(violations) == 1
        assert violations[0]["name"] == "clip_fraction"


class TestApproxKl:
    """Invariant 2: approx_kl must stay below a configurable ceiling (default 0.05)."""

    def test_approx_kl_just_below_ceiling_passes(self):
        cb = _make_callback()
        cb.model = _make_model(name_to_value={"train/approx_kl": 0.0499})
        cb.num_timesteps = 2048
        cb._on_rollout_end()
        assert cb.get_violations() == []

    def test_approx_kl_at_ceiling_flags(self):
        """Boundary: at the ceiling counts as a violation (bound is exclusive)."""
        cb = _make_callback()
        cb.model = _make_model(name_to_value={"train/approx_kl": 0.05})
        cb.num_timesteps = 2048
        cb._on_rollout_end()
        violations = cb.get_violations()
        assert len(violations) == 1
        assert violations[0]["name"] == "approx_kl"

    def test_approx_kl_above_ceiling_flags(self):
        cb = _make_callback()
        cb.model = _make_model(name_to_value={"train/approx_kl": 0.2})
        cb.num_timesteps = 2048
        cb._on_rollout_end()
        violations = cb.get_violations()
        assert len(violations) == 1
        assert violations[0]["name"] == "approx_kl"

    def test_approx_kl_ceiling_is_configurable(self):
        cb = _make_callback(approx_kl_max=0.5)
        cb.model = _make_model(name_to_value={"train/approx_kl": 0.2})
        cb.num_timesteps = 2048
        cb._on_rollout_end()
        assert cb.get_violations() == []


class TestExplainedVarianceTrend:
    """Invariant 3: explained_variance should trend upward, not collapse."""

    def test_rising_sequence_passes(self):
        cb = _make_callback()
        for i, ev in enumerate([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8]):
            cb.model = _make_model(name_to_value={"train/explained_variance": ev})
            cb.num_timesteps = (i + 1) * 1000
            cb._on_rollout_end()
        assert cb.get_violations() == []

    def test_collapsing_sequence_flags(self):
        """A healthy rising run that suddenly collapses must be flagged."""
        cb = _make_callback()
        for i, ev in enumerate([0.1, 0.3, 0.5, 0.7, -0.6]):
            cb.model = _make_model(name_to_value={"train/explained_variance": ev})
            cb.num_timesteps = (i + 1) * 1000
            cb._on_rollout_end()
        violations = cb.get_violations()
        assert any(v["name"] == "explained_variance_trend" for v in violations)

    def test_healthy_dip_above_floor_does_not_flag(self):
        """A sharp EV dip that stays in the healthy band is normal volatility.

        Regression for the false positive seen on the 2026-07-18 Breakout
        baseline: EV fell 0.77 -> 0.35 (a >0.3 drop) but 0.35 is healthy.
        The floor gate must suppress this.
        """
        cb = _make_callback()
        for i, ev in enumerate([0.75, 0.80, 0.77, 0.78, 0.35]):
            cb.model = _make_model(name_to_value={"train/explained_variance": ev})
            cb.num_timesteps = (i + 1) * 1000
            cb._on_rollout_end()
        assert cb.get_violations() == []

    def test_stuck_negative_late_in_training_flags(self):
        """explained_variance stuck negative well past the grace period is flagged."""
        cb = _make_callback(explained_variance_grace_updates=3)
        for i, ev in enumerate([-0.1] * 6):
            cb.model = _make_model(name_to_value={"train/explained_variance": ev})
            cb.num_timesteps = (i + 1) * 1000
            cb._on_rollout_end()
        violations = cb.get_violations()
        assert any(v["name"] == "explained_variance_stuck_negative" for v in violations)

    def test_stuck_negative_within_grace_period_passes(self):
        """Negative EV early in training (within the grace period) is expected, not flagged."""
        cb = _make_callback(explained_variance_grace_updates=10)
        for i, ev in enumerate([-0.1] * 3):
            cb.model = _make_model(name_to_value={"train/explained_variance": ev})
            cb.num_timesteps = (i + 1) * 1000
            cb._on_rollout_end()
        assert cb.get_violations() == []


class TestAdvantages:
    """Invariant 4: rollout-buffer advantages must be finite and non-degenerate.

    NOTE: SB3 normalizes advantages per-minibatch inside `train()`, which
    runs *after* `_on_rollout_end`. So this checks the raw, pre-normalization
    buffer as an approximation -- it cannot see the true post-normalization
    mean/std. We only assert finiteness and non-degeneracy here, not an
    exact zero-mean/unit-std claim.
    """

    def test_healthy_near_zero_mean_unit_std_advantages_pass(self):
        cb = _make_callback()
        rng = np.random.default_rng(42)
        advantages = rng.normal(loc=0.0, scale=1.0, size=2048)
        cb.model = _make_model(advantages=advantages)
        cb.num_timesteps = 4096
        cb._on_rollout_end()
        assert cb.get_violations() == []

    def test_degenerate_all_same_advantages_flag(self):
        cb = _make_callback()
        advantages = np.full(256, 3.0)
        cb.model = _make_model(advantages=advantages)
        cb.num_timesteps = 4096
        cb._on_rollout_end()
        violations = cb.get_violations()
        assert any(v["name"] == "advantages_degenerate" for v in violations)

    def test_nan_advantages_flag(self):
        cb = _make_callback()
        advantages = np.full(256, np.nan)
        cb.model = _make_model(advantages=advantages)
        cb.num_timesteps = 4096
        cb._on_rollout_end()
        violations = cb.get_violations()
        assert any(v["name"] == "advantages_non_finite" for v in violations)

    def test_inf_advantages_flag(self):
        cb = _make_callback()
        rng = np.random.default_rng(1)
        advantages = rng.normal(size=256)
        advantages[5] = np.inf
        cb.model = _make_model(advantages=advantages)
        cb.num_timesteps = 4096
        cb._on_rollout_end()
        violations = cb.get_violations()
        assert any(v["name"] == "advantages_non_finite" for v in violations)


class TestStrictMode:
    """strict=True raises on violation; strict=False (default) only records."""

    def test_default_is_not_strict_and_does_not_raise(self):
        cb = _make_callback()
        assert cb.strict is False
        cb.model = _make_model(name_to_value={"train/clip_fraction": 0.9})
        cb.num_timesteps = 1024
        cb._on_rollout_end()  # must not raise
        assert len(cb.get_violations()) == 1

    def test_strict_true_raises_on_violation(self):
        cb = _make_callback(strict=True)
        cb.model = _make_model(name_to_value={"train/clip_fraction": 0.9})
        cb.num_timesteps = 1024
        with pytest.raises(PPOInvariantError):
            cb._on_rollout_end()

    def test_strict_true_does_not_raise_when_healthy(self):
        cb = _make_callback(strict=True)
        cb.model = _make_model(
            name_to_value={
                "train/clip_fraction": 0.1,
                "train/approx_kl": 0.01,
                "train/explained_variance": 0.5,
            }
        )
        cb.num_timesteps = 1024
        cb._on_rollout_end()  # must not raise
        assert cb.get_violations() == []


class TestGetViolations:
    def test_get_violations_returns_empty_list_when_healthy(self):
        cb = _make_callback()
        cb.model = _make_model(
            name_to_value={
                "train/clip_fraction": 0.1,
                "train/approx_kl": 0.01,
                "train/explained_variance": 0.5,
            }
        )
        cb.num_timesteps = 1024
        cb._on_rollout_end()
        assert cb.get_violations() == []

    def test_get_violations_accumulates_across_rollouts(self):
        cb = _make_callback()
        for step in (1000, 2000):
            cb.model = _make_model(name_to_value={"train/clip_fraction": 0.9})
            cb.num_timesteps = step
            cb._on_rollout_end()
        assert len(cb.get_violations()) == 2

    def test_on_step_returns_true(self):
        cb = _make_callback()
        assert cb._on_step() is True


class TestConfigDefault:
    """invariant_checks is opt-in and off by default so existing runs/tests are unaffected."""

    def test_invariant_checks_disabled_by_default(self):
        config = ExperimentConfig(
            name="invariant_config_test",
            environment=EnvironmentConfig(platform="atari", game_id="breakout"),
        )
        assert config.invariant_checks.enabled is False


class TestTrainerWiring:
    """The trainer only adds PPOInvariantCallback when invariant_checks.enabled is True."""

    def _make_config(self, tmp_path, invariant_checks_overrides=None):
        overrides = dict(invariant_checks_overrides or {})
        config = ExperimentConfig(
            name="invariant_wiring_test",
            environment=EnvironmentConfig(platform="atari", game_id="breakout", n_envs=1),
        )
        config.training.eval_freq = 0  # skip real eval env requirement
        for key, value in overrides.items():
            setattr(config.invariant_checks, key, value)
        return config

    def test_callback_absent_by_default(self, tmp_path):
        config = self._make_config(tmp_path)
        trainer = Trainer(config=config, output_dir=tmp_path / "run")
        callback_list = trainer._create_callbacks(eval_env=MagicMock())
        assert not any(isinstance(cb, PPOInvariantCallback) for cb in callback_list.callbacks)

    def test_callback_present_when_enabled(self, tmp_path):
        config = self._make_config(tmp_path, {"enabled": True})
        trainer = Trainer(config=config, output_dir=tmp_path / "run")
        callback_list = trainer._create_callbacks(eval_env=MagicMock())
        matches = [cb for cb in callback_list.callbacks if isinstance(cb, PPOInvariantCallback)]
        assert len(matches) == 1
