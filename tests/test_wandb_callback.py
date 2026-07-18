"""Tests for WandbCallback: config/git provenance logging, PPO metric
forwarding, and graceful degradation (M1b/T5, R8).

wandb is mocked throughout; these tests must never touch the network.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from golds.config.schema import EnvironmentConfig, ExperimentConfig
from golds.training.wandb_callback import WandbCallback


def _make_config(**overrides) -> ExperimentConfig:
    defaults = dict(
        name="wandb_test_experiment",
        environment=EnvironmentConfig(platform="atari", game_id="breakout"),
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _make_model(name_to_value=None, ep_rewards=None):
    """Build a MagicMock standing in for the SB3 model the callback reads from."""
    model = MagicMock()
    model.logger.name_to_value = name_to_value or {}
    model.ep_info_buffer = (
        [{"r": r, "l": 100} for r in ep_rewards] if ep_rewards is not None else []
    )
    return model


class TestWandbInit:
    """T-init: wandb.init is called with the full config, git provenance, and run name."""

    def test_wandb_init_called_with_git_sha_and_experiment_name(self):
        expected_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()

        config = _make_config(name="wandb_test_experiment")
        cb = WandbCallback(experiment_config=config, project="golds-test", verbose=0)

        with patch("golds.training.wandb_callback.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            cb._on_training_start()

        assert mock_wandb.init.called
        _, kwargs = mock_wandb.init.call_args
        assert kwargs["name"] == "wandb_test_experiment"
        assert kwargs["project"] == "golds-test"
        assert kwargs["config"]["git_sha"] == expected_sha
        assert "git_dirty" in kwargs["config"]
        assert kwargs["config"]["name"] == "wandb_test_experiment"
        assert cb.enabled is True


class TestWandbMetricForwarding:
    """T-metrics: PPO health metrics from the SB3 logger are forwarded to wandb.log."""

    def test_metrics_forwarded_with_expected_keys(self):
        config = _make_config()
        cb = WandbCallback(experiment_config=config, verbose=0)
        cb.enabled = True
        cb.num_timesteps = 4096
        cb.model = _make_model(
            name_to_value={
                "train/entropy_loss": -0.5,
                "train/policy_gradient_loss": 0.01,
                "train/value_loss": 2.3,
                "train/approx_kl": 0.012,
                "train/clip_fraction": 0.08,
                "train/explained_variance": 0.75,
                "time/fps": 850,
            },
            ep_rewards=[10.0, 20.0],
        )

        with patch("golds.training.wandb_callback.wandb") as mock_wandb:
            cb._on_rollout_end()

        assert mock_wandb.log.called
        metrics, kwargs = mock_wandb.log.call_args
        logged = metrics[0]
        assert kwargs["step"] == 4096
        for key in (
            "train/entropy_loss",
            "train/policy_gradient_loss",
            "train/value_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/explained_variance",
            "time/fps",
            "rollout/ep_rew_mean",
        ):
            assert key in logged
        assert logged["rollout/ep_rew_mean"] == 15.0

    def test_metrics_not_logged_when_disabled(self):
        """No wandb.log call at all when the callback is disabled (init never succeeded)."""
        config = _make_config()
        cb = WandbCallback(experiment_config=config, verbose=0)
        cb.enabled = False
        cb.num_timesteps = 1000
        cb.model = _make_model(name_to_value={"train/entropy_loss": -0.5})

        with patch("golds.training.wandb_callback.wandb") as mock_wandb:
            cb._on_rollout_end()

        assert not mock_wandb.log.called


class TestWandbGracefulDegradation:
    """T-degrade: wandb failures never propagate and never crash a training run.

    This is the load-bearing test: a multi-day run must not die because
    W&B is unreachable, unauthenticated, or misbehaving.
    """

    def test_init_raising_does_not_propagate_and_disables_logging(self):
        config = _make_config()
        cb = WandbCallback(experiment_config=config, verbose=0)

        with patch("golds.training.wandb_callback.wandb") as mock_wandb:
            mock_wandb.init.side_effect = RuntimeError("network unreachable")
            # Must not raise.
            cb._on_training_start()

        assert cb.enabled is False

        # Downstream callback methods must also return normally afterward.
        cb.num_timesteps = 2048
        cb.model = _make_model(name_to_value={"train/entropy_loss": -0.1})
        with patch("golds.training.wandb_callback.wandb") as mock_wandb:
            cb._on_rollout_end()  # must not raise
            assert not mock_wandb.log.called

        assert cb._on_step() is True

        # Training end must not raise either, and must not call wandb.finish
        # since the run was never actually started.
        with patch("golds.training.wandb_callback.wandb") as mock_wandb:
            cb._on_training_end()
            assert not mock_wandb.finish.called

    def test_log_raising_mid_training_does_not_propagate(self):
        """A transient failure during a rollout log must not crash training."""
        config = _make_config()
        cb = WandbCallback(experiment_config=config, verbose=0)
        cb.enabled = True
        cb.num_timesteps = 8192
        cb.model = _make_model(name_to_value={"train/value_loss": 1.0})

        with patch("golds.training.wandb_callback.wandb") as mock_wandb:
            mock_wandb.log.side_effect = RuntimeError("wandb service down")
            cb._on_rollout_end()  # must not raise

        assert cb._on_step() is True

    def test_finish_raising_does_not_propagate(self):
        config = _make_config()
        cb = WandbCallback(experiment_config=config, verbose=0)
        cb.enabled = True

        with patch("golds.training.wandb_callback.wandb") as mock_wandb:
            mock_wandb.finish.side_effect = RuntimeError("boom")
            cb._on_training_end()  # must not raise

        assert cb.enabled is False


class TestWandbConfigDefault:
    """T-config: wandb is opt-in and off by default so existing runs are unaffected."""

    def test_wandb_disabled_by_default(self):
        config = _make_config()
        assert config.wandb.enabled is False
