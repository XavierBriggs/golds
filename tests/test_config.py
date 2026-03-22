"""Tests for configuration system."""

import pytest

from golds.config.loader import ConfigLoader, deep_merge
from golds.config.schema import EnvironmentConfig, ExperimentConfig, PPOConfig


class TestPPOConfig:
    """Tests for PPOConfig."""

    def test_default_values(self):
        """Test PPOConfig default values."""
        config = PPOConfig()
        assert config.learning_rate == 2.5e-4
        assert config.n_steps == 128
        assert config.gamma == 0.99
        assert config.clip_range == 0.1

    def test_custom_values(self):
        """Test PPOConfig with custom values."""
        config = PPOConfig(learning_rate=1e-3, n_steps=256)
        assert config.learning_rate == 1e-3
        assert config.n_steps == 256

    def test_validation(self):
        """Test PPOConfig validation."""
        with pytest.raises(ValueError):
            PPOConfig(gamma=1.5)  # Must be <= 1


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig."""

    def test_atari_config(self):
        """Test Atari environment config."""
        config = EnvironmentConfig(
            platform="atari",
            game_id="space_invaders",
        )
        assert config.platform == "atari"
        assert config.n_envs == 8

    def test_retro_config(self):
        """Test Retro environment config."""
        config = EnvironmentConfig(
            platform="retro",
            game_id="super_mario_bros",
            state="Level1-1",
        )
        assert config.platform == "retro"
        assert config.state == "Level1-1"


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_full_config(self, sample_config):
        """Test full experiment config."""
        config = ExperimentConfig(**sample_config)
        assert config.name == "test_experiment"
        assert config.environment.game_id == "space_invaders"

    def test_name_validation(self, sample_config):
        """Test experiment name validation."""
        sample_config["name"] = "invalid name!"
        with pytest.raises(ValueError):
            ExperimentConfig(**sample_config)

    def test_to_ppo_kwargs(self, sample_config):
        """Test PPO kwargs conversion."""
        config = ExperimentConfig(**sample_config)
        kwargs = config.to_ppo_kwargs()
        assert "learning_rate" in kwargs
        assert "gamma" in kwargs


class TestDeepMerge:
    """Tests for deep_merge function."""

    def test_simple_merge(self):
        """Test simple dictionary merge."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Test nested dictionary merge."""
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}


class TestConfigLoader:
    """Tests for ConfigLoader."""

    def test_load_defaults(self, configs_dir):
        """Test loading default config."""
        loader = ConfigLoader(configs_dir)
        assert loader._defaults is not None
        assert "ppo" in loader._defaults

    def test_create_from_args(self):
        """Test creating config from arguments."""
        loader = ConfigLoader()
        config = loader.create_from_args(
            game_id="space_invaders",
            platform="atari",
            n_envs=4,
            total_timesteps=10000,
        )
        assert config.environment.game_id == "space_invaders"
        assert config.environment.n_envs == 4


class TestV3ConfigFields:
    """Tests for v3 training upgrade config fields."""

    def test_environment_config_v3_fields(self):
        from golds.config.schema import EnvironmentConfig

        cfg = EnvironmentConfig(
            platform="retro",
            game_id="sonic_the_hedgehog",
            action_set="platformer",
            sticky_action_prob=0.25,
            levels=["GreenHillZone.Act1", "GreenHillZone.Act2"],
            death_penalty=-1.0,
            collectible_reward_scale=0.01,
            time_penalty=-0.001,
        )
        assert cfg.action_set == "platformer"
        assert cfg.sticky_action_prob == 0.25
        assert len(cfg.levels) == 2
        assert cfg.death_penalty == -1.0
        assert cfg.collectible_reward_scale == 0.01
        assert cfg.time_penalty == -0.001

    def test_environment_config_v3_defaults(self):
        from golds.config.schema import EnvironmentConfig

        cfg = EnvironmentConfig(platform="atari", game_id="pong")
        assert cfg.action_set == "full"
        assert cfg.sticky_action_prob == 0.0
        assert cfg.levels == []
        assert cfg.death_penalty == 0.0
        assert cfg.collectible_reward_scale == 0.0
        assert cfg.time_penalty == 0.0

    def test_training_config_rnd_fields(self):
        from golds.config.schema import TrainingConfig

        cfg = TrainingConfig(rnd_enabled=True, rnd_reward_scale=0.05, rnd_learning_rate=5e-5)
        assert cfg.rnd_enabled is True
        assert cfg.rnd_reward_scale == 0.05
        assert cfg.rnd_learning_rate == 5e-5

    def test_training_config_rnd_defaults(self):
        from golds.config.schema import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.rnd_enabled is False
        assert cfg.rnd_reward_scale == 0.01
        assert cfg.rnd_learning_rate == 1e-4
