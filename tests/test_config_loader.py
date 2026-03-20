"""Tests for configuration loading and merging."""

import pytest

from golds.config.loader import ConfigLoader, deep_merge
from golds.config.schema import ExperimentConfig, PPOConfig

# ---- deep_merge tests ----


def test_deep_merge_simple():
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 99, "z": 100}}
    result = deep_merge(base, override)
    assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}


def test_deep_merge_override():
    """Override replaces non-dict values even if base has a dict."""
    base = {"a": {"x": 1}}
    override = {"a": "replaced"}
    result = deep_merge(base, override)
    assert result == {"a": "replaced"}


# ---- ConfigLoader tests ----


def test_load_defaults(project_root):
    loader = ConfigLoader(config_dir=project_root / "configs")
    # The loader should have loaded defaults; verify by checking the internal dict
    assert "ppo" in loader._defaults
    assert "environment" in loader._defaults
    assert "training" in loader._defaults


def test_load_game_config(project_root):
    loader = ConfigLoader(config_dir=project_root / "configs")
    config = loader.load_game("breakout")
    assert isinstance(config, ExperimentConfig)
    assert config.environment.game_id == "breakout"
    assert config.environment.platform == "atari"


# ---- config_hash tests ----


def test_config_hash_deterministic(sample_config):
    config = ExperimentConfig(**sample_config)
    h1 = config.config_hash()
    h2 = config.config_hash()
    assert h1 == h2


def test_config_hash_different(sample_config):
    config1 = ExperimentConfig(**sample_config)

    modified = sample_config.copy()
    modified["ppo"] = {**sample_config.get("ppo", {}), "learning_rate": 1e-3}
    config2 = ExperimentConfig(**modified)

    assert config1.config_hash() != config2.config_hash()


# ---- New field defaults ----


def test_new_fields_have_defaults(sample_config):
    config = ExperimentConfig(**sample_config)
    assert config.round == 1
    assert config.version == ""
    assert config.parent_run is None
    assert config.environment.reward_regime == "clipped"


def test_schedule_fields_have_defaults():
    ppo = PPOConfig()
    assert ppo.lr_schedule == "constant"
    assert ppo.clip_schedule == "constant"


# ---- to_ppo_kwargs tests ----


def test_to_ppo_kwargs_constant(sample_config):
    config = ExperimentConfig(**sample_config)
    kwargs = config.to_ppo_kwargs()
    # With constant schedule, lr and clip should be plain floats
    assert isinstance(kwargs["learning_rate"], float)
    assert isinstance(kwargs["clip_range"], float)
    assert kwargs["learning_rate"] == config.ppo.learning_rate
    assert kwargs["clip_range"] == config.ppo.clip_range


def test_to_ppo_kwargs_linear(sample_config):
    sample_config["ppo"] = {
        **sample_config.get("ppo", {}),
        "lr_schedule": "linear",
        "clip_schedule": "linear",
    }
    config = ExperimentConfig(**sample_config)
    kwargs = config.to_ppo_kwargs()
    # With linear schedule, lr and clip should be callables
    assert callable(kwargs["learning_rate"])
    assert callable(kwargs["clip_range"])
    # At progress_remaining=1.0, should return initial value
    assert kwargs["learning_rate"](1.0) == pytest.approx(config.ppo.learning_rate)
    assert kwargs["clip_range"](1.0) == pytest.approx(config.ppo.clip_range)
