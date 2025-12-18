"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def configs_dir(project_root: Path) -> Path:
    """Return the configs directory."""
    return project_root / "configs"


@pytest.fixture
def sample_config() -> dict:
    """Return a sample configuration dictionary."""
    return {
        "name": "test_experiment",
        "description": "Test experiment",
        "environment": {
            "platform": "atari",
            "game_id": "space_invaders",
            "n_envs": 2,
            "frame_stack": 4,
        },
        "ppo": {
            "learning_rate": 2.5e-4,
            "n_steps": 128,
        },
        "training": {
            "total_timesteps": 1000,
            "eval_freq": 500,
            "device": "cpu",
        },
    }
