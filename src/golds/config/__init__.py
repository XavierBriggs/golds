"""Configuration module for GOLDS."""

from golds.config.loader import ConfigLoader
from golds.config.schema import EnvironmentConfig, ExperimentConfig, PPOConfig, TrainingConfig

__all__ = [
    "ExperimentConfig",
    "PPOConfig",
    "EnvironmentConfig",
    "TrainingConfig",
    "ConfigLoader",
]
