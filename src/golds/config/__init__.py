"""Configuration module for GOLDS."""

from golds.config.schema import ExperimentConfig, PPOConfig, EnvironmentConfig, TrainingConfig
from golds.config.loader import ConfigLoader

__all__ = [
    "ExperimentConfig",
    "PPOConfig",
    "EnvironmentConfig",
    "TrainingConfig",
    "ConfigLoader",
]
