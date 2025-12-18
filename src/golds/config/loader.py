"""Configuration loading and merging utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from golds.config.schema import ExperimentConfig

# Default configuration values
DEFAULT_CONFIG: dict[str, Any] = {
    "ppo": {
        "learning_rate": 2.5e-4,
        "n_steps": 128,
        "batch_size": 256,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "clip_range_vf": None,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "environment": {
        "n_envs": 8,
        "frame_stack": 4,
        "frame_skip": 4,
        "screen_size": 84,
        "grayscale": True,
        "clip_reward": True,
        "terminal_on_life_loss": True,
        "use_subproc": True,
    },
    "training": {
        "total_timesteps": 10_000_000,
        "eval_freq": 50_000,
        "eval_episodes": 10,
        "save_freq": 100_000,
        "log_interval": 1,
        "seed": None,
        "device": "auto",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base dictionary.

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ConfigLoader:
    """Load and validate configuration files."""

    def __init__(self, config_dir: Path | None = None) -> None:
        """Initialize config loader.

        Args:
            config_dir: Directory containing config files. Defaults to 'configs/'.
        """
        self.config_dir = config_dir or Path("configs")
        self._defaults = self._load_defaults()

    def _load_defaults(self) -> dict[str, Any]:
        """Load default configuration from file or use built-in defaults."""
        defaults_path = self.config_dir / "defaults.yaml"
        if defaults_path.exists():
            with open(defaults_path) as f:
                loaded = yaml.safe_load(f) or {}
            return deep_merge(DEFAULT_CONFIG, loaded)
        return DEFAULT_CONFIG.copy()

    def load(self, config_path: Path | str) -> ExperimentConfig:
        """Load and validate experiment configuration.

        Args:
            config_path: Path to YAML config file

        Returns:
            Validated ExperimentConfig
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}

        # Merge with defaults
        merged = deep_merge(self._defaults, user_config)

        # Validate with Pydantic
        return ExperimentConfig(**merged)

    def load_game(self, game_id: str) -> ExperimentConfig:
        """Load configuration for a specific game.

        Args:
            game_id: Game identifier (e.g., 'space_invaders')

        Returns:
            Validated ExperimentConfig
        """
        game_config_path = self.config_dir / "games" / f"{game_id}.yaml"
        if not game_config_path.exists():
            raise FileNotFoundError(
                f"No config found for game: {game_id}. "
                f"Expected at: {game_config_path}"
            )
        return self.load(game_config_path)

    def create_from_args(
        self,
        game_id: str,
        platform: str,
        *,
        n_envs: int = 8,
        total_timesteps: int = 10_000_000,
        seed: int | None = None,
        device: str = "auto",
    ) -> ExperimentConfig:
        """Create configuration from command-line arguments.

        Args:
            game_id: Game identifier
            platform: Platform ('atari' or 'retro')
            n_envs: Number of parallel environments
            total_timesteps: Total training timesteps
            seed: Random seed
            device: Device to use

        Returns:
            ExperimentConfig
        """
        config = deep_merge(
            self._defaults,
            {
                "name": f"{game_id}_run",
                "environment": {
                    "platform": platform,
                    "game_id": game_id,
                    "n_envs": n_envs,
                },
                "training": {
                    "total_timesteps": total_timesteps,
                    "seed": seed,
                    "device": device,
                },
            },
        )
        return ExperimentConfig(**config)
