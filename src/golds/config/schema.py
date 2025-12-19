"""Pydantic schemas for configuration validation."""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PPOConfig(BaseModel):
    """PPO hyperparameters following DeepMind defaults."""

    learning_rate: float = Field(default=2.5e-4, ge=0, description="Learning rate")
    n_steps: int = Field(default=128, ge=1, description="Number of steps per rollout")
    batch_size: int = Field(default=256, ge=1, description="Minibatch size")
    n_epochs: int = Field(default=4, ge=1, description="Number of epochs per update")
    gamma: float = Field(default=0.99, ge=0, le=1, description="Discount factor")
    gae_lambda: float = Field(default=0.95, ge=0, le=1, description="GAE lambda")
    clip_range: float = Field(default=0.1, ge=0, description="PPO clip parameter")
    clip_range_vf: Optional[float] = Field(
        default=None, description="Value function clip range (None = no clipping)"
    )
    ent_coef: float = Field(default=0.01, ge=0, description="Entropy coefficient")
    vf_coef: float = Field(default=0.5, ge=0, description="Value function coefficient")
    max_grad_norm: float = Field(default=0.5, ge=0, description="Max gradient norm")


class EnvironmentConfig(BaseModel):
    """Environment configuration."""

    platform: Literal["atari", "retro"] = Field(
        description="Platform: 'atari' for Atari 2600, 'retro' for NES/SNES"
    )
    game_id: str = Field(description="Game identifier (e.g., 'space_invaders')")
    n_envs: int = Field(default=8, ge=1, description="Number of parallel environments")
    frame_stack: int = Field(default=4, ge=1, description="Number of frames to stack")
    frame_skip: int = Field(default=4, ge=1, description="Number of frames to skip")
    screen_size: int = Field(default=84, ge=1, description="Screen size after resize")
    grayscale: bool = Field(default=True, description="Convert to grayscale")
    clip_reward: bool = Field(default=True, description="Clip rewards to {-1, 0, 1}")
    terminal_on_life_loss: bool = Field(
        default=True, description="End episode on life loss (Atari only)"
    )
    use_subproc: bool = Field(
        default=True, description="Use SubprocVecEnv for parallel environments"
    )
    state: Optional[str] = Field(
        default=None, description="Initial state for retro games"
    )
    players: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Number of players (retro only). Use 2 for PvP/self-play states.",
    )
    opponent: Literal["none", "random", "noop", "model", "self_play"] = Field(
        default="none",
        description="Opponent policy mode when players=2 (retro only).",
    )
    opponent_model_path: Optional[str] = Field(
        default=None,
        description="Path to opponent model .zip (used when opponent='model').",
    )


class TrainingConfig(BaseModel):
    """Training configuration."""

    total_timesteps: int = Field(
        default=10_000_000, ge=1, description="Total training timesteps"
    )
    eval_freq: int = Field(
        default=50_000,
        ge=0,
        description="Evaluation frequency (timesteps); 0 disables evaluation",
    )
    eval_episodes: int = Field(default=10, ge=1, description="Episodes per evaluation")
    eval_deterministic: bool = Field(
        default=True,
        description="Whether evaluation uses deterministic actions (if False, uses stochastic actions).",
    )
    save_freq: int = Field(
        default=100_000,
        ge=0,
        description="Checkpoint frequency (timesteps); 0 disables checkpointing",
    )
    log_interval: int = Field(default=1, ge=1, description="Logging interval (updates)")
    seed: Optional[int] = Field(default=None, description="Random seed")
    device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto", description="Device to use"
    )
    self_play_snapshot_freq: int = Field(
        default=0,
        ge=0,
        description="If >0 and opponent='self_play', save opponent snapshots every N timesteps.",
    )
    self_play_max_snapshots: int = Field(
        default=5,
        ge=1,
        description="Max number of opponent snapshots to keep (self-play only).",
    )


class ExperimentConfig(BaseModel):
    """Full experiment configuration."""

    name: str = Field(description="Experiment name (alphanumeric, underscores, dashes)")
    description: str = Field(default="", description="Experiment description")
    environment: EnvironmentConfig = Field(description="Environment configuration")
    ppo: PPOConfig = Field(default_factory=PPOConfig, description="PPO hyperparameters")
    training: TrainingConfig = Field(
        default_factory=TrainingConfig, description="Training configuration"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure experiment name is filesystem-safe."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Name must be alphanumeric with underscores/dashes only")
        return v

    def to_ppo_kwargs(self) -> dict:
        """Convert PPO config to kwargs for SB3 PPO."""
        return {
            "learning_rate": self.ppo.learning_rate,
            "n_steps": self.ppo.n_steps,
            "batch_size": self.ppo.batch_size,
            "n_epochs": self.ppo.n_epochs,
            "gamma": self.ppo.gamma,
            "gae_lambda": self.ppo.gae_lambda,
            "clip_range": self.ppo.clip_range,
            "clip_range_vf": self.ppo.clip_range_vf,
            "ent_coef": self.ppo.ent_coef,
            "vf_coef": self.ppo.vf_coef,
            "max_grad_norm": self.ppo.max_grad_norm,
        }
