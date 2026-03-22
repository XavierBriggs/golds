"""Pydantic schemas for configuration validation."""

from __future__ import annotations

import re
from typing import Literal

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
    clip_range_vf: float | None = Field(
        default=None, description="Value function clip range (None = no clipping)"
    )
    ent_coef: float = Field(default=0.01, ge=0, description="Entropy coefficient")
    vf_coef: float = Field(default=0.5, ge=0, description="Value function coefficient")
    max_grad_norm: float = Field(default=0.5, ge=0, description="Max gradient norm")
    policy: Literal["CnnPolicy", "CnnLstmPolicy"] = Field(
        default="CnnPolicy", description="Policy architecture"
    )
    lr_schedule: Literal["constant", "linear", "cosine"] = Field(
        default="constant", description="Learning rate schedule"
    )
    clip_schedule: Literal["constant", "linear"] = Field(
        default="constant", description="Clip range schedule"
    )


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
    state: str | None = Field(default=None, description="Initial state for retro games")
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
    opponent_model_path: str | None = Field(
        default=None,
        description="Path to opponent model .zip (used when opponent='model').",
    )
    reward_regime: Literal["clipped", "raw", "normalized"] = Field(
        default="clipped",
        description="Reward handling: clipped={-1,0,+1}, raw=game score, normalized=VecNormalize",
    )
    x_pos_reward_scale: float = Field(
        default=0.0,
        ge=0,
        description="Scale for x-position reward shaping (0 = disabled). Platformers only.",
    )
    max_episode_steps: int = Field(
        default=0,
        ge=0,
        description="Max steps per episode before truncation (0 = disabled). Useful for fighting games.",
    )
    action_set: Literal["full", "platformer", "fighter", "puzzle"] = Field(
        default="full",
        description="Action space reduction set. 'full' uses all filtered actions.",
    )
    sticky_action_prob: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Probability of repeating previous action (0 = disabled, 0.25 = standard).",
    )
    levels: list[str] = Field(
        default_factory=list,
        description="Level rotation list for multi-level training. Empty = single default level.",
    )
    death_penalty: float = Field(
        default=0.0,
        le=0,
        description="Penalty applied on episode termination (should be negative or 0).",
    )
    collectible_reward_scale: float = Field(
        default=0.0,
        ge=0,
        description="Scale for collectible bonuses (rings/coins). 0 = disabled.",
    )
    time_penalty: float = Field(
        default=0.0,
        le=0,
        description="Per-step penalty to encourage speed (should be negative or 0).",
    )


class TrainingConfig(BaseModel):
    """Training configuration."""

    total_timesteps: int = Field(default=10_000_000, ge=1, description="Total training timesteps")
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
    seed: int | None = Field(default=None, description="Random seed")
    device: Literal["auto", "cuda", "mps", "cpu"] = Field(
        default="auto", description="Device to use (auto detects best: cuda > mps > cpu)"
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
    self_play_sampling: Literal["uniform", "proportional", "pfsp"] = Field(
        default="uniform",
        description="Opponent sampling strategy for self-play (uniform, proportional, or pfsp).",
    )
    rnd_enabled: bool = Field(
        default=False,
        description="Enable RND (Random Network Distillation) intrinsic reward.",
    )
    rnd_reward_scale: float = Field(
        default=0.01,
        ge=0,
        description="Scale factor for RND intrinsic reward.",
    )
    rnd_learning_rate: float = Field(
        default=1e-4,
        ge=0,
        description="Learning rate for RND predictor network.",
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
    round: int = Field(default=1, ge=1, description="Training round number")
    version: str = Field(default="", description="Free-form version tag")
    parent_run: str | None = Field(default=None, description="Run this was resumed from")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure experiment name is filesystem-safe."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Name must be alphanumeric with underscores/dashes only")
        return v

    def config_hash(self) -> str:
        """SHA256 hash of the serialized config for deduplication."""
        import hashlib
        import json

        # Hash only the training-relevant fields, not metadata
        data = {
            "environment": self.environment.model_dump(),
            "ppo": self.ppo.model_dump(),
            "training": self.training.model_dump(),
        }
        blob = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def to_ppo_kwargs(self) -> dict:
        """Convert PPO config to kwargs for SB3 PPO."""
        from golds.training.schedules import cosine_schedule, linear_schedule

        lr = self.ppo.learning_rate
        if self.ppo.lr_schedule == "linear":
            lr = linear_schedule(self.ppo.learning_rate)
        elif self.ppo.lr_schedule == "cosine":
            lr = cosine_schedule(self.ppo.learning_rate)

        clip = self.ppo.clip_range
        if self.ppo.clip_schedule == "linear":
            clip = linear_schedule(self.ppo.clip_range)

        return {
            "learning_rate": lr,
            "n_steps": self.ppo.n_steps,
            "batch_size": self.ppo.batch_size,
            "n_epochs": self.ppo.n_epochs,
            "gamma": self.ppo.gamma,
            "gae_lambda": self.ppo.gae_lambda,
            "clip_range": clip,
            "clip_range_vf": self.ppo.clip_range_vf,
            "ent_coef": self.ppo.ent_coef,
            "vf_coef": self.ppo.vf_coef,
            "max_grad_norm": self.ppo.max_grad_norm,
        }
