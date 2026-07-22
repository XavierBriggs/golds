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
    stochastic_frameskip: bool = Field(
        default=False,
        description=(
            "Sample frame skip uniformly from [frameskip_min, frameskip_max] each "
            "step instead of a fixed frame_skip (the Sonic/Retro benchmark recipe: "
            "timing noise breaks deterministic freeze/oscillation loops). "
            "0 = disabled, falls back to fixed frame_skip."
        ),
    )
    frameskip_min: int = Field(
        default=2, ge=1, description="Minimum frame skip when stochastic_frameskip=True."
    )
    frameskip_max: int = Field(
        default=4, ge=1, description="Maximum frame skip when stochastic_frameskip=True."
    )
    screen_size: int = Field(default=84, ge=1, description="Screen size after resize")
    grayscale: bool = Field(default=True, description="Convert to grayscale")
    clip_reward: bool = Field(default=True, description="Clip rewards to {-1, 0, 1}")
    eval_clip_reward: bool | None = Field(
        default=None,
        description="Clip rewards during eval. None inherits clip_reward. "
        "Set false for a raw-score eval comparable to published baselines.",
    )
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
    progress_mode: Literal["delta_x", "delta_max_x"] = Field(
        default="delta_x",
        description=(
            "X-position reward shaping mode. 'delta_x' (default) rewards raw "
            "per-step movement and punishes backtracking; 'delta_max_x' rewards "
            "only new furthest-right progress (the standard Sonic recipe, "
            "ADR-004) and never punishes backtracking."
        ),
    )
    level_end_x: float | None = Field(
        default=None,
        description=(
            "Per-level x-position threshold for completion detection. None "
            "disables threshold-based completion (placeholder until the value "
            "is known from ROM data, e.g. GHZ Act 1)."
        ),
    )
    completion_bonus: float = Field(
        default=0.0,
        ge=0,
        description="One-time bonus reward awarded when level completion is first detected.",
    )
    level_end_info_key: str | None = Field(
        default=None,
        description=(
            "Optional retro info dict key that signals level completion when "
            "truthy (e.g. a level/act-change flag). None disables this signal."
        ),
    )
    stall_limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Truncate the episode (timeout, not death) after this many steps "
            "with no improvement in max_x. None disables anti-stall termination "
            "(backward compatible)."
        ),
    )


class TrainingConfig(BaseModel):
    """Training configuration."""

    total_timesteps: int = Field(default=10_000_000, ge=1, description="Total training timesteps")
    eval_freq: int = Field(
        default=50_000,
        ge=0,
        description="Evaluation frequency (timesteps); 0 disables evaluation",
    )
    eval_episodes: int = Field(
        default=10,
        ge=1,
        description="Episodes for the FINAL end-of-run eval (the formal baseline / "
        "north-star completion-rate measurement). Keep high (e.g. 100).",
    )
    periodic_eval_episodes: int = Field(
        default=10,
        ge=1,
        description="Episodes per PERIODIC in-training eval (every eval_freq steps). "
        "Keep small: 100-episode periodic evals dominate wall-clock, especially in "
        "single-emulator retro. Decoupled from eval_episodes.",
    )
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
    video_freq: int = Field(
        default=0,
        ge=0,
        description="Record and send a gameplay video every N timesteps (0 = disabled). Also records at start and end.",
    )
    video_length: int = Field(
        default=4000,
        ge=100,
        description="Number of frames per recorded video.",
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


class WandbConfig(BaseModel):
    """Weights & Biases observability configuration (M1b/T5, R8).

    Disabled by default so existing runs and tests are unaffected. When
    enabled, ``WandbCallback`` logs the full experiment config plus git
    provenance, PPO health metrics, and system stats. It never crashes
    training if W&B is unreachable, unauthenticated, or the package is
    missing (see ``golds.training.wandb_callback``). Video logging is
    deferred to M3 (needs the retro video-subprocess path).
    """

    enabled: bool = Field(
        default=False, description="Enable the W&B observability callback for this run."
    )
    project: str = Field(default="golds", description="W&B project name.")
    entity: str | None = Field(
        default=None, description="W&B entity (team/user). None uses the W&B default."
    )
    mode: Literal["online", "offline", "disabled"] | None = Field(
        default=None,
        description="W&B run mode override. None defers to the WANDB_MODE env var / W&B default.",
    )
    tags: list[str] = Field(default_factory=list, description="Tags to attach to the W&B run.")


class InvariantChecksConfig(BaseModel):
    """Live PPO training-health invariant-check configuration (R10, G5).

    Disabled by default so existing runs and tests are unaffected. When
    enabled, ``PPOInvariantCallback`` checks clip_fraction, approx_kl,
    explained_variance trend, and rollout-buffer advantage health at the
    end of every rollout, so a bad run is caught within a few rollouts
    instead of hours later. Diagnostic by default (violations are logged
    and recorded, not fatal); see ``golds.training.invariant_callback``.
    """

    enabled: bool = Field(
        default=False, description="Enable the PPO invariant-check callback for this run."
    )
    clip_fraction_min: float = Field(
        default=0.0, description="Exclusive lower bound for a healthy clip_fraction."
    )
    clip_fraction_max: float = Field(
        default=0.3, description="Exclusive upper bound for a healthy clip_fraction."
    )
    approx_kl_max: float = Field(
        default=0.05, description="Ceiling for approx_kl; at/above this is a violation."
    )
    explained_variance_window: int = Field(
        default=5,
        ge=2,
        description="Rolling window size (updates) for the explained_variance trend check.",
    )
    explained_variance_drop: float = Field(
        default=0.3,
        description="How far below the rolling mean explained_variance may drop before flagging.",
    )
    explained_variance_grace_updates: int = Field(
        default=5,
        ge=0,
        description="Updates explained_variance may stay negative before 'stuck negative' flags.",
    )
    advantage_std_min: float = Field(
        default=1e-6,
        ge=0,
        description="Minimum std of raw rollout-buffer advantages before they're degenerate.",
    )
    strict: bool = Field(
        default=False,
        description="If True, raise on the first violation instead of only recording it. "
        "Intended for tests/CI, not production runs.",
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
    wandb: WandbConfig = Field(
        default_factory=WandbConfig, description="W&B observability configuration."
    )
    invariant_checks: InvariantChecksConfig = Field(
        default_factory=InvariantChecksConfig,
        description="PPO training-health invariant-check configuration.",
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
