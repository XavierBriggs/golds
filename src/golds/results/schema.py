"""Structured training and evaluation result schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class EvalResult(BaseModel):
    """Single evaluation run results."""

    mean_reward: float = Field(description="Mean episode reward across evaluation episodes.")
    std_reward: float = Field(description="Standard deviation of episode rewards.")
    min_reward: float = Field(description="Minimum episode reward observed.")
    max_reward: float = Field(description="Maximum episode reward observed.")
    median_reward: float = Field(default=0.0, description="Median episode reward.")
    mean_length: float = Field(description="Mean episode length in steps.")
    std_length: float = Field(description="Standard deviation of episode lengths.")
    n_episodes: int = Field(description="Number of evaluation episodes run.")
    deterministic: bool = Field(default=True, description="Whether deterministic policy was used.")
    seed: int | None = Field(default=None, description="Random seed used for evaluation.")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the evaluation was run."
    )


class TrainingResult(BaseModel):
    """Complete training run results.

    Captures identity, metrics, paths, metadata, and comparison baselines
    for a single training run.
    """

    # Identity
    game_id: str = Field(description="Atari game identifier (e.g. 'breakout').")
    experiment_name: str = Field(description="Human-readable experiment name.")
    config_hash: str = Field(description="Hash of the training configuration for dedup.")
    round: int = Field(default=1, description="Training round number (for multi-round runs).")

    # Training metrics
    total_timesteps_completed: int = Field(description="Actual timesteps completed.")
    total_timesteps_target: int = Field(description="Target timesteps for the run.")
    wall_time_seconds: float = Field(description="Wall-clock training duration in seconds.")
    best_eval_reward: float | None = Field(
        default=None, description="Best mean eval reward seen during training."
    )
    final_eval_reward: float | None = Field(
        default=None, description="Final evaluation mean reward."
    )

    # Evaluation
    eval_100ep: EvalResult | None = Field(
        default=None, description="Full 100-episode evaluation results."
    )

    # Paths (relative to outputs/)
    best_model_path: str = Field(default="", description="Path to best model checkpoint.")
    final_model_path: str = Field(default="", description="Path to final model checkpoint.")
    tensorboard_log_dir: str = Field(default="", description="Path to TensorBoard log directory.")

    # Metadata
    device: str = Field(default="auto", description="Training device (e.g. 'cuda', 'cpu', 'auto').")
    n_envs: int = Field(default=1, description="Number of parallel training environments.")
    started_at: datetime = Field(
        default_factory=datetime.now, description="Training start timestamp."
    )
    completed_at: datetime | None = Field(
        default=None, description="Training completion timestamp."
    )
    exit_code: int = Field(default=0, description="Process exit code (0 = success).")
    resumed_from: str | None = Field(
        default=None, description="Path to model this run was resumed from."
    )
    reward_regime: str = Field(default="clipped", description="Reward regime ('clipped' or 'raw').")

    # Comparison
    human_score: float | None = Field(
        default=None, description="Published human score for comparison."
    )
    human_normalized_score: float | None = Field(
        default=None, description="Human-normalized score: (agent - random) / (human - random)."
    )
    published_ppo_score: float | None = Field(
        default=None, description="Published PPO baseline score."
    )
