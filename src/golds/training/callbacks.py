"""Custom callbacks for training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """Callback for saving the best model based on training reward.

    Monitors the mean reward over the last 100 episodes and saves
    the model when a new best is achieved.
    """

    def __init__(
        self,
        check_freq: int,
        log_dir: str | Path,
        model_name: str = "best_model",
        verbose: int = 1,
    ) -> None:
        """Initialize callback.

        Args:
            check_freq: Check for best model every N calls
            log_dir: Directory to save models
            model_name: Name for saved model
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = Path(log_dir)
        self.model_name = model_name
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        """Initialize callback after model setup."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """Check for best model."""
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward from info
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean(
                    [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                )
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Mean reward: {mean_reward:.2f} - "
                        f"Best: {self.best_mean_reward:.2f}"
                    )

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.log_dir}")
                    self.model.save(self.log_dir / self.model_name)

        return True


class TensorBoardVideoCallback(BaseCallback):
    """Callback for recording videos to TensorBoard."""

    def __init__(
        self,
        eval_env: VecEnv,
        record_freq: int = 100_000,
        n_episodes: int = 1,
        verbose: int = 0,
    ) -> None:
        """Initialize callback.

        Args:
            eval_env: Environment for recording
            record_freq: Record every N timesteps
            n_episodes: Number of episodes to record
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.record_freq = record_freq
        self.n_episodes = n_episodes

    def _on_step(self) -> bool:
        """Record video if needed."""
        if self.n_calls % self.record_freq == 0:
            # Recording would require additional setup with VecVideoRecorder
            # This is a placeholder for the recording logic
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Video recording triggered")
        return True


class ProgressCallback(BaseCallback):
    """Callback for displaying training progress."""

    def __init__(
        self,
        total_timesteps: int,
        display_freq: int = 10_000,
        verbose: int = 1,
    ) -> None:
        """Initialize callback.

        Args:
            total_timesteps: Total training timesteps
            display_freq: Display progress every N timesteps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.display_freq = display_freq

    def _on_step(self) -> bool:
        """Display progress."""
        if self.num_timesteps % self.display_freq == 0:
            progress = self.num_timesteps / self.total_timesteps * 100
            if self.verbose > 0:
                print(
                    f"Progress: {self.num_timesteps:,}/{self.total_timesteps:,} "
                    f"({progress:.1f}%)"
                )
        return True


def create_eval_callback(
    eval_env: VecEnv,
    log_dir: str | Path,
    eval_freq: int = 50_000,
    n_eval_episodes: int = 10,
    n_envs: int = 1,
) -> EvalCallback:
    """Create evaluation callback with best model saving.

    Args:
        eval_env: Environment for evaluation
        log_dir: Directory for logs and models
        eval_freq: Evaluation frequency in timesteps
        n_eval_episodes: Episodes per evaluation
        n_envs: Number of training environments (for freq adjustment)

    Returns:
        Configured EvalCallback
    """
    log_dir = Path(log_dir)
    best_model_dir = log_dir / "best"
    eval_log_dir = log_dir / "eval"

    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    return EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=eval_freq // n_envs,  # Adjust for vectorized env
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
