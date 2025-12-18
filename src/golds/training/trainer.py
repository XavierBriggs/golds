"""Main training orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv

from golds.config.schema import ExperimentConfig
from golds.environments.factory import EnvironmentFactory
from golds.training.callbacks import (
    ProgressCallback,
    SaveOnBestTrainingRewardCallback,
    create_eval_callback,
)
from golds.utils.device import get_device

console = Console()


class Trainer:
    """Main training orchestrator.

    Handles environment creation, model setup, and training execution
    with proper callbacks and logging.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Path | str,
        resume_from: Path | str | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            config: Experiment configuration
            output_dir: Directory for outputs (models, logs, videos)
            resume_from: Path to checkpoint to resume from
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.resume_from = Path(resume_from) if resume_from else None

        # Create output directories
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "logs"
        self.video_dir = self.output_dir / "videos"
        self.checkpoint_dir = self.model_dir / "checkpoints"

        for d in [self.model_dir, self.log_dir, self.video_dir, self.checkpoint_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Save config
        self._save_config()

    def _save_config(self) -> None:
        """Save configuration to output directory."""
        import json

        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.model_dump(), f, indent=2)

    def _create_train_env(self) -> VecEnv:
        """Create training environment."""
        env_config = self.config.environment

        return EnvironmentFactory.create(
            game_id=env_config.game_id,
            n_envs=env_config.n_envs,
            frame_stack=env_config.frame_stack,
            seed=self.config.training.seed,
            use_subproc=env_config.use_subproc,
            wrapper_kwargs={
                "terminal_on_life_loss": env_config.terminal_on_life_loss,
                "clip_reward": env_config.clip_reward,
            },
        )

    def _create_eval_env(self) -> VecEnv:
        """Create evaluation environment."""
        env_config = self.config.environment

        return EnvironmentFactory.create_eval_env(
            game_id=env_config.game_id,
            frame_stack=env_config.frame_stack,
            seed=self.config.training.seed,
        )

    def _create_model(self, train_env: VecEnv) -> PPO:
        """Create or load PPO model.

        Args:
            train_env: Training environment

        Returns:
            PPO model
        """
        device = get_device(self.config.training.device)
        console.print(f"[cyan]Using device: {device}[/cyan]")

        if self.resume_from and self.resume_from.exists():
            console.print(f"[yellow]Resuming from: {self.resume_from}[/yellow]")
            model = PPO.load(self.resume_from, env=train_env, device=device)
        else:
            ppo_kwargs = self.config.to_ppo_kwargs()
            model = PPO(
                policy="CnnPolicy",
                env=train_env,
                tensorboard_log=str(self.log_dir),
                device=device,
                verbose=1,
                **ppo_kwargs,
            )

        return model

    def _create_callbacks(self, eval_env: VecEnv) -> CallbackList:
        """Create training callbacks.

        Args:
            eval_env: Evaluation environment

        Returns:
            CallbackList with all callbacks
        """
        callbacks = []

        # Evaluation callback
        eval_callback = create_eval_callback(
            eval_env=eval_env,
            log_dir=self.output_dir,
            eval_freq=self.config.training.eval_freq,
            n_eval_episodes=self.config.training.eval_episodes,
            n_envs=self.config.environment.n_envs,
        )
        callbacks.append(eval_callback)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.training.save_freq // self.config.environment.n_envs,
            save_path=str(self.checkpoint_dir),
            name_prefix=self.config.name,
        )
        callbacks.append(checkpoint_callback)

        # Training reward callback
        reward_callback = SaveOnBestTrainingRewardCallback(
            check_freq=10_000 // self.config.environment.n_envs,
            log_dir=self.model_dir / "best_training",
            model_name="best_training_model",
        )
        callbacks.append(reward_callback)

        # Progress callback
        progress_callback = ProgressCallback(
            total_timesteps=self.config.training.total_timesteps,
            display_freq=50_000,
        )
        callbacks.append(progress_callback)

        return CallbackList(callbacks)

    def train(self) -> PPO:
        """Execute training.

        Returns:
            Trained PPO model
        """
        console.print(f"[bold green]Starting training: {self.config.name}[/bold green]")
        console.print(f"Game: {self.config.environment.game_id}")
        console.print(f"Platform: {self.config.environment.platform}")
        console.print(f"Timesteps: {self.config.training.total_timesteps:,}")
        console.print(f"Parallel envs: {self.config.environment.n_envs}")
        console.print()

        # Create environments
        console.print("[cyan]Creating environments...[/cyan]")
        train_env = self._create_train_env()
        eval_env = self._create_eval_env()

        try:
            # Create model
            console.print("[cyan]Setting up PPO model...[/cyan]")
            model = self._create_model(train_env)

            # Create callbacks
            callbacks = self._create_callbacks(eval_env)

            # Train
            console.print("[bold green]Starting training loop...[/bold green]")
            console.print()

            model.learn(
                total_timesteps=self.config.training.total_timesteps,
                callback=callbacks,
                log_interval=self.config.training.log_interval,
                tb_log_name=self.config.name,
                reset_num_timesteps=self.resume_from is None,
            )

            # Save final model
            final_model_path = self.model_dir / "final_model"
            model.save(final_model_path)
            console.print(f"[green]Final model saved to: {final_model_path}[/green]")

            return model

        finally:
            train_env.close()
            eval_env.close()

    def get_model_paths(self) -> dict[str, Path]:
        """Get paths to saved models.

        Returns:
            Dictionary of model paths
        """
        return {
            "final": self.model_dir / "final_model.zip",
            "best_eval": self.output_dir / "best" / "best_model.zip",
            "best_training": self.model_dir / "best_training" / "best_training_model.zip",
            "checkpoints": self.checkpoint_dir,
        }
