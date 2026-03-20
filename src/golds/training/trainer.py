"""Main training orchestrator."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecEnv

from golds.config.schema import ExperimentConfig
from golds.environments.factory import EnvironmentFactory
from golds.training.callbacks import (
    ProgressCallback,
    ResultsCallback,
    SafeCheckpointCallback,
    SaveOnBestTrainingRewardCallback,
    SelfPlaySnapshotCallback,
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
            state=env_config.state,
            use_subproc=env_config.use_subproc,
            reward_regime=env_config.reward_regime,
            players=env_config.players,
            opponent_mode=env_config.opponent,
            opponent_model_path=env_config.opponent_model_path,
            opponent_snapshot_dir=str(self.output_dir / "self_play" / "opponents")
            if env_config.opponent == "self_play"
            else None,
            wrapper_kwargs={
                "terminal_on_life_loss": env_config.terminal_on_life_loss,
                "clip_reward": env_config.clip_reward,
            },
        )

    def _create_eval_env(self) -> VecEnv:
        """Create evaluation environment."""
        env_config = self.config.environment

        opponent_mode = env_config.opponent
        opponent_model_path = env_config.opponent_model_path
        opponent_snapshot_dir = (
            str(self.output_dir / "self_play" / "opponents")
            if env_config.opponent == "self_play"
            else None
        )
        # Keep evaluation stationary for self-play training so "best_model" is meaningful.
        if env_config.players == 2 and env_config.opponent == "self_play":
            opponent_mode = "noop"
            opponent_model_path = None
            opponent_snapshot_dir = None

        return EnvironmentFactory.create_eval_env(
            game_id=env_config.game_id,
            frame_stack=env_config.frame_stack,
            seed=self.config.training.seed,
            state=env_config.state,
            reward_regime=env_config.reward_regime,
            players=env_config.players,
            opponent_mode=opponent_mode,
            opponent_model_path=opponent_model_path,
            opponent_snapshot_dir=opponent_snapshot_dir,
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

        policy = self.config.ppo.policy

        # Select PPO implementation
        if policy == "CnnLstmPolicy":
            try:
                from sb3_contrib import RecurrentPPO
            except ImportError:
                raise ImportError(
                    "RecurrentPPO requires sb3-contrib. Install with: pip install sb3-contrib"
                )
            ppo_cls = RecurrentPPO
            console.print("[cyan]Using RecurrentPPO (LSTM policy)[/cyan]")
        else:
            ppo_cls = PPO

        if self.resume_from and self.resume_from.exists():
            console.print(f"[yellow]Resuming from: {self.resume_from}[/yellow]")
            model = ppo_cls.load(self.resume_from, env=train_env, device=device)

            # Load VecNormalize stats if they exist
            vec_norm_path = self.model_dir / "vec_normalize.pkl"
            if vec_norm_path.exists():
                from stable_baselines3.common.vec_env import VecNormalize as VecNorm

                env = train_env
                while env is not None:
                    if isinstance(env, VecNorm):
                        VecNorm.load(str(vec_norm_path), env)
                        break
                    env = getattr(env, "venv", None)
        else:
            ppo_kwargs = self.config.to_ppo_kwargs()
            model = ppo_cls(
                policy=policy,
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

        # Evaluation callback (optional)
        if self.config.training.eval_freq > 0:
            eval_callback = create_eval_callback(
                eval_env=eval_env,
                log_dir=self.output_dir,
                eval_freq=self.config.training.eval_freq,
                n_eval_episodes=self.config.training.eval_episodes,
                n_envs=self.config.environment.n_envs,
                deterministic=self.config.training.eval_deterministic,
            )
            callbacks.append(eval_callback)

        # Checkpoint callback (optional)
        if self.config.training.save_freq > 0:
            checkpoint_callback = SafeCheckpointCallback(
                save_freq=max(1, self.config.training.save_freq // self.config.environment.n_envs),
                save_path=str(self.checkpoint_dir),
                name_prefix=self.config.name,
            )
            callbacks.append(checkpoint_callback)

        # Training reward callback
        reward_callback = SaveOnBestTrainingRewardCallback(
            check_freq=max(1, 10_000 // self.config.environment.n_envs),
            log_dir=self.model_dir / "best_training",
            model_name="best_training_model",
        )
        callbacks.append(reward_callback)

        # Progress callback (with Telegram updates every 1M steps)
        progress_callback = ProgressCallback(
            total_timesteps=self.config.training.total_timesteps,
            display_freq=50_000,
            telegram_freq=1_000_000,
            game_id=self.config.environment.game_id,
            experiment_name=self.config.name,
        )
        callbacks.append(progress_callback)

        # Self-play snapshots (optional)
        if (
            self.config.environment.players == 2
            and self.config.environment.opponent == "self_play"
            and self.config.training.self_play_snapshot_freq > 0
        ):
            snapshot_dir = self.output_dir / "self_play" / "opponents"
            callbacks.append(
                SelfPlaySnapshotCallback(
                    snapshot_dir=snapshot_dir,
                    save_freq=self.config.training.self_play_snapshot_freq,
                    max_snapshots=self.config.training.self_play_max_snapshots,
                    verbose=0,
                )
            )

        # Results tracking callback
        results_callback = ResultsCallback(
            game_id=self.config.environment.game_id,
            experiment_name=self.config.name,
            config_hash=self.config.config_hash(),
            round=self.config.round,
            total_timesteps_target=self.config.training.total_timesteps,
            device=get_device(self.config.training.device),
            n_envs=self.config.environment.n_envs,
            reward_regime=self.config.environment.reward_regime,
            output_dir=str(self.output_dir),
            resumed_from=str(self.resume_from) if self.resume_from else None,
            verbose=1,
        )
        callbacks.append(results_callback)

        return CallbackList(callbacks)

    def _check_output_disk_space(self) -> None:
        """Fail fast when the output filesystem is likely to run out of space.

        On Windows-mounted filesystems under WSL (e.g. `/mnt/c`), disk-full
        conditions sometimes surface as `OSError: [Errno 5] Input/output error`
        during checkpoint saves.
        """
        ignore = os.environ.get("GOLDS_IGNORE_DISK_SPACE", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if ignore:
            return

        try:
            usage = shutil.disk_usage(self.output_dir)
        except Exception:
            # If we cannot determine disk usage, don't block training.
            return

        free = usage.free
        total = usage.total
        used = usage.used
        used_pct = (used / total) * 100 if total else 0.0

        # Require more headroom when we are writing frequent artifacts.
        # (TensorBoard logs, eval logs, best model, checkpoints, final model)
        min_free_bytes = 2 * 1024**3  # 2 GiB
        if free >= min_free_bytes and used_pct < 99.5:
            return

        out = self.output_dir.resolve().as_posix()
        is_wsl_windows_mount = out.startswith("/mnt/") and len(out.split("/", 3)) >= 3

        def fmt(n: int) -> str:
            for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
                if n < 1024:
                    return f"{n:.1f} {unit}"
                n /= 1024
            return f"{n:.1f} PiB"

        hint = (
            "Free up disk space or change the output directory to a Linux filesystem "
            "(recommended under WSL)."
        )
        if is_wsl_windows_mount:
            hint += " Example: `uv run golds train run ... --output ~/golds_outputs`"

        msg = (
            f"Low disk space for training outputs.\n"
            f"- Output dir: {self.output_dir}\n"
            f"- Free: {fmt(int(free))} / Total: {fmt(int(total))} ({used_pct:.1f}% used)\n"
            f"- {hint}\n"
            f"- Set `GOLDS_IGNORE_DISK_SPACE=1` to bypass this check."
        )

        raise RuntimeError(msg)

    @staticmethod
    def _reraise_storage_error(exc: OSError, context: str) -> None:
        errno = getattr(exc, "errno", None)
        if errno not in {5, 28}:
            raise

        raise RuntimeError(
            f"{context} failed due to a filesystem I/O error ({exc}).\n"
            "This is commonly caused by running out of disk space on Windows-mounted "
            "drives under WSL.\n"
            "Fix: free disk space, or run training with `--output` pointing to a Linux "
            "filesystem path (e.g. `~/golds_outputs`)."
        ) from exc

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

        # Telegram notifications (best-effort, never blocks training)
        from golds.notifications.telegram import TelegramNotifier

        notifier = TelegramNotifier()
        notifier.send_training_start(
            experiment_name=self.config.name,
            game_id=self.config.environment.game_id,
            total_timesteps=self.config.training.total_timesteps,
            device=get_device(self.config.training.device),
        )

        # Fail fast if the output filesystem is close to full.
        self._check_output_disk_space()

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

            try:
                total_timesteps = self.config.training.total_timesteps
                if self.resume_from is not None:
                    # SB3 `learn(total_timesteps=...)` interprets `total_timesteps` as
                    # "additional timesteps" when `reset_num_timesteps=False` by adding
                    # the current `model.num_timesteps` internally. We want config
                    # `total_timesteps` to behave like a target total, so we convert
                    # it into a "remaining timesteps" budget here.
                    already = int(getattr(model, "num_timesteps", 0) or 0)
                    remaining = max(0, total_timesteps - already)
                    console.print(
                        f"[yellow]Resume progress: {already:,}/{total_timesteps:,} "
                        f"(remaining: {remaining:,})[/yellow]"
                    )
                    total_timesteps = remaining

                if total_timesteps <= 0:
                    console.print(
                        "[yellow]Target timesteps already reached; skipping training loop.[/yellow]"
                    )
                    if self.resume_from is not None:
                        console.print(f"[yellow]Already completed: {already:,} timesteps[/yellow]")
                    # Copy best eval model to final if it exists
                    best_model = self.output_dir / "best" / "best_model.zip"
                    if best_model.exists():
                        final_path = self.model_dir / "final_model.zip"
                        shutil.copy2(best_model, final_path)
                        console.print(f"[green]Copied best eval model to: {final_path}[/green]")
                else:
                    model.learn(
                        total_timesteps=total_timesteps,
                        callback=callbacks,
                        log_interval=self.config.training.log_interval,
                        tb_log_name=self.config.name,
                        reset_num_timesteps=self.resume_from is None,
                    )
            except OSError as e:
                self._reraise_storage_error(e, "Training (checkpoint/log save)")

            # Save VecNormalize stats if applicable
            from stable_baselines3.common.vec_env import VecNormalize as VecNorm

            env = train_env
            while env is not None:
                if isinstance(env, VecNorm):
                    env.save(str(self.model_dir / "vec_normalize.pkl"))
                    break
                env = getattr(env, "venv", None)

            # Save final model
            final_model_path = self.model_dir / "final_model"
            try:
                model.save(final_model_path)
            except OSError as e:
                self._reraise_storage_error(e, "Final model save")
            console.print(f"[green]Final model saved to: {final_model_path}[/green]")

            notifier.send_training_complete(
                experiment_name=self.config.name,
                game_id=self.config.environment.game_id,
                wall_time_seconds=0.0,  # ResultsCallback tracks precise time
                total_timesteps=int(getattr(model, "num_timesteps", 0) or 0),
            )

            return model

        except Exception as e:
            notifier.send_training_failed(
                experiment_name=self.config.name,
                game_id=self.config.environment.game_id,
                error=str(e),
            )
            raise

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
