"""Custom callbacks for training."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, is_vecenv_wrapped


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
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Mean reward: {mean_reward:.2f} - Best: {self.best_mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.log_dir}")
                    try:
                        self.model.save(self.log_dir / self.model_name)
                    except OSError as e:
                        if self.verbose > 0:
                            print(
                                f"[warn] best_training save failed ({e}); continuing without saving."
                            )

        return True


class ProgressCallback(BaseCallback):
    """Callback for displaying training progress and sending Telegram updates."""

    def __init__(
        self,
        total_timesteps: int,
        display_freq: int = 10_000,
        telegram_freq: int = 1_000_000,
        game_id: str = "",
        experiment_name: str = "",
        verbose: int = 1,
    ) -> None:
        """Initialize callback.

        Args:
            total_timesteps: Total training timesteps
            display_freq: Display progress every N timesteps
            telegram_freq: Send Telegram update every N timesteps
            game_id: Game identifier for Telegram messages
            experiment_name: Experiment name for Telegram messages
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.display_freq = display_freq
        self.telegram_freq = telegram_freq
        self.game_id = game_id
        self.experiment_name = experiment_name
        self._last_telegram_step = 0

    def _on_step(self) -> bool:
        """Display progress and send Telegram updates."""
        if self.num_timesteps % self.display_freq == 0:
            progress = self.num_timesteps / self.total_timesteps * 100
            if self.verbose > 0:
                print(
                    f"Progress: {self.num_timesteps:,}/{self.total_timesteps:,} ({progress:.1f}%)"
                )

        # Telegram update every telegram_freq steps
        if (
            self.telegram_freq > 0
            and self.num_timesteps - self._last_telegram_step >= self.telegram_freq
        ):
            self._last_telegram_step = self.num_timesteps
            progress = self.num_timesteps / self.total_timesteps * 100
            # Get current mean reward if available
            reward_str = ""
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                reward_str = f"\nMean Reward: {mean_reward:.1f}"
            try:
                from golds.notifications.telegram import TelegramNotifier

                notifier = TelegramNotifier()
                notifier.send(
                    f"📊 <b>Progress Update</b>\n"
                    f"Game: {self.game_id}\n"
                    f"Steps: {self.num_timesteps:,}/{self.total_timesteps:,} ({progress:.0f}%)"
                    f"{reward_str}"
                )
            except Exception:
                pass

        return True


def create_eval_callback(
    eval_env: VecEnv,
    log_dir: str | Path,
    eval_freq: int = 50_000,
    n_eval_episodes: int = 10,
    n_envs: int = 1,
    deterministic: bool = True,
) -> EvalCallback:
    """Create evaluation callback with best model saving.

    Args:
        eval_env: Environment for evaluation
        log_dir: Directory for logs and models
        eval_freq: Evaluation frequency in timesteps
        n_eval_episodes: Episodes per evaluation
        n_envs: Number of training environments (for freq adjustment)
        deterministic: Use deterministic policy during evaluation

    Returns:
        Configured EvalCallback
    """
    log_dir = Path(log_dir)
    best_model_dir = log_dir / "best"
    eval_log_dir = log_dir / "eval"

    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    # SB3 warns if eval env has no Monitor. However, wrapping a Gym Monitor inside
    # a VecMonitor can overwrite `info["episode"]` and change what's reported.
    # We only add VecMonitor when there is no existing Monitor wrapper detected.
    has_vec_monitor = False
    has_gym_monitor = False
    try:
        has_vec_monitor = is_vecenv_wrapped(eval_env, VecMonitor)
    except Exception:
        has_vec_monitor = False
    try:
        wrapped = eval_env.env_is_wrapped(Monitor)
        # Can be list-like (per-env) or scalar.
        if isinstance(wrapped, (list, tuple, np.ndarray)):
            has_gym_monitor = bool(wrapped[0])
        else:
            has_gym_monitor = bool(wrapped)
    except Exception:
        has_gym_monitor = False

    if not has_vec_monitor and not has_gym_monitor:
        eval_env = VecMonitor(eval_env)

    return VerboseEvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=max(1, eval_freq // n_envs),  # Adjust for vectorized env
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=False,
        warn=False,
    )


class SafeCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that never crashes training on save errors."""

    def _on_step(self) -> bool:
        try:
            return super()._on_step()
        except OSError as e:
            if getattr(self, "verbose", 0) > 0:
                print(f"[warn] checkpoint save failed ({e}); continuing without saving.")
            return True
        except Exception as e:
            if getattr(self, "verbose", 0) > 0:
                print(f"[warn] checkpoint callback error ({e}); continuing.")
            return True


class VerboseEvalCallback(EvalCallback):
    """EvalCallback with lightweight progress logging and safety.

    SB3's EvalCallback is often "silent" while evaluation is running, which can look
    like a hang for long-episode games. This subclass prints a start line and a
    periodic heartbeat during evaluation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._eval_start_time: float | None = None
        self._eval_last_heartbeat: float = 0.0
        self._eval_episodes_done: int = 0
        try:
            self._eval_heartbeat_seconds = float(
                os.environ.get("GOLDS_EVAL_HEARTBEAT_SECONDS", "60")
            )
        except Exception:
            self._eval_heartbeat_seconds = 60.0

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        super()._log_success_callback(locals_, globals_)

        if self._eval_start_time is None:
            return

        now = time.monotonic()
        if now - self._eval_last_heartbeat >= self._eval_heartbeat_seconds:
            self._eval_last_heartbeat = now
            if getattr(self, "verbose", 0) >= 1:
                print(
                    f"[eval] running... {self._eval_episodes_done}/{self.n_eval_episodes} episodes done "
                    f"(t={self.num_timesteps})",
                    flush=True,
                )

        done = bool(locals_.get("done", False))
        info = locals_.get("info", {}) or {}
        if done and isinstance(info, dict) and "episode" in info:
            self._eval_episodes_done += 1
            if getattr(self, "verbose", 0) >= 1:
                ep = info["episode"]
                r = ep.get("r")
                ep_len = ep.get("l")
                print(
                    f"[eval] episode {self._eval_episodes_done}/{self.n_eval_episodes} done: "
                    f"reward={r} len={ep_len}",
                    flush=True,
                )

    def _on_step(self) -> bool:
        is_eval_step = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if is_eval_step:
            self._eval_start_time = time.monotonic()
            self._eval_last_heartbeat = self._eval_start_time
            self._eval_episodes_done = 0
            if getattr(self, "verbose", 0) >= 1:
                print(
                    f"[eval] starting: n_eval_episodes={self.n_eval_episodes} "
                    f"deterministic={self.deterministic} (t={self.num_timesteps})",
                    flush=True,
                )

        try:
            ok = super()._on_step()
        except OSError as e:
            if getattr(self, "verbose", 0) >= 1:
                print(f"[warn] eval/best_model save failed ({e}); continuing.")
            ok = True
        except Exception as e:
            if getattr(self, "verbose", 0) >= 1:
                print(f"[warn] eval callback error ({e}); continuing.")
            ok = True

        if is_eval_step and self._eval_start_time is not None and getattr(self, "verbose", 0) >= 1:
            elapsed = time.monotonic() - self._eval_start_time
            print(f"[eval] finished in {elapsed:.1f}s (t={self.num_timesteps})", flush=True)
            self._eval_start_time = None

        return ok


class VideoProgressCallback(BaseCallback):
    """Record gameplay videos at regular intervals and send to Telegram.

    Records at: training start (step 0), every ``video_freq`` steps, and training end.
    Videos are saved locally to ``output_dir/videos/`` and sent via Telegram.
    """

    def __init__(
        self,
        game_id: str,
        output_dir: str | Path,
        video_freq: int = 10_000_000,
        video_length: int = 4000,
        n_envs: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.game_id = game_id
        self.output_dir = Path(output_dir)
        self.video_freq = video_freq
        self.video_length = video_length
        self.n_envs = n_envs
        self._last_video_step = -1
        self._video_dir = self.output_dir / "videos"
        self._recorded_start = False

    def _init_callback(self) -> None:
        self._video_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        # Record at step 0 (first call)
        if not self._recorded_start:
            self._recorded_start = True
            self._record_and_send(label="start")

        # Record every video_freq steps
        if (
            self.video_freq > 0
            and self.num_timesteps - self._last_video_step >= self.video_freq
        ):
            self._last_video_step = self.num_timesteps
            self._record_and_send(label=f"{self.num_timesteps // 1_000_000}M")

        return True

    def _on_training_end(self) -> None:
        self._record_and_send(label="final")

    def _record_and_send(self, label: str) -> None:
        """Record a video and send to Telegram. Best-effort, never crashes training."""
        import threading

        def _do_record():
            try:
                self._record_video(label)
            except Exception as e:
                if self.verbose > 0:
                    print(f"[video] recording failed: {e}")

        # Run in background thread so training isn't blocked
        t = threading.Thread(target=_do_record, daemon=True)
        t.start()

    def _record_video(self, label: str) -> None:
        """Actually record the video and send it."""
        from stable_baselines3.common.vec_env import VecVideoRecorder

        from golds.environments.factory import EnvironmentFactory

        # Create a fresh eval env
        env = EnvironmentFactory.create_eval_env(
            game_id=self.game_id, frame_stack=4, seed=0
        )

        video_name = f"{self.game_id}_{label}"
        env = VecVideoRecorder(
            env,
            str(self._video_dir),
            record_video_trigger=lambda x: x == 0,
            video_length=self.video_length,
            name_prefix=video_name,
        )

        # Use current model weights
        model = self.model
        obs = env.reset()

        for _ in range(self.video_length):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)
            if dones[0]:
                obs = env.reset()

        env.close()

        # Find the generated mp4
        import glob

        mp4_files = sorted(
            glob.glob(str(self._video_dir / f"{video_name}*.mp4")),
            key=os.path.getmtime,
            reverse=True,
        )
        if not mp4_files:
            return

        mp4_path = mp4_files[0]

        # Get current mean reward for caption
        reward_str = ""
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            reward_str = f"\nMean Reward: {mean_reward:.1f}"

        if self.verbose > 0:
            print(f"[video] saved: {mp4_path}")

        # Send to Telegram
        try:
            from golds.notifications.telegram import TelegramNotifier

            notifier = TelegramNotifier()
            if notifier.enabled:
                caption = (
                    f"🎮 <b>{self.game_id}</b> — {label}\n"
                    f"Steps: {self.num_timesteps:,}"
                    f"{reward_str}"
                )
                notifier.send_video(mp4_path, caption=caption)
        except Exception:
            pass  # Best-effort


class SelfPlaySnapshotCallback(BaseCallback):
    """Periodically save the current policy to a snapshot directory for self-play opponents.

    Uses an atomic rename so env workers don't try to load a partially-written file.
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        save_freq: int,
        max_snapshots: int = 5,
        prefix: str = "opponent",
        verbose: int = 0,
    ) -> None:
        """Initialize snapshot callback.

        Args:
            snapshot_dir: Directory to save opponent snapshots.
            save_freq: Save a snapshot every N timesteps.
            max_snapshots: Maximum number of snapshots to keep.
            prefix: Filename prefix for snapshot files.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.snapshot_dir = Path(snapshot_dir)
        self.save_freq = int(save_freq)
        self.max_snapshots = int(max_snapshots)
        self.prefix = prefix

    def _init_callback(self) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.save_freq <= 0:
            return True
        if self.num_timesteps <= 0:
            return True
        if self.num_timesteps % self.save_freq != 0:
            return True

        step = int(self.num_timesteps)
        final_path = self.snapshot_dir / f"{self.prefix}_{step}.zip"
        tmp_path = self.snapshot_dir / f".{self.prefix}_{step}.zip.tmp"

        try:
            self.model.save(tmp_path)
            tmp_path.replace(final_path)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            return True

        # Prune old snapshots by step number (best-effort).
        snapshots = sorted(self.snapshot_dir.glob(f"{self.prefix}_*.zip"))
        if len(snapshots) > self.max_snapshots:

            def step_key(p: Path) -> int:
                try:
                    return int(p.stem.rsplit("_", 1)[-1])
                except Exception:
                    return 0

            snapshots = sorted(snapshots, key=step_key, reverse=True)
            for p in snapshots[self.max_snapshots :]:
                try:
                    p.unlink()
                except Exception:
                    pass

        return True


class ResultsCallback(BaseCallback):
    """Record training results to the ResultStore on completion.

    Tracks wall-clock time and assembles a TrainingResult when training ends.
    """

    def __init__(
        self,
        game_id: str,
        experiment_name: str,
        config_hash: str,
        round: int,
        total_timesteps_target: int,
        device: str,
        n_envs: int,
        reward_regime: str = "clipped",
        output_dir: str | Path = "",
        results_path: str | Path = "results.json",
        resumed_from: str | None = None,
        verbose: int = 0,
    ) -> None:
        """Initialize results callback.

        Args:
            game_id: Atari game identifier.
            experiment_name: Human-readable experiment name.
            config_hash: Hash of the training config for dedup.
            round: Training round number.
            total_timesteps_target: Target timesteps for the run.
            device: Training device string.
            n_envs: Number of parallel environments.
            reward_regime: Reward regime ('clipped' or 'raw').
            output_dir: Base output directory for model paths.
            results_path: Path to the results JSON file.
            resumed_from: Path to model this run was resumed from.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.game_id = game_id
        self.experiment_name = experiment_name
        self.config_hash = config_hash
        self.round = round
        self.total_timesteps_target = total_timesteps_target
        self.device = device
        self.n_envs = n_envs
        self.reward_regime = reward_regime
        self.output_dir = str(output_dir)
        self.results_path = Path(results_path)
        self.resumed_from = resumed_from
        self._start_time: float | None = None

    def _on_training_start(self) -> None:
        self._start_time = time.monotonic()

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        """Assemble and save TrainingResult."""
        if self._start_time is None:
            return

        from datetime import datetime

        from golds.results.schema import TrainingResult
        from golds.results.store import ResultStore

        wall_time = time.monotonic() - self._start_time

        # Try to get best eval reward from sibling eval callback or eval log file
        best_eval_reward = None
        # Method 1: walk parent's callback list for EvalCallback.best_mean_reward
        for parent in [getattr(self, "parent", None)]:
            if parent is None:
                continue
            for cb in getattr(parent, "callbacks", []):
                if hasattr(cb, "best_mean_reward") and cb.best_mean_reward != -np.inf:
                    best_eval_reward = float(cb.best_mean_reward)
                    break
        # Method 2: read from eval log file (npz) if callback lookup failed
        if best_eval_reward is None:
            eval_npz = Path(self.output_dir) / "eval" / "evaluations.npz"
            if eval_npz.exists():
                try:
                    data = np.load(str(eval_npz))
                    if "results" in data:
                        # results shape: (n_evals, n_episodes)
                        mean_per_eval = data["results"].mean(axis=1)
                        best_eval_reward = float(mean_per_eval.max())
                except Exception:
                    pass

        result = TrainingResult(
            game_id=self.game_id,
            experiment_name=self.experiment_name,
            config_hash=self.config_hash,
            round=self.round,
            total_timesteps_completed=int(self.num_timesteps),
            total_timesteps_target=self.total_timesteps_target,
            wall_time_seconds=wall_time,
            best_eval_reward=best_eval_reward,
            device=self.device,
            n_envs=self.n_envs,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            exit_code=0,
            resumed_from=self.resumed_from,
            reward_regime=self.reward_regime,
            best_model_path=str(Path(self.output_dir) / "best" / "best_model.zip"),
            final_model_path=str(Path(self.output_dir) / "models" / "final_model.zip"),
            tensorboard_log_dir=str(Path(self.output_dir) / "logs"),
        )

        try:
            store = ResultStore(self.results_path)
            store.add_result(result)
            if self.verbose > 0:
                print(f"[results] Training result saved to {self.results_path}")
        except Exception as e:
            if self.verbose > 0:
                print(f"[warn] Failed to save training result: {e}")
