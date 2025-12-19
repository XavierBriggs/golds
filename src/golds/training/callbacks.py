"""Custom callbacks for training."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, is_vecenv_wrapped
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
                    try:
                        self.model.save(self.log_dir / self.model_name)
                    except OSError as e:
                        if self.verbose > 0:
                            print(
                                f"[warn] best_training save failed ({e}); continuing without saving."
                            )

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
    deterministic: bool = True,
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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._eval_start_time: float | None = None
        self._eval_last_heartbeat: float = 0.0
        self._eval_episodes_done: int = 0
        try:
            self._eval_heartbeat_seconds = float(os.environ.get("GOLDS_EVAL_HEARTBEAT_SECONDS", "60"))
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
                l = ep.get("l")
                print(
                    f"[eval] episode {self._eval_episodes_done}/{self.n_eval_episodes} done: "
                    f"reward={r} len={l}",
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
