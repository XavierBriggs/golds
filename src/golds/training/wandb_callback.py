"""Weights & Biases observability callback (M1b/T5, R8).

Logs, for every training run: the full experiment config plus git
provenance, PPO health metrics (episode reward, entropy, explained
variance, approx KL, clip fraction, value/policy loss), and system stats
(GPU utilization, env FPS).

Video logging is explicitly deferred (see ADR-003): it needs the retro
video-subprocess path built in M3, so this callback never attempts it.

Graceful degradation is the load-bearing property here: multi-day runs
must never die because W&B is unreachable, unauthenticated, or the
package is missing. Every W&B call is wrapped so a failure logs a
warning and training continues.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from golds.utils.git_info import get_git_provenance

try:
    import wandb

    _WANDB_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - defensive; wandb is a declared dependency
    wandb = None  # type: ignore[assignment]
    _WANDB_IMPORT_ERROR = e

# PPO health metrics recorded by SB3 under `self.model.logger.name_to_value`
# that we forward to W&B verbatim when present.
_PPO_METRIC_KEYS = (
    "train/entropy_loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
    "train/loss",
    "time/fps",
)


class WandbCallback(BaseCallback):
    """Log config, git provenance, PPO health metrics, and system stats to W&B.

    Never raises: if W&B is unavailable, unauthenticated, or ``wandb.init``
    fails for any reason, the callback disables itself and training
    continues unaffected.
    """

    def __init__(
        self,
        experiment_config: Any,
        project: str = "golds",
        entity: str | None = None,
        mode: str | None = None,
        tags: list[str] | None = None,
        verbose: int = 0,
    ) -> None:
        """Initialize the callback.

        Args:
            experiment_config: The full ``ExperimentConfig`` for this run
                (logged as the W&B run config, along with git provenance).
            project: W&B project name.
            entity: W&B entity (team/user). ``None`` uses the W&B default.
            mode: W&B run mode override (e.g. ``"offline"``, ``"disabled"``).
                ``None`` defers to the ``WANDB_MODE`` env var / W&B default.
            tags: Tags to attach to the W&B run.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.experiment_config = experiment_config
        self.project = project
        self.entity = entity
        self.mode = mode
        self.tags = list(tags) if tags else None
        self.enabled = False
        self._run: Any = None

    def _on_training_start(self) -> None:
        """Start a W&B run, logging the full config + git provenance.

        Best-effort: any failure disables W&B logging for this run without
        raising, so training always proceeds.
        """
        self.enabled = False
        self._run = None

        if wandb is None:
            if self.verbose > 0:
                print(
                    f"[wandb] package unavailable ({_WANDB_IMPORT_ERROR}); "
                    "disabling W&B logging for this run."
                )
            return

        try:
            config_dict = self.experiment_config.model_dump(mode="json")
        except Exception:
            config_dict = {}

        git_sha, git_dirty = get_git_provenance()
        config_dict["git_sha"] = git_sha
        config_dict["git_dirty"] = git_dirty

        run_name = getattr(self.experiment_config, "name", None)

        try:
            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=run_name,
                config=config_dict,
                mode=self.mode,
                tags=self.tags,
                reinit=True,
            )
            self.enabled = True
        except Exception as e:
            self.enabled = False
            self._run = None
            if self.verbose > 0:
                print(f"[wandb] init failed ({e}); continuing training without W&B logging.")

    def _collect_metrics(self) -> dict[str, float]:
        """Gather PPO health metrics + system stats currently available.

        Reads from the SB3 logger's ``name_to_value`` plus ``ep_info_buffer``
        (episode reward is read from the buffer directly rather than the
        logger, since the logger clears train/rollout keys on each dump).
        """
        metrics: dict[str, float] = {}

        name_to_value = {}
        try:
            name_to_value = dict(getattr(self.model.logger, "name_to_value", {}) or {})
        except Exception:
            name_to_value = {}

        for key in _PPO_METRIC_KEYS:
            if key in name_to_value:
                try:
                    metrics[key] = float(name_to_value[key])
                except (TypeError, ValueError):
                    pass

        try:
            ep_info_buffer = getattr(self.model, "ep_info_buffer", None)
            if ep_info_buffer:
                metrics["rollout/ep_rew_mean"] = float(
                    np.mean([ep_info["r"] for ep_info in ep_info_buffer])
                )
        except Exception:
            pass

        gpu_util = self._get_gpu_util()
        if gpu_util is not None:
            metrics["system/gpu_util"] = gpu_util

        return metrics

    @staticmethod
    def _get_gpu_util() -> float | None:
        """Best-effort GPU utilization (percent), or None if unavailable."""
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            return float(torch.cuda.utilization())
        except Exception:
            return None

    def _on_rollout_end(self) -> None:
        """Log PPO health metrics + system stats for the completed rollout."""
        if not self.enabled or wandb is None:
            return

        try:
            metrics = self._collect_metrics()
            if metrics:
                wandb.log(metrics, step=int(self.num_timesteps))
        except Exception as e:
            if self.verbose > 0:
                print(f"[wandb] log failed ({e}); continuing training without this update.")

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        """Finish the W&B run. Best-effort; never raises."""
        if not self.enabled or wandb is None:
            return

        try:
            wandb.finish()
        except Exception as e:
            if self.verbose > 0:
                print(f"[wandb] finish failed ({e}); continuing.")
        finally:
            self.enabled = False
            self._run = None
