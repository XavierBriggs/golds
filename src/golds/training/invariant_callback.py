"""Live PPO training-health invariant checks (R10, G5).

Checks, on every completed rollout, whether the PPO run still looks
healthy: bounded clip fraction, bounded approx-KL, a non-collapsing
explained-variance trend, and finite/non-degenerate advantages. The goal
is to catch a bad run early (within a few rollouts) instead of only
discovering it hours later when eval reward has flatlined.

Diagnostic by default: a violation logs a loud warning and is recorded
via ``get_violations()``, but does NOT stop training. Set ``strict=True``
(intended for tests/CI, not long production runs) to raise
``PPOInvariantError`` on the first flagged invariant instead.

Invariants (see docs/inception/spec.md R10 and 03-synthesis.md G5):

1. ``clip_fraction`` stays within the open interval ``(clip_fraction_min,
   clip_fraction_max)`` (default ``(0, 0.3)``). 0 means the policy is
   never being clipped (learning rate/clip range likely too small or the
   policy has stopped moving); >= 0.3 means updates are too aggressive.
2. ``approx_kl`` stays below a configurable ceiling (default 0.05). A
   large KL step indicates the policy is moving too fast between
   updates and may be diverging.
3. ``explained_variance`` should trend upward over training, not
   collapse. We do NOT require strict monotonicity per update -- EV is
   noisy update-to-update. Instead we keep a short rolling window and
   flag either (a) a sharp drop: the latest value falls more than
   ``explained_variance_drop`` below the rolling mean of the window, or
   (b) EV still negative after ``explained_variance_grace_updates``
   updates (a value network that never learns to predict returns).
4. Rollout-buffer advantages are finite and non-degenerate.
   IMPORTANT LIMITATION: SB3 normalizes advantages per-minibatch inside
   ``PPO.train()``, which runs *after* ``_on_rollout_end`` fires. The
   ``rollout_buffer.advantages`` array seen here is therefore the RAW,
   pre-normalization GAE advantage, not the normalized value actually
   used in the policy loss. Asserting "zero-mean/unit-std" against the
   raw buffer would be checking something that isn't true by
   construction. Instead we check the honest, weaker property: the raw
   advantages are finite (no NaN/Inf) and have non-degenerate spread
   (std above ``advantage_std_min``) -- a collapsed or exploded
   advantage signal is a real problem even before normalization.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PPOInvariantError(RuntimeError):
    """Raised by ``PPOInvariantCallback`` when ``strict=True`` and an invariant fails."""


class PPOInvariantCallback(BaseCallback):
    """Check PPO training-health invariants at the end of every rollout.

    Reads ``train/clip_fraction``, ``train/approx_kl``, and
    ``train/explained_variance`` from ``self.model.logger.name_to_value``
    (populated by SB3 during ``PPO.train()``, same source used by
    ``WandbCallback``), and the raw advantages from
    ``self.model.rollout_buffer.advantages``.
    """

    def __init__(
        self,
        clip_fraction_min: float = 0.0,
        clip_fraction_max: float = 0.3,
        approx_kl_max: float = 0.05,
        explained_variance_window: int = 5,
        explained_variance_drop: float = 0.3,
        explained_variance_grace_updates: int = 5,
        advantage_std_min: float = 1e-6,
        strict: bool = False,
        verbose: int = 0,
    ) -> None:
        """Initialize the callback.

        Args:
            clip_fraction_min: Exclusive lower bound for a healthy clip_fraction.
            clip_fraction_max: Exclusive upper bound for a healthy clip_fraction.
            approx_kl_max: Exclusive ceiling for approx_kl; at/above this is a violation.
            explained_variance_window: Number of past updates kept for the
                explained_variance rolling-mean comparison.
            explained_variance_drop: How far below the rolling mean the latest
                explained_variance may fall before it's flagged as collapsing.
            explained_variance_grace_updates: Number of updates EV is allowed
                to stay negative before "stuck negative" is flagged.
            advantage_std_min: Minimum std of raw rollout-buffer advantages
                before they're considered degenerate.
            strict: If True, raise ``PPOInvariantError`` on the first
                flagged invariant instead of only recording it. Intended for
                tests/CI; production runs should stay diagnostic-only.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.clip_fraction_min = clip_fraction_min
        self.clip_fraction_max = clip_fraction_max
        self.approx_kl_max = approx_kl_max
        self.explained_variance_window = explained_variance_window
        self.explained_variance_drop = explained_variance_drop
        self.explained_variance_grace_updates = explained_variance_grace_updates
        self.advantage_std_min = advantage_std_min
        self.strict = strict

        self._ev_history: deque[float] = deque(maxlen=explained_variance_window)
        self._violations: list[dict[str, Any]] = []
        self._n_updates = 0

    def get_violations(self) -> list[dict[str, Any]]:
        """Return all invariant violations recorded so far.

        Each violation is a dict with keys ``name``, ``message``, ``value``,
        and ``step`` (the ``num_timesteps`` at which it was detected).
        """
        return list(self._violations)

    def _record_violation(self, name: str, message: str, value: float | None = None) -> None:
        violation = {
            "name": name,
            "message": message,
            "value": value,
            "step": int(getattr(self, "num_timesteps", 0)),
        }
        self._violations.append(violation)
        if self.verbose > 0:
            print(f"[ppo-invariant] VIOLATION ({name}) at step {violation['step']}: {message}")
        if self.strict:
            raise PPOInvariantError(f"{name}: {message}")

    def _get_metric(self, name_to_value: dict[str, Any], key: str) -> float | None:
        if key not in name_to_value:
            return None
        try:
            value = float(name_to_value[key])
        except (TypeError, ValueError):
            return None
        if np.isnan(value):
            return None
        return value

    def _check_clip_fraction(self, name_to_value: dict[str, Any]) -> None:
        cf = self._get_metric(name_to_value, "train/clip_fraction")
        if cf is None:
            return
        if not (self.clip_fraction_min < cf < self.clip_fraction_max):
            self._record_violation(
                "clip_fraction",
                f"clip_fraction {cf:.4f} outside the open interval "
                f"({self.clip_fraction_min}, {self.clip_fraction_max})",
                value=cf,
            )

    def _check_approx_kl(self, name_to_value: dict[str, Any]) -> None:
        kl = self._get_metric(name_to_value, "train/approx_kl")
        if kl is None:
            return
        if kl >= self.approx_kl_max:
            self._record_violation(
                "approx_kl",
                f"approx_kl {kl:.4f} at/above the ceiling {self.approx_kl_max}",
                value=kl,
            )

    def _check_explained_variance(self, name_to_value: dict[str, Any]) -> None:
        ev = self._get_metric(name_to_value, "train/explained_variance")
        if ev is None:
            return

        window = self._ev_history
        if len(window) >= 2:
            window_mean = float(np.mean(window))
            if ev < window_mean - self.explained_variance_drop:
                self._record_violation(
                    "explained_variance_trend",
                    f"explained_variance {ev:.3f} dropped more than "
                    f"{self.explained_variance_drop} below the rolling mean "
                    f"{window_mean:.3f} over the last {len(window)} updates",
                    value=ev,
                )

        if ev < 0 and self._n_updates > self.explained_variance_grace_updates:
            self._record_violation(
                "explained_variance_stuck_negative",
                f"explained_variance {ev:.3f} is still negative after "
                f"{self._n_updates} updates (grace period: "
                f"{self.explained_variance_grace_updates})",
                value=ev,
            )

        window.append(ev)

    def _check_advantages(self) -> None:
        rollout_buffer = getattr(self.model, "rollout_buffer", None)
        advantages = getattr(rollout_buffer, "advantages", None)
        if advantages is None:
            return

        arr = np.asarray(advantages, dtype=np.float64).ravel()
        if arr.size == 0:
            return

        if not np.all(np.isfinite(arr)):
            self._record_violation(
                "advantages_non_finite",
                "rollout buffer advantages contain NaN/Inf values (checked "
                "pre-normalization; see class docstring on the "
                "normalization-timing limitation)",
            )
            return

        std = float(arr.std())
        if std < self.advantage_std_min:
            self._record_violation(
                "advantages_degenerate",
                f"rollout buffer advantages have near-zero std ({std:.2e} < "
                f"{self.advantage_std_min}); degenerate/collapsed advantage "
                "signal (checked pre-normalization; see class docstring)",
                value=std,
            )

    def _on_rollout_end(self) -> None:
        """Check all invariants for the rollout that just completed."""
        self._n_updates += 1

        name_to_value: dict[str, Any] = {}
        try:
            name_to_value = dict(getattr(self.model.logger, "name_to_value", {}) or {})
        except Exception:
            name_to_value = {}

        self._check_clip_fraction(name_to_value)
        self._check_approx_kl(name_to_value)
        self._check_explained_variance(name_to_value)
        self._check_advantages()

    def _on_step(self) -> bool:
        return True
