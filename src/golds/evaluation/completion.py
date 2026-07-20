"""Completion-rate evaluation (R11 / north-star goal G6).

Aggregates ``PlatformerRewardWrapper``'s per-step ``info["level_complete"]``
signal into a completion RATE over N episodes. This is the measurement
behind G6: "the agent completes GHZ Act 1 in >= 80% of a 100-episode
deterministic eval" (see ``docs/inception/spec.md``).

Detection is done entirely through the info dict returned by ``VecEnv.step``,
never by reaching into the wrapper object through the vec env. Info flows
cleanly through SB3 VecEnv wrappers (``VecFrameStack``, ``VecTransposeImage``,
etc.); wrapper-object access does not, so this is the only approach that
survives the full training/eval env stack.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_MAX_X_INFO_KEY = "x"


def evaluate_completion_rate(
    model: Any,
    eval_env: Any,
    n_episodes: int,
    deterministic: bool = True,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run ``n_episodes`` episodes and compute the level-completion rate.

    An episode counts as "completed" if ``info["level_complete"]`` was ever
    True on any step of that episode — even if the episode continues running
    after the signpost is reached (matches ``PlatformerRewardWrapper``, which
    latches ``_completed`` for the rest of the episode).

    Deterministic Sonic episodes can get stuck (agent never reaches the
    signpost) and then run until the in-game timer expires, which can take a
    very long time or effectively never end depending on the ROM/state. If
    ``max_steps`` is set, any episode that reaches that many steps without a
    natural ``done`` is force-ended: it is recorded with whatever
    ``level_complete`` had latched to by then (usually False) and a
    ``capped=True`` marker, and the eval moves on to the next episode instead
    of looping forever.

    Args:
        model: An SB3-like model exposing ``predict(obs, deterministic=...)``.
        eval_env: An SB3 VecEnv (or compatible fake) exposing ``reset()`` ->
            obs and ``step(action)`` -> ``(obs, rewards, dones, infos)``,
            where ``infos`` is a list/tuple of per-sub-env info dicts.
        n_episodes: Number of episodes to evaluate (deterministic protocol
            uses 100 per the north-star spec).
        deterministic: Use the deterministic policy (default True, matching
            the north-star eval protocol).
        max_steps: Optional per-episode step cap. ``None`` (default)
            preserves the old unbounded behavior: episodes only end via the
            env's own ``done`` signal, however long that takes. Capping is
            only reliable for a single-env VecEnv (``num_envs == 1``): when
            the cap fires we force the next episode by calling
            ``eval_env.reset()`` directly, which restarts every sub-env. With
            ``num_envs > 1`` that reset would silently truncate every other
            sub-env's in-progress episode and corrupt its counts, so this
            raises ``ValueError`` instead of guessing. Callers that need
            capping with a multi-env VecEnv must build a ``num_envs == 1``
            eval env (see ``EnvironmentFactory.create_eval_env``, which
            always does this).

    Returns:
        dict with keys:
        - ``completion_rate``: float in [0, 1], ``n_completed / n_episodes``.
        - ``n_episodes``: int, number of episodes actually recorded.
        - ``n_completed``: int, number of those episodes that completed.
        - ``n_capped``: int, number of episodes force-ended by ``max_steps``
          (always 0 when ``max_steps`` is None).
        - ``mean_reward``: float, mean per-episode summed reward.
        - ``mean_max_x``: float | None, mean of per-episode max x-position,
          only over episodes where an ``"x"`` key was observed in info (None
          if never observed — i.e. not available for this game/wrapper).
        - ``per_episode``: list of dicts, one per episode, each with
          ``completed`` (bool), ``reward`` (float), ``max_x``
          (float | None), and ``capped`` (bool).

    Raises:
        ValueError: if ``max_steps`` is set and ``eval_env.num_envs != 1``.
    """
    num_envs = getattr(eval_env, "num_envs", 1)

    if max_steps is not None and num_envs != 1:
        raise ValueError(
            f"max_steps={max_steps} requires eval_env.num_envs == 1 "
            f"(got {num_envs}). Capping a multi-env VecEnv would require "
            "resetting ALL sub-envs when just one hits the cap, silently "
            "truncating the others' in-progress episodes and corrupting "
            "their counts. Build the eval env with num_envs=1, or leave "
            "max_steps=None for the old unbounded behavior."
        )

    per_episode: list[dict[str, Any]] = []

    cur_completed = [False] * num_envs
    cur_reward = [0.0] * num_envs
    cur_max_x: list[float | None] = [None] * num_envs
    cur_steps = [0] * num_envs

    obs = eval_env.reset()

    while len(per_episode) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = eval_env.step(action)

        force_reset = False
        for i in range(num_envs):
            info = infos[i]
            cur_reward[i] += float(rewards[i])
            cur_steps[i] += 1

            if info.get("level_complete"):
                cur_completed[i] = True

            if _MAX_X_INFO_KEY in info:
                x_val = float(info[_MAX_X_INFO_KEY])
                cur_max_x[i] = x_val if cur_max_x[i] is None else max(cur_max_x[i], x_val)

            done = bool(dones[i])
            capped = not done and max_steps is not None and cur_steps[i] >= max_steps

            if done or capped:
                if len(per_episode) < n_episodes:
                    per_episode.append(
                        {
                            "completed": cur_completed[i],
                            "reward": cur_reward[i],
                            "max_x": cur_max_x[i],
                            "capped": capped,
                        }
                    )
                cur_completed[i] = False
                cur_reward[i] = 0.0
                cur_max_x[i] = None
                cur_steps[i] = 0
                # `done` episodes already caused the VecEnv to auto-reset
                # internally (standard SB3 semantics); a `capped` episode did
                # not end naturally, so we must force the reset ourselves.
                if capped and len(per_episode) < n_episodes:
                    force_reset = True

        if force_reset:
            # num_envs == 1 is guaranteed here (validated above), so this
            # can't clobber another sub-env's in-progress episode.
            obs = eval_env.reset()

    n_completed = sum(1 for ep in per_episode if ep["completed"])
    n_capped = sum(1 for ep in per_episode if ep["capped"])
    n_recorded = len(per_episode)
    completion_rate = (n_completed / n_recorded) if n_recorded else 0.0
    mean_reward = float(np.mean([ep["reward"] for ep in per_episode])) if per_episode else 0.0
    max_x_values = [ep["max_x"] for ep in per_episode if ep["max_x"] is not None]
    mean_max_x = float(np.mean(max_x_values)) if max_x_values else None

    return {
        "completion_rate": completion_rate,
        "n_episodes": n_recorded,
        "n_completed": n_completed,
        "n_capped": n_capped,
        "mean_reward": mean_reward,
        "mean_max_x": mean_max_x,
        "per_episode": per_episode,
    }
