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
) -> dict[str, Any]:
    """Run ``n_episodes`` episodes and compute the level-completion rate.

    An episode counts as "completed" if ``info["level_complete"]`` was ever
    True on any step of that episode — even if the episode continues running
    after the signpost is reached (matches ``PlatformerRewardWrapper``, which
    latches ``_completed`` for the rest of the episode).

    Args:
        model: An SB3-like model exposing ``predict(obs, deterministic=...)``.
        eval_env: An SB3 VecEnv (or compatible fake) exposing ``reset()`` ->
            obs and ``step(action)`` -> ``(obs, rewards, dones, infos)``,
            where ``infos`` is a list/tuple of per-sub-env info dicts.
        n_episodes: Number of episodes to evaluate (deterministic protocol
            uses 100 per the north-star spec).
        deterministic: Use the deterministic policy (default True, matching
            the north-star eval protocol).

    Returns:
        dict with keys:
        - ``completion_rate``: float in [0, 1], ``n_completed / n_episodes``.
        - ``n_episodes``: int, number of episodes actually recorded.
        - ``n_completed``: int, number of those episodes that completed.
        - ``mean_reward``: float, mean per-episode summed reward.
        - ``mean_max_x``: float | None, mean of per-episode max x-position,
          only over episodes where an ``"x"`` key was observed in info (None
          if never observed — i.e. not available for this game/wrapper).
        - ``per_episode``: list of dicts, one per episode, each with
          ``completed`` (bool), ``reward`` (float), and ``max_x``
          (float | None).
    """
    num_envs = getattr(eval_env, "num_envs", 1)

    per_episode: list[dict[str, Any]] = []

    cur_completed = [False] * num_envs
    cur_reward = [0.0] * num_envs
    cur_max_x: list[float | None] = [None] * num_envs

    obs = eval_env.reset()

    while len(per_episode) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = eval_env.step(action)

        for i in range(num_envs):
            info = infos[i]
            cur_reward[i] += float(rewards[i])

            if info.get("level_complete"):
                cur_completed[i] = True

            if _MAX_X_INFO_KEY in info:
                x_val = float(info[_MAX_X_INFO_KEY])
                cur_max_x[i] = x_val if cur_max_x[i] is None else max(cur_max_x[i], x_val)

            if bool(dones[i]):
                if len(per_episode) < n_episodes:
                    per_episode.append(
                        {
                            "completed": cur_completed[i],
                            "reward": cur_reward[i],
                            "max_x": cur_max_x[i],
                        }
                    )
                cur_completed[i] = False
                cur_reward[i] = 0.0
                cur_max_x[i] = None

    n_completed = sum(1 for ep in per_episode if ep["completed"])
    n_recorded = len(per_episode)
    completion_rate = (n_completed / n_recorded) if n_recorded else 0.0
    mean_reward = (
        float(np.mean([ep["reward"] for ep in per_episode])) if per_episode else 0.0
    )
    max_x_values = [ep["max_x"] for ep in per_episode if ep["max_x"] is not None]
    mean_max_x = float(np.mean(max_x_values)) if max_x_values else None

    return {
        "completion_rate": completion_rate,
        "n_episodes": n_recorded,
        "n_completed": n_completed,
        "mean_reward": mean_reward,
        "mean_max_x": mean_max_x,
        "per_episode": per_episode,
    }
