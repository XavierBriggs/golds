"""Atari environment maker using gymnasium/ale-py with SB3 wrappers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from golds.environments.atari.env_id import resolve_atari_env_id

_ALE_PY_V0_V4_REGISTERED = False
_ALE_PY_V5_REGISTERED = False


def _register_ale_envs(env_id: str) -> None:
    """Register Atari envs in the current process (required for SubprocVecEnv workers)."""
    global _ALE_PY_V0_V4_REGISTERED, _ALE_PY_V5_REGISTERED

    # If the env is already registered in this process, do nothing.
    if env_id in gym.envs.registry:
        if env_id.startswith("ALE/"):
            _ALE_PY_V5_REGISTERED = True
        else:
            _ALE_PY_V0_V4_REGISTERED = True
        return

    try:
        import ale_py
    except ImportError as e:
        raise RuntimeError("Atari support requires `ale-py` (install `gymnasium[atari]`).") from e

    # Some versions of `ale-py` register envs at import time.
    if env_id in gym.envs.registry:
        if env_id.startswith("ALE/"):
            _ALE_PY_V5_REGISTERED = True
        else:
            _ALE_PY_V0_V4_REGISTERED = True
        return

    if env_id.startswith("ALE/"):
        if not _ALE_PY_V5_REGISTERED:
            ale_py.register_v5_envs()
            _ALE_PY_V5_REGISTERED = True
        return

    if not _ALE_PY_V0_V4_REGISTERED:
        ale_py.register_v0_v4_envs()
        _ALE_PY_V0_V4_REGISTERED = True


def _ensure_atari_registered(env_id: str) -> None:
    """Ensure Atari envs are available and `env_id` is registered.

    We do this before spawning subprocess envs so we fail fast with a useful
    error message (otherwise you tend to get `ConnectionResetError`).
    """
    # Register envs in the current process (Gymnasium does not auto-import ale-py).
    _register_ale_envs(env_id)
    try:
        gym.spec(env_id)
    except gym.error.Error as e:
        # Help users who omit the version suffix.
        suggested = resolve_atari_env_id(env_id)
        if suggested != env_id:
            _register_ale_envs(suggested)
            try:
                gym.spec(suggested)
            except gym.error.Error:
                pass
            else:
                raise RuntimeError(
                    f"Unknown Atari environment id: '{env_id}'. Try '{suggested}'."
                ) from e

        raise RuntimeError(f"Unknown Atari environment id: '{env_id}'.") from e


def make_atari_vec_env(
    env_id: str,
    n_envs: int = 8,
    seed: int | None = None,
    state: str | None = None,  # Ignored for Atari
    use_subproc: bool = True,
    wrapper_kwargs: dict | None = None,
    **kwargs,
) -> VecEnv:
    """Create a vectorized Atari environment with DeepMind preprocessing.

    Uses SB3's AtariWrapper preprocessing which applies:
    - NoopResetEnv: Random no-ops at start
    - MaxAndSkipEnv: Frame skipping with max pooling
    - EpisodicLifeEnv: Episode ends on life loss (optional)
    - FireResetEnv: Fire on reset for games that require it
    - WarpFrame: Grayscale and resize to 84x84
    - ClipRewardEnv: Clip rewards to {-1, 0, 1}

    Args:
        env_id: Atari environment ID (e.g., 'SpaceInvadersNoFrameskip-v4')
        n_envs: Number of parallel environments
        seed: Random seed
        state: Ignored for Atari (used by retro)
        use_subproc: Use SubprocVecEnv instead of DummyVecEnv
        wrapper_kwargs: Additional wrapper configuration
        **kwargs: Additional arguments (ignored)

    Returns:
        Preprocessed VecEnv
    """
    env_id = resolve_atari_env_id(env_id)
    _ensure_atari_registered(env_id)

    # Select vectorization class
    vec_env_cls = SubprocVecEnv if use_subproc else DummyVecEnv

    # Default wrapper kwargs
    default_wrapper_kwargs = {
        "noop_max": 30,
        "frame_skip": 4,
        "screen_size": 84,
        "terminal_on_life_loss": True,
        "clip_reward": True,
    }

    # Merge with user-provided kwargs, filtering out retro-specific keys
    if wrapper_kwargs:
        _atari_allowed = {"noop_max", "frame_skip", "screen_size", "terminal_on_life_loss", "clip_reward"}
        filtered = {k: v for k, v in wrapper_kwargs.items() if k in _atari_allowed}
        default_wrapper_kwargs.update(filtered)

    # Use a callable env constructor so each SubprocVecEnv worker imports/registers Atari envs.
    def env_ctor(**env_kwargs: Any) -> gym.Env:
        _register_ale_envs(env_id)
        # SB3 expects `rgb_array` by default for video recording.
        kwargs = {"render_mode": "rgb_array"}
        kwargs.update(env_kwargs)
        try:
            return gym.make(env_id, **kwargs)
        except TypeError:
            # Some envs don't accept render_mode
            return gym.make(env_id, **env_kwargs)

    try:
        vec_env = make_vec_env(
            env_ctor,
            n_envs=n_envs,
            seed=seed,
            vec_env_cls=vec_env_cls,
            wrapper_class=AtariWrapper,
            wrapper_kwargs=default_wrapper_kwargs,
        )
    except PermissionError:
        # If `forkserver` is not permitted, fall back to `fork` (Linux) or `spawn`.
        vec_env_kwargs: dict[str, Any]
        try:
            vec_env_kwargs = {"start_method": "fork"}
            vec_env = make_vec_env(
                env_ctor,
                n_envs=n_envs,
                seed=seed,
                vec_env_cls=vec_env_cls,
                vec_env_kwargs=vec_env_kwargs,
                wrapper_class=AtariWrapper,
                wrapper_kwargs=default_wrapper_kwargs,
            )
        except Exception:
            vec_env_kwargs = {"start_method": "spawn"}
            vec_env = make_vec_env(
                env_ctor,
                n_envs=n_envs,
                seed=seed,
                vec_env_cls=vec_env_cls,
                vec_env_kwargs=vec_env_kwargs,
                wrapper_class=AtariWrapper,
                wrapper_kwargs=default_wrapper_kwargs,
            )

    return vec_env
