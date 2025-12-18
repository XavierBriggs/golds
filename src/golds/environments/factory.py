"""Unified environment factory for creating preprocessed vectorized environments."""

from __future__ import annotations

import multiprocessing
import sys
from typing import Callable

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from golds.environments.registry import GameRegistry


class EnvironmentFactory:
    """Factory for creating preprocessed vectorized environments.

    Supports both Atari (via gymnasium/ale-py) and Retro (via stable-retro)
    environments with a unified interface.
    """

    _platform_makers: dict[str, Callable[..., VecEnv]] = {}

    @classmethod
    def register_platform(cls, platform: str, maker_fn: Callable[..., VecEnv]) -> None:
        """Register a maker function for a platform.

        Args:
            platform: Platform name ('atari' or 'retro')
            maker_fn: Function that creates a VecEnv
        """
        cls._platform_makers[platform] = maker_fn

    @classmethod
    def create(
        cls,
        game_id: str,
        n_envs: int = 8,
        frame_stack: int = 4,
        seed: int | None = None,
        state: str | None = None,
        use_subproc: bool = True,
        **kwargs,
    ) -> VecEnv:
        """Create a fully preprocessed vectorized environment.

        Args:
            game_id: Game identifier (e.g., 'space_invaders')
            n_envs: Number of parallel environments
            frame_stack: Number of frames to stack
            seed: Random seed
            use_subproc: Use SubprocVecEnv instead of DummyVecEnv
            **kwargs: Additional arguments passed to platform maker

        Returns:
            Preprocessed VecEnv ready for training
        """
        # Look up game in registry
        game = GameRegistry.get(game_id)

        # Get platform-specific maker
        if game.platform not in cls._platform_makers:
            raise ValueError(
                f"No maker registered for platform: {game.platform}. "
                f"Available: {list(cls._platform_makers.keys())}"
            )

        maker = cls._platform_makers[game.platform]

        # Set multiprocessing start method for Windows/WSL2 compatibility
        if use_subproc and sys.platform != "linux":
            try:
                multiprocessing.set_start_method("spawn", force=True)
            except RuntimeError:
                pass  # Already set

        effective_state = state if state is not None else game.default_state

        # Create base vectorized environment
        vec_env = maker(
            env_id=game.env_id,
            n_envs=n_envs,
            seed=seed,
            state=effective_state,
            use_subproc=use_subproc,
            **kwargs,
        )

        # Apply common wrappers
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
        vec_env = VecTransposeImage(vec_env)  # HWC -> CHW for PyTorch

        return vec_env

    @classmethod
    def create_eval_env(
        cls,
        game_id: str,
        frame_stack: int = 4,
        seed: int | None = None,
        state: str | None = None,
        **kwargs,
    ) -> VecEnv:
        """Create a single environment for evaluation.

        Args:
            game_id: Game identifier
            frame_stack: Number of frames to stack
            seed: Random seed
            **kwargs: Additional arguments

        Returns:
            Single preprocessed VecEnv for evaluation
        """
        # Evaluation uses single env without subprocesses
        kwargs["terminal_on_life_loss"] = False  # Don't terminate on life loss
        return cls.create(
            game_id=game_id,
            n_envs=1,
            frame_stack=frame_stack,
            seed=seed,
            state=state,
            use_subproc=False,  # DummyVecEnv for evaluation
            **kwargs,
        )


def _lazy_register_platforms() -> None:
    """Lazily register platform makers to avoid import cycles."""
    if not EnvironmentFactory._platform_makers:
        from golds.environments.atari.maker import make_atari_vec_env
        from golds.environments.retro.maker import make_retro_vec_env

        EnvironmentFactory.register_platform("atari", make_atari_vec_env)
        EnvironmentFactory.register_platform("retro", make_retro_vec_env)


# Patch create to lazily register
_original_create = EnvironmentFactory.create


@classmethod
def _create_with_lazy_register(cls, *args, **kwargs) -> VecEnv:
    _lazy_register_platforms()
    return _original_create.__func__(cls, *args, **kwargs)


EnvironmentFactory.create = _create_with_lazy_register
