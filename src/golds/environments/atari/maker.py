"""Atari environment maker using gymnasium/ale-py with SB3 wrappers."""

from __future__ import annotations

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


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

    Uses SB3's make_atari_env which applies:
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

    # Merge with user-provided kwargs
    if wrapper_kwargs:
        default_wrapper_kwargs.update(wrapper_kwargs)

    # Create environment
    vec_env = make_atari_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_env_cls,
        wrapper_kwargs=default_wrapper_kwargs,
    )

    return vec_env


class AtariEnvironmentMaker:
    """Class-based Atari environment maker for compatibility."""

    # Supported Atari games with their environment IDs
    GAMES: dict[str, str] = {
        "space_invaders": "SpaceInvadersNoFrameskip-v4",
        "breakout": "BreakoutNoFrameskip-v4",
        "pong": "PongNoFrameskip-v4",
        "qbert": "QbertNoFrameskip-v4",
        "seaquest": "SeaquestNoFrameskip-v4",
        "asteroids": "AsteroidsNoFrameskip-v4",
        "ms_pacman": "MsPacmanNoFrameskip-v4",
        "enduro": "EnduroNoFrameskip-v4",
        "beam_rider": "BeamRiderNoFrameskip-v4",
        "freeway": "FreewayNoFrameskip-v4",
    }

    def make(
        self,
        game_id: str,
        n_envs: int = 8,
        seed: int | None = None,
        **kwargs,
    ) -> VecEnv:
        """Create Atari environment.

        Args:
            game_id: Game identifier or full env ID
            n_envs: Number of parallel environments
            seed: Random seed
            **kwargs: Additional arguments

        Returns:
            Preprocessed VecEnv
        """
        # Convert game_id to env_id if needed
        env_id = self.GAMES.get(game_id, game_id)
        return make_atari_vec_env(env_id=env_id, n_envs=n_envs, seed=seed, **kwargs)

    def supported_games(self) -> list[str]:
        """Return list of supported game IDs."""
        return list(self.GAMES.keys())
