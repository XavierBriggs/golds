"""Retro environment maker for NES/SNES games using stable-retro."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import gymnasium as gym
import numpy as np

try:
    import retro

    RETRO_AVAILABLE = True
except ImportError:
    RETRO_AVAILABLE = False

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


class RetroPreprocessing(gym.Wrapper):
    """Preprocessing wrapper for retro environments.

    Applies DeepMind-style preprocessing:
    - Grayscale conversion
    - Resize to 84x84
    - Reward clipping (optional)
    """

    def __init__(
        self,
        env: gym.Env,
        screen_size: int = 84,
        grayscale: bool = True,
        clip_reward: bool = True,
    ) -> None:
        """Initialize preprocessing wrapper.

        Args:
            env: Environment to wrap
            screen_size: Size to resize frames to
            grayscale: Convert to grayscale
            clip_reward: Clip rewards to {-1, 0, 1}
        """
        super().__init__(env)
        self.screen_size = screen_size
        self.grayscale = grayscale
        self.clip_reward = clip_reward

        # Update observation space
        if grayscale:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(screen_size, screen_size, 1),
                dtype=np.uint8,
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(screen_size, screen_size, 3),
                dtype=np.uint8,
            )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Process observation."""
        # Convert to grayscale
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # Resize
        obs = cv2.resize(obs, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)

        # Add channel dimension for grayscale
        if self.grayscale:
            obs = obs[:, :, np.newaxis]

        return obs

    def step(self, action: Any) -> tuple:
        """Step with reward clipping."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.observation(obs)

        if self.clip_reward:
            reward = np.sign(reward)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> tuple:
        """Reset with observation processing."""
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class FrameSkip(gym.Wrapper):
    """Frame skipping wrapper that takes max over last frames."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        """Initialize frame skip wrapper.

        Args:
            env: Environment to wrap
            skip: Number of frames to skip
        """
        super().__init__(env)
        self.skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action: Any) -> tuple:
        """Step with frame skipping."""
        total_reward = 0.0
        terminated = truncated = False

        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            # Store last two frames for max pooling
            if i == self.skip - 2:
                self._obs_buffer[0] = obs
            if i == self.skip - 1:
                self._obs_buffer[1] = obs

            if terminated or truncated:
                break

        # Max pool over last 2 frames
        obs = self._obs_buffer.max(axis=0)
        return obs, total_reward, terminated, truncated, info


def make_retro_env(
    game: str,
    state: str | None = None,
    screen_size: int = 84,
    grayscale: bool = True,
    clip_reward: bool = True,
    frame_skip: int = 4,
    players: int = 1,
    opponent_mode: str = "none",
    opponent_model_path: str | None = None,
    opponent_snapshot_dir: str | None = None,
    opponent_reload_interval_steps: int = 500,
) -> gym.Env:
    """Create a single retro environment with preprocessing.

    Args:
        game: Game ID (e.g., 'SuperMarioBros-Nes')
        state: Initial state (e.g., 'Level1-1')
        screen_size: Size to resize frames to
        grayscale: Convert to grayscale
        clip_reward: Clip rewards
        frame_skip: Number of frames to skip

    Returns:
        Preprocessed gym.Env
    """
    if not RETRO_AVAILABLE:
        raise ImportError("stable-retro is not installed. Install with: pip install stable-retro")

    # Create base environment
    env = retro.make(
        game=game,
        state=state if state is not None else retro.State.DEFAULT,
        use_restricted_actions=retro.Actions.FILTERED,
        players=players,
        # Avoid OpenGL/pyglet viewer requirements during headless training.
        render_mode="rgb_array",
    )

    # Add Monitor so SB3 evaluation uses true episode returns/lengths via info["episode"].
    env = Monitor(env)

    # Apply frame skipping
    if frame_skip > 1:
        env = FrameSkip(env, skip=frame_skip)

    # Apply preprocessing
    env = RetroPreprocessing(
        env,
        screen_size=screen_size,
        grayscale=grayscale,
        clip_reward=clip_reward,
    )

    return env


def make_retro_vec_env(
    env_id: str,
    n_envs: int = 8,
    seed: int | None = None,
    state: str | None = None,
    use_subproc: bool = True,
    players: int = 1,
    opponent_mode: str = "none",
    opponent_model_path: str | None = None,
    opponent_snapshot_dir: str | None = None,
    opponent_reload_interval_steps: int = 500,
    wrapper_kwargs: dict | None = None,
    **kwargs,
) -> VecEnv:
    """Create a vectorized retro environment.

    Args:
        env_id: Retro game ID (e.g., 'SuperMarioBros-Nes')
        n_envs: Number of parallel environments
        seed: Random seed
        state: Initial state
        use_subproc: Use SubprocVecEnv instead of DummyVecEnv
        wrapper_kwargs: Preprocessing configuration
        **kwargs: Additional arguments (ignored)

    Returns:
        Vectorized retro environment
    """
    if not RETRO_AVAILABLE:
        raise ImportError("stable-retro is not installed. Install with: pip install stable-retro")

    # Default wrapper kwargs
    default_kwargs = {
        "screen_size": 84,
        "grayscale": True,
        "clip_reward": True,
        "frame_skip": 4,
    }

    if wrapper_kwargs:
        # Retro preprocessing does not support Atari-specific options like
        # `terminal_on_life_loss`; ignore unknown keys.
        allowed = {"screen_size", "grayscale", "clip_reward", "frame_skip"}
        filtered = {k: v for k, v in wrapper_kwargs.items() if k in allowed}
        default_kwargs.update(filtered)

    def make_env(rank: int) -> Callable[[], gym.Env]:
        """Create env factory function for vectorization."""

        def _init() -> gym.Env:
            env = make_retro_env(
                game=env_id,
                state=state,
                **default_kwargs,
                players=players,
                opponent_mode=opponent_mode,
                opponent_model_path=opponent_model_path,
                opponent_snapshot_dir=opponent_snapshot_dir,
                opponent_reload_interval_steps=opponent_reload_interval_steps,
            )
            if seed is not None:
                env.reset(seed=seed + rank)
            return env

        return _init

    # Create vectorized environment
    env_fns = [make_env(i) for i in range(n_envs)]

    if use_subproc:
        try:
            vec_env = SubprocVecEnv(env_fns)
        except PermissionError:
            # Some environments disallow `forkserver` (SB3's default when available).
            # `fork` works on Linux without requiring an importable `__main__` file
            # (unlike `spawn`).
            try:
                vec_env = SubprocVecEnv(env_fns, start_method="fork")
            except Exception:
                vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)

    return vec_env
