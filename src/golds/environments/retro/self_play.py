"""Self-play / opponent wrappers for stable-retro 2-player environments.

Important: the opponent policy must see the same observation format as the learner.
In this project, observations are stacked/transposed at the VecEnv level, so the
primary wrapper is `VecTwoPlayerOpponentWrapper`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from zipfile import BadZipFile, ZipFile

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

OpponentMode = Literal["none", "random", "noop", "model", "self_play"]


def _is_valid_sb3_zip(path: Path) -> bool:
    try:
        with ZipFile(path, "r") as zf:
            _ = zf.read("data")
        return True
    except (BadZipFile, KeyError, OSError):
        return False


def _latest_snapshot(snapshot_dir: Path) -> Path | None:
    if not (snapshot_dir.exists() and snapshot_dir.is_dir()):
        return None
    # Expect names like opponent_<timesteps>.zip, but fall back to mtime ordering.
    zips = sorted(snapshot_dir.glob("*.zip"))
    if not zips:
        return None

    def key(p: Path) -> tuple[int, float]:
        stem = p.stem
        # opponent_1234567
        try:
            if "_" in stem:
                maybe = stem.rsplit("_", 1)[-1]
                return (int(maybe), p.stat().st_mtime)
        except Exception:
            pass
        return (0, p.stat().st_mtime)

    for p in sorted(zips, key=key, reverse=True):
        if _is_valid_sb3_zip(p):
            return p
    return None


@dataclass
class OpponentSpec:
    mode: OpponentMode = "none"
    model_path: Path | None = None
    snapshot_dir: Path | None = None
    reload_interval_steps: int = 500  # env steps


class TwoPlayerOpponentWrapper(gym.Wrapper):
    """Wrap a 2-player retro env to expose only Player-1 actions.

    The wrapped env must have a MultiBinary action space whose size is divisible by 2.
    """

    def __init__(self, env: gym.Env, *, opponent: OpponentSpec) -> None:
        super().__init__(env)
        self.opponent = opponent

        if not isinstance(env.action_space, gym.spaces.MultiBinary):
            raise TypeError(
                f"Expected MultiBinary action_space for 2-player env, got: {env.action_space}"
            )
        total = int(env.action_space.n)
        if total % 2 != 0:
            raise ValueError(f"2-player action space size must be even, got: {total}")

        self._buttons_per_player = total // 2
        self.action_space = gym.spaces.MultiBinary(self._buttons_per_player)

        self._last_obs: Any = None
        self._step_count = 0
        self._opponent_model = None
        self._opponent_model_path_loaded: Path | None = None

        if self.opponent.mode == "model" and self.opponent.model_path:
            self._load_opponent_model(self.opponent.model_path)

    def reset(self, **kwargs) -> tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._step_count = 0
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        p1 = np.asarray(action, dtype=np.int8).reshape(-1)
        if p1.shape[0] != self._buttons_per_player:
            raise ValueError(
                f"Expected P1 action length {self._buttons_per_player}, got {p1.shape[0]}"
            )

        p2 = self._opponent_action()
        combined = np.concatenate([p1, p2], axis=0)

        obs, reward, terminated, truncated, info = self.env.step(combined)
        self._last_obs = obs
        self._step_count += 1
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _maybe_reload_snapshot(self) -> None:
        if self.opponent.mode != "self_play":
            return
        if self.opponent.snapshot_dir is None:
            return
        if self._step_count % max(1, int(self.opponent.reload_interval_steps)) != 0:
            return
        latest = _latest_snapshot(self.opponent.snapshot_dir)
        if latest is None:
            return
        if self._opponent_model_path_loaded == latest:
            return
        self._load_opponent_model(latest)

    def _load_opponent_model(self, path: Path) -> None:
        # Lazy import: stable_baselines3 is heavy and this code may run in workers.
        from stable_baselines3 import PPO

        if not path.exists():
            return
        try:
            self._opponent_model = PPO.load(path, device="cpu")
            self._opponent_model_path_loaded = path
        except Exception:
            # Ignore bad/corrupt snapshots and keep previous opponent.
            return

    def _opponent_action(self) -> np.ndarray:
        self._maybe_reload_snapshot()

        mode = self.opponent.mode
        if mode == "none" or mode == "noop":
            return np.zeros((self._buttons_per_player,), dtype=np.int8)
        if mode == "random":
            # Random 0/1 per button.
            return np.random.randint(0, 2, size=(self._buttons_per_player,), dtype=np.int8)

        # Model-based opponent
        model = self._opponent_model
        if model is None:
            # Fall back to noop if model not loaded.
            return np.zeros((self._buttons_per_player,), dtype=np.int8)

        obs = self._last_obs
        if obs is None:
            return np.zeros((self._buttons_per_player,), dtype=np.int8)

        act, _ = model.predict(obs, deterministic=True)
        return np.asarray(act, dtype=np.int8).reshape(self._buttons_per_player)


class VecTwoPlayerOpponentWrapper(VecEnvWrapper):
    """VecEnv wrapper for 2-player retro envs that exposes only Player-1 actions.

    The underlying VecEnv must have MultiBinary action space with size divisible by 2.
    """

    def __init__(self, venv: VecEnv, *, opponent: OpponentSpec):
        super().__init__(venv)
        self.opponent = opponent

        if not isinstance(venv.action_space, gym.spaces.MultiBinary):
            raise TypeError(
                f"Expected MultiBinary action_space for 2-player env, got: {venv.action_space}"
            )
        total = int(venv.action_space.n)
        if total % 2 != 0:
            raise ValueError(f"2-player action space size must be even, got: {total}")

        self._buttons_per_player = total // 2
        self.action_space = gym.spaces.MultiBinary(self._buttons_per_player)

        self._last_obs: Any = None
        self._step_count = 0
        self._opponent_model = None
        self._opponent_model_path_loaded: Path | None = None

        if self.opponent.mode == "model" and self.opponent.model_path:
            self._load_opponent_model(self.opponent.model_path)

    def reset(self) -> Any:
        obs = self.venv.reset()
        self._last_obs = obs
        self._step_count = 0
        return obs

    def step_async(self, actions: Any) -> None:
        p1 = np.asarray(actions, dtype=np.int8)
        if p1.ndim == 1:
            p1 = p1.reshape((self.num_envs, -1))
        if p1.shape[1] != self._buttons_per_player:
            raise ValueError(
                f"Expected P1 action width {self._buttons_per_player}, got {p1.shape[1]}"
            )

        p2 = self._opponent_action(num_envs=self.num_envs)
        combined = np.concatenate([p1, p2], axis=1)
        self.venv.step_async(combined)

    def step_wait(self) -> tuple[Any, np.ndarray, np.ndarray, list[dict]]:
        obs, rewards, dones, infos = self.venv.step_wait()
        self._last_obs = obs
        self._step_count += 1
        return obs, rewards, dones, infos

    def _maybe_reload_snapshot(self) -> None:
        if self.opponent.mode != "self_play":
            return
        if self.opponent.snapshot_dir is None:
            return
        if self._step_count % max(1, int(self.opponent.reload_interval_steps)) != 0:
            return
        latest = _latest_snapshot(self.opponent.snapshot_dir)
        if latest is None:
            return
        if self._opponent_model_path_loaded == latest:
            return
        self._load_opponent_model(latest)

    def _load_opponent_model(self, path: Path) -> None:
        from stable_baselines3 import PPO

        if not path.exists():
            return
        try:
            self._opponent_model = PPO.load(path, device="cpu")
            self._opponent_model_path_loaded = path
        except Exception:
            return

    def _opponent_action(self, *, num_envs: int) -> np.ndarray:
        self._maybe_reload_snapshot()

        mode = self.opponent.mode
        if mode == "none" or mode == "noop":
            return np.zeros((num_envs, self._buttons_per_player), dtype=np.int8)
        if mode == "random":
            return np.random.randint(0, 2, size=(num_envs, self._buttons_per_player), dtype=np.int8)

        model = self._opponent_model
        if model is None:
            return np.zeros((num_envs, self._buttons_per_player), dtype=np.int8)

        obs = self._last_obs
        if obs is None:
            return np.zeros((num_envs, self._buttons_per_player), dtype=np.int8)

        act, _ = model.predict(obs, deterministic=True)
        act = np.asarray(act, dtype=np.int8)
        if act.ndim == 1:
            act = act.reshape((num_envs, -1))
        return act.reshape((num_envs, self._buttons_per_player))
