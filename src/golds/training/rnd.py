"""Random Network Distillation (RND) intrinsic reward module."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnvWrapper


class RNDNetwork(nn.Module):
    """Small CNN mapping (1, 84, 84) grayscale frames to 512-dim embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class RunningMeanStd:
    """Welford online mean/variance tracker."""

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count: float = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: float,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count


class RNDRewardWrapper(VecEnvWrapper):
    """VecEnv wrapper that adds RND intrinsic reward inline during step_wait()."""

    def __init__(
        self,
        venv: VecEnvWrapper,
        scale: float = 1.0,
        learning_rate: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        super().__init__(venv)
        self.scale = scale
        self.device = torch.device(device)

        self.target = RNDNetwork().to(self.device)
        self.predictor = RNDNetwork().to(self.device)

        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=learning_rate
        )

        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=(1, 84, 84))

    def _extract_frame(self, obs: np.ndarray) -> np.ndarray:
        """Extract a single grayscale frame from potentially stacked observations."""
        if obs.ndim == 4 and obs.shape[1] > 1:
            # Stacked frames: take the last channel as a single frame
            return obs[:, -1:, :, :]
        if obs.ndim == 4 and obs.shape[1] == 1:
            return obs
        # HWC format
        if obs.ndim == 4 and obs.shape[3] >= 1:
            return obs[:, :, :, -1:]
        return obs

    def _normalize_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Normalize observations using running statistics."""
        self.obs_rms.update(obs)
        normalized = (obs - self.obs_rms.mean) / (
            np.sqrt(self.obs_rms.var) + 1e-8
        )
        normalized = np.clip(normalized, -5.0, 5.0)
        return torch.as_tensor(normalized, dtype=torch.float32, device=self.device)

    def _compute_intrinsic_reward(self, obs: np.ndarray) -> np.ndarray:
        """Compute RND intrinsic reward from observations."""
        frame = self._extract_frame(obs)
        normalized = self._normalize_obs(frame)

        with torch.no_grad():
            target_features = self.target(normalized)
        predictor_features = self.predictor(normalized)

        # MSE per sample
        intrinsic_reward = (
            (target_features - predictor_features).pow(2).mean(dim=1)
        )

        # Train predictor
        loss = intrinsic_reward.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        intrinsic_np = intrinsic_reward.detach().cpu().numpy()

        # Normalize intrinsic reward
        self.reward_rms.update(intrinsic_np.reshape(-1, 1))
        intrinsic_np = intrinsic_np / (np.sqrt(self.reward_rms.var.item()) + 1e-8)

        return intrinsic_np

    def step_wait(self) -> tuple:
        obs, rewards, dones, infos = self.venv.step_wait()
        intrinsic = self._compute_intrinsic_reward(obs)
        rewards = rewards + self.scale * intrinsic
        for i, info in enumerate(infos):
            info["rnd_intrinsic_reward"] = intrinsic[i]
        return obs, rewards, dones, infos

    def reset(self) -> np.ndarray:
        return self.venv.reset()
