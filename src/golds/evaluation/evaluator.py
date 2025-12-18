"""Model evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from golds.environments.factory import EnvironmentFactory

console = Console()


class Evaluator:
    """Model evaluation utility.

    Evaluates trained models on environments and computes metrics.
    """

    def __init__(
        self,
        model_path: Path | str,
        game_id: str,
        frame_stack: int = 4,
    ) -> None:
        """Initialize evaluator.

        Args:
            model_path: Path to trained model
            game_id: Game identifier
            frame_stack: Number of frames to stack
        """
        self.model_path = Path(model_path)
        self.game_id = game_id
        self.frame_stack = frame_stack

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

    def _create_env(self, render: bool = False) -> VecEnv:
        """Create evaluation environment.

        Args:
            render: Whether to render (not implemented for VecEnv)

        Returns:
            Evaluation environment
        """
        return EnvironmentFactory.create_eval_env(
            game_id=self.game_id,
            frame_stack=self.frame_stack,
        )

    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Evaluate model over multiple episodes.

        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy
            render: Render environment (not implemented)
            verbose: Print progress

        Returns:
            Dictionary with evaluation metrics
        """
        env = self._create_env(render=render)
        model = PPO.load(self.model_path)

        episode_rewards: list[float] = []
        episode_lengths: list[int] = []

        try:
            for episode in range(n_episodes):
                obs = env.reset()
                done = False
                episode_reward = 0.0
                episode_length = 0

                while not done:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, dones, infos = env.step(action)
                    episode_reward += reward[0]
                    episode_length += 1

                    # Check for episode end
                    if dones[0]:
                        done = True
                        # Get actual episode info if available
                        if "episode" in infos[0]:
                            episode_reward = infos[0]["episode"]["r"]
                            episode_length = infos[0]["episode"]["l"]

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                if verbose:
                    console.print(
                        f"Episode {episode + 1}/{n_episodes}: "
                        f"Reward = {episode_reward:.2f}, "
                        f"Length = {episode_length}"
                    )

        finally:
            env.close()

        results = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "std_length": float(np.std(episode_lengths)),
            "n_episodes": n_episodes,
        }

        return results

    def print_results(self, results: dict[str, float]) -> None:
        """Print evaluation results in a table.

        Args:
            results: Evaluation results dictionary
        """
        table = Table(title=f"Evaluation Results - {self.game_id}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Mean Reward", f"{results['mean_reward']:.2f}")
        table.add_row("Std Reward", f"{results['std_reward']:.2f}")
        table.add_row("Min Reward", f"{results['min_reward']:.2f}")
        table.add_row("Max Reward", f"{results['max_reward']:.2f}")
        table.add_row("Mean Length", f"{results['mean_length']:.0f}")
        table.add_row("Episodes", str(results["n_episodes"]))

        console.print(table)


def quick_evaluate(
    model_path: Path | str,
    game_id: str,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> dict[str, float]:
    """Quick evaluation function.

    Args:
        model_path: Path to trained model
        game_id: Game identifier
        n_episodes: Number of episodes
        deterministic: Use deterministic policy

    Returns:
        Evaluation results
    """
    evaluator = Evaluator(model_path, game_id)
    results = evaluator.evaluate(n_episodes=n_episodes, deterministic=deterministic)
    evaluator.print_results(results)
    return results
