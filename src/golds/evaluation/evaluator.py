"""Model evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from golds.results.schema import EvalResult

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
        n_episodes: int = 30,
        deterministic: bool = True,
        render: bool = False,
        verbose: bool = True,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate model over multiple episodes.

        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy
            render: Render environment (not implemented)
            verbose: Print progress
            seed: Optional seed for reproducible evaluation

        Returns:
            Dictionary with evaluation metrics including per-episode data
        """
        env = self._create_env(render=render)
        model = PPO.load(self.model_path)

        if seed is not None:
            env.seed(seed)

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

                episode_rewards.append(float(episode_reward))
                episode_lengths.append(int(episode_length))

                if verbose:
                    console.print(
                        f"Episode {episode + 1}/{n_episodes}: "
                        f"Reward = {episode_reward:.2f}, "
                        f"Length = {episode_length}"
                    )

        finally:
            env.close()

        rewards_arr = np.array(episode_rewards)
        lengths_arr = np.array(episode_lengths)

        results: dict[str, Any] = {
            "mean_reward": float(np.mean(rewards_arr)),
            "std_reward": float(np.std(rewards_arr)),
            "min_reward": float(np.min(rewards_arr)),
            "max_reward": float(np.max(rewards_arr)),
            "median_reward": float(np.median(rewards_arr)),
            "mean_length": float(np.mean(lengths_arr)),
            "std_length": float(np.std(lengths_arr)),
            "n_episodes": n_episodes,
            "deterministic": deterministic,
            "seed": seed,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }

        return results

    def benchmark(
        self,
        n_episodes: int = 100,
        seeds: list[int] | None = None,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        """Run standardized benchmark: n_episodes per seed, aggregate results.

        Default: 100 episodes x seeds [42, 123, 456].
        Returns dict with aggregated stats including CI95.

        Args:
            n_episodes: Number of episodes per seed
            seeds: List of seeds to evaluate across
            deterministic: Use deterministic policy

        Returns:
            Dictionary with aggregated benchmark results
        """
        if seeds is None:
            seeds = [42, 123, 456]

        all_rewards: list[float] = []
        all_lengths: list[int] = []
        seed_results: list[dict[str, Any]] = []

        for seed in seeds:
            result = self.evaluate(
                n_episodes=n_episodes,
                deterministic=deterministic,
                seed=seed,
            )
            all_rewards.extend(result["episode_rewards"])
            all_lengths.extend(result["episode_lengths"])
            seed_results.append(result)

        # Aggregate across all seeds
        rewards_arr = np.array(all_rewards)
        lengths_arr = np.array(all_lengths)

        # 95% confidence interval
        n = len(rewards_arr)
        ci95 = 1.96 * np.std(rewards_arr) / np.sqrt(n) if n > 1 else 0.0

        return {
            "mean_reward": float(np.mean(rewards_arr)),
            "std_reward": float(np.std(rewards_arr)),
            "min_reward": float(np.min(rewards_arr)),
            "max_reward": float(np.max(rewards_arr)),
            "median_reward": float(np.median(rewards_arr)),
            "ci95": float(ci95),
            "mean_length": float(np.mean(lengths_arr)),
            "std_length": float(np.std(lengths_arr)),
            "n_episodes": n,
            "n_seeds": len(seeds),
            "seeds": seeds,
            "deterministic": deterministic,
            "per_seed_results": seed_results,
        }

    def export_results(self, results: dict, path: str | Path) -> None:
        """Export evaluation results as JSON.

        Args:
            results: Evaluation or benchmark results dictionary
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    def to_eval_result(self, results: dict[str, Any], seed: int | None = None) -> EvalResult:
        """Convert results dict to an EvalResult schema object.

        Args:
            results: Evaluation results dictionary
            seed: Optional seed override (defaults to seed in results)

        Returns:
            EvalResult schema object
        """
        from golds.results.schema import EvalResult

        return EvalResult(
            mean_reward=results["mean_reward"],
            std_reward=results["std_reward"],
            min_reward=results["min_reward"],
            max_reward=results["max_reward"],
            median_reward=results.get("median_reward", 0.0),
            mean_length=results["mean_length"],
            std_length=results["std_length"],
            n_episodes=results["n_episodes"],
            deterministic=results.get("deterministic", True),
            seed=seed if seed is not None else results.get("seed"),
        )

    def print_results(self, results: dict[str, Any]) -> None:
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
        if "median_reward" in results:
            table.add_row("Median Reward", f"{results['median_reward']:.2f}")
        if "ci95" in results:
            table.add_row("CI 95%", f"\u00b1{results['ci95']:.2f}")
        table.add_row("Mean Length", f"{results['mean_length']:.0f}")
        table.add_row("Episodes", str(results["n_episodes"]))
        if "n_seeds" in results:
            table.add_row("Seeds", str(results["n_seeds"]))

        console.print(table)


def quick_evaluate(
    model_path: Path | str,
    game_id: str,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> dict[str, Any]:
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
