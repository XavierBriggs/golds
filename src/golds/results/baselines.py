"""Published human/random/PPO baseline scores for standard games."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaselineScores:
    """Published scores for a game."""

    human: float
    random: float
    dqn: float | None = None
    ppo: float | None = None


# Sources: Mnih et al. 2015 (DQN Nature), Schulman et al. 2017 (PPO), Machado et al. 2018
BASELINES: dict[str, BaselineScores] = {
    "space_invaders": BaselineScores(human=1668.7, random=148.0, dqn=1976.0, ppo=942.0),
    "breakout": BaselineScores(human=30.5, random=1.7, dqn=401.2, ppo=274.8),
    "pong": BaselineScores(human=14.6, random=-20.7, dqn=18.9, ppo=20.7),
    "qbert": BaselineScores(human=13455.0, random=163.9, dqn=10596.0, ppo=14293.3),
    "seaquest": BaselineScores(human=42054.7, random=68.4, dqn=5286.0, ppo=1204.5),
    "asteroids": BaselineScores(human=47388.7, random=719.1, dqn=1629.0, ppo=2097.5),
    "ms_pacman": BaselineScores(human=6951.6, random=307.3, dqn=2311.0, ppo=1691.8),
    "montezuma_revenge": BaselineScores(human=4753.3, random=0.0, dqn=0.0, ppo=0.0),
    "enduro": BaselineScores(human=860.5, random=0.0, dqn=301.8, ppo=1064.0),
    "frostbite": BaselineScores(human=4334.7, random=65.2, dqn=328.3, ppo=314.2),
}


def human_normalized_score(game_id: str, agent_score: float) -> float | None:
    """Compute human-normalized score: (agent - random) / (human - random).

    Returns None if the game has no baselines or human == random.
    """
    baseline = BASELINES.get(game_id)
    if baseline is None:
        return None
    denominator = baseline.human - baseline.random
    if denominator == 0:
        return None
    return (agent_score - baseline.random) / denominator
