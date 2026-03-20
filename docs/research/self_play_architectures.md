# Competitive Self-Play Architectures for Reinforcement Learning

*Research notes for single-GPU training with a snapshot pool, targeting fighting games (Mortal Kombat, Street Fighter).*

---

## 1. AlphaStar League Training Architecture

DeepMind's AlphaStar (Vinyals et al., 2019) introduced the **league training** framework to handle StarCraft II's enormous strategy space. The league maintains three agent roles:

- **Main agents** — trained to beat all opponents in the league, optimizing for robust generalist play.
- **Main exploiters** — trained specifically to find weaknesses in the current main agents, then periodically reset.
- **League exploiters** — trained to beat the entire population, ensuring no historical strategy is forgotten.

Each agent selects opponents via a matchmaking mechanism (see PFSP below). Snapshots of agents are frozen into the league at regular intervals, creating a growing pool of diverse opponents. The key insight is that **role asymmetry** prevents strategy collapse: exploiters force the main agents to patch weaknesses, while main agents pursue breadth.

**Single-GPU adaptation:** Run one agent at a time but maintain the role concept. Alternate training phases: train the main policy for N steps, then switch to an exploiter phase that specifically targets the main policy's weaknesses. Freeze snapshots into the pool after each phase.

*Reference: [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://www.nature.com/articles/s41586-019-1724-z) (Nature, 2019)*

## 2. OpenAI Five Opponent Sampling Strategy

OpenAI Five (OpenAI, 2019) used a simpler self-play scheme for Dota 2. The training opponent was sampled from:

- **80% of the time:** the most recent version of the agent (latest snapshot).
- **20% of the time:** a uniformly random past version from the snapshot pool.

This mixture balances two goals: staying competitive against the current frontier (preventing catastrophic forgetting of recent skills) while maintaining robustness against older strategies. The 80/20 split is a practical starting point; tuning it is game-dependent.

*Reference: [OpenAI Five (blog post)](https://openai.com/research/openai-five) (2019)*

## 3. Elo-Based Opponent Selection / Prioritized Fictitious Self-Play (PFSP)

AlphaStar formalized opponent selection with **Prioritized Fictitious Self-Play**. For a learning agent with index `i`, the probability of selecting opponent `j` from the pool is:

```
P(j) = f(P_beat(i, j)) / sum_k f(P_beat(i, k))
```

Where `P_beat(i, j)` is the estimated win rate of agent `i` against opponent `j`, and `f` is a weighting function. Common choices for `f`:

| Weighting function | Formula | Effect |
|---|---|---|
| **Hard** | `f(x) = (1-x)^p` where `p > 1` | Focuses on opponents the agent loses to most |
| **Variance** | `f(x) = x * (1-x)` | Prioritizes opponents near 50% win rate (maximum learning signal) |
| **Uniform** | `f(x) = 1` | Equal probability for all opponents |

The **variance weighting** `f(x) = x(1-x)` is recommended for single-GPU setups because it naturally avoids wasting compute on opponents that are either too easy (win rate near 1.0) or too hard (win rate near 0.0).

**Implementation:** Maintain a win-rate matrix `W[i][j]` updated with exponential moving average after each evaluation episode. Recompute opponent sampling probabilities every K training steps.

*Reference: [Vinyals et al., 2019](https://www.nature.com/articles/s41586-019-1724-z); see Supplementary Methods for PFSP details.*

## 4. Population-Based Training (PBT) Integration with Self-Play

Jaderberg et al. (2017, 2019) showed that **Population-Based Training** can co-evolve hyperparameters alongside self-play. In PBT:

1. Train a population of agents in parallel, each with different hyperparameters (learning rate, entropy coefficient, discount factor, etc.).
2. Periodically evaluate agents against each other.
3. **Exploit:** replace underperforming agents' weights with copies of top performers.
4. **Explore:** perturb the copied hyperparameters randomly.

For fighting games on a single GPU, full PBT is expensive. A practical alternative is **sequential PBT**: train 2-4 agents in round-robin, evaluating after each phase, and copying weights/hyperparameters from the best performer. Even with just 2 agents, this provides hyperparameter adaptation that pure self-play lacks.

Key hyperparameters to evolve for PPO self-play: learning rate (3e-4 to 1e-3), entropy coefficient (0.001 to 0.05), GAE lambda (0.9 to 0.99), and reward shaping weights.

*Reference: [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846) (Jaderberg et al., 2017)*

## 5. Simple Self-Play Baselines

Before implementing complex schemes, establish baselines:

- **Latest-snapshot self-play:** Always train against the most recent frozen copy of the agent. Simple but prone to strategy cycling (A beats B, B beats C, C beats A) and catastrophic forgetting.
- **Uniform snapshot pool:** Maintain a buffer of past snapshots (e.g., every 100K steps) and sample opponents uniformly. More stable than latest-snapshot but wastes compute on obsolete opponents.
- **Window sampling:** Sample uniformly from only the most recent W snapshots. A practical middle ground; W=10-20 works well for fighting games.

**Snapshot frequency matters.** Too frequent (every 1K steps) creates near-identical opponents that provide little diversity. Too infrequent (every 1M steps) misses important intermediate strategies. For fighting games with PPO, snapshotting every 50K-200K environment steps is a reasonable range.

## 6. Practical Implementation for Fighting Games with PPO

For Mortal Kombat / Street Fighter with PPO on a single GPU:

**Recommended architecture:**
1. **Snapshot pool** of 20-50 past policies, stored as model weight files.
2. **Opponent sampling:** Start with the OpenAI Five 80/20 split (latest/uniform). Graduate to PFSP with variance weighting once you have 10+ snapshots.
3. **PPO settings:** Clip ratio 0.2, entropy coefficient 0.01 (anneal to 0.001), mini-batch size 256-512, rollout length 128-256 steps.
4. **Reward shaping:** Combine sparse (round win/loss) with dense signals (damage dealt, health differential). Gradually anneal dense rewards toward zero to avoid reward hacking.

**Training loop (single-GPU):**
```
for each training iteration:
    sample opponent from snapshot pool (PFSP or 80/20)
    load opponent weights (frozen, no grad)
    collect rollouts: agent vs opponent for N steps
    update agent with PPO
    every S steps: freeze agent snapshot into pool
    every E steps: run evaluation tournament, update win-rate matrix
```

Fighting games have short episodes (rounds last 30-90 seconds at typical frame-skip), so rollout collection is fast. Budget 10-20M environment frames for initial training; self-play benefits compound over longer runs (50-100M frames).

*Reference: [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017)*

## 7. Key Metrics to Track

| Metric | What it reveals | How to compute |
|---|---|---|
| **Elo rating** | Overall skill progression | Run round-robin tournaments every E steps; update Elo with K=32 |
| **Win rate vs. latest** | Whether training is making progress | Evaluate current agent vs. the previous snapshot |
| **Win rate matrix** | Strategy diversity and cycling | Full pairwise evaluation of top-N pool members |
| **Exploitability proxy** | Robustness | Win rate of a fresh exploiter trained for M steps against the agent |
| **Strategy entropy** | Diversity of play | Track action distribution entropy during evaluation rollouts |
| **Elo volatility** | Training stability | Standard deviation of Elo over a rolling window |

**Detecting strategy cycling:** If the win-rate matrix shows a strong non-transitive pattern (agent at step 300K beats agent at 200K, which beats agent at 100K, which beats 300K), switch from latest-snapshot to PFSP sampling.

**Elo implementation:** Use the standard formula: `E_a = 1 / (1 + 10^((R_b - R_a)/400))`, update with `R_a' = R_a + K * (S_a - E_a)` where `S_a` is the actual result (1/0.5/0). Start all snapshots at Elo 1000. Run at least 30 games per pair for stable estimates.

---

## Summary: Recommended Progression for Single-GPU Setup

1. **Start simple:** Latest-snapshot self-play with PPO. Validate the environment and reward signal.
2. **Add a snapshot pool:** Window sampling (W=20) with snapshots every 100K steps.
3. **Upgrade to PFSP:** Once pool has 10+ members, switch to variance-weighted opponent selection.
4. **Track Elo and win-rate matrix:** Detect and respond to strategy cycling.
5. **Optional: Add exploiter phases** inspired by AlphaStar league roles.

## Key References

- Vinyals, O. et al. "Grandmaster level in StarCraft II using multi-agent reinforcement learning." *Nature* 575 (2019): 350-354. https://www.nature.com/articles/s41586-019-1724-z
- OpenAI. "OpenAI Five." 2019. https://openai.com/research/openai-five
- Jaderberg, M. et al. "Population Based Training of Neural Networks." *arXiv:1711.09846* (2017). https://arxiv.org/abs/1711.09846
- Schulman, J. et al. "Proximal Policy Optimization Algorithms." *arXiv:1707.06347* (2017). https://arxiv.org/abs/1707.06347
- Lanctot, M. et al. "OpenSpiel: A Framework for Reinforcement Learning in Games." *arXiv:1908.09453* (2019). https://arxiv.org/abs/1908.09453
- Bansal, T. et al. "Emergent Complexity via Multi-Agent Competition." *ICLR 2018*. https://arxiv.org/abs/1710.03748
