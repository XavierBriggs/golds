# Reward Shaping Techniques for Game-Playing RL Agents

> Research survey for the GOLDS training system. Covers theory, pitfalls, and practical guidance for Atari, platformer, and fighting-game environments.

---

## 1. Potential-Based Reward Shaping (PBRS)

Ng, Harada, and Russell (ICML 1999) proved that **adding a shaping reward of the form**

```
F(s, s') = gamma * Phi(s') - Phi(s)
```

to the environment reward **preserves the set of optimal policies** under any MDP. `Phi(s)` is an arbitrary real-valued *potential function* over states, and `gamma` is the discount factor. The intuition: because the shaping telescopes across a trajectory, the total additional reward on any path from start to terminal is a constant that depends only on the start state, so relative policy rankings are unchanged.

**Practical use in games:** define `Phi(s)` as a domain heuristic (e.g., x-position in a platformer, health delta in a fighter) and add `F(s, s')` to the environment reward. This accelerates learning without introducing local optima that pure heuristic rewards can create. The key constraint is that `Phi` must depend only on the state, not on the action -- otherwise the policy-invariance guarantee breaks.

**Reference:** Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations: Theory and application to reward shaping.* ICML. [https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)

---

## 2. Reward Hacking Pitfalls in Retro Games

Using raw game score as the sole reward signal is deceptively dangerous:

- **Score loops:** Agents discover repeatable sub-sequences that yield small positive rewards indefinitely (e.g., collecting a respawning coin, toggling a bonus trigger) instead of progressing.
- **Pause-screen exploits:** Some retro ROMs increment score during pause menus or intro screens; agents learn to avoid gameplay entirely.
- **Timer abuse:** Games that award a time bonus on level completion can cause agents to suicide quickly rather than play well if the per-step penalty outweighs expected future reward.
- **Reward-irrelevant objectives:** In platformers, game score may reward killing enemies rather than finishing levels, leading to agents that camp in safe spots farming enemies.

**Mitigation:** Combine game score with auxiliary shaped rewards (see sections 1 and 5) and manually verify with recorded rollouts that agents are exhibiting intended behavior. OpenAI's *Faulty Reward Functions in the Wild* (Amodei et al., 2016) and follow-up work catalog dozens of such failure modes.

**Reference:** Amodei, D. et al. (2016). *Concrete Problems in AI Safety.* [https://arxiv.org/abs/1606.06565](https://arxiv.org/abs/1606.06565)

---

## 3. VecNormalize Running Reward Statistics (SB3)

`stable_baselines3.common.vec_env.VecNormalize` wraps a vectorized environment and maintains a **running mean and variance** of discounted returns (not per-step rewards). It normalizes each reward by dividing by the running standard deviation, producing roughly unit-variance reward signal.

Key implementation details:

- **`gamma` parameter** must match the algorithm's discount factor. It is used to compute discounted return estimates internally (`ret = ret * gamma + reward`), and a mismatch causes the normalization scale to be wrong.
- **`training` flag:** Set `vec_env.training = False` during evaluation so statistics are not polluted by evaluation episodes.
- **Serialization:** Save/load with `vec_env.save("vec_normalize.pkl")` and restore before inference; mismatched statistics produce garbage rewards.
- **When it helps:** Environments with highly variable reward magnitudes across episodes or games (our multi-game setting). It stabilizes PPO's value function learning.
- **When it hurts:** (a) Very short training runs where the running statistics have not converged, causing early instability. (b) Environments where reward sparsity is the main challenge -- normalization can shrink rare large rewards toward zero, making them harder to learn from. (c) When wrapping already-clipped rewards (values in {-1, 0, 1}), the normalization adds noise with no benefit.

**Reference:** SB3 documentation, [https://stable-baselines3.readthedocs.io/en/master/common/vec_env.html#vecnormalize](https://stable-baselines3.readthedocs.io/en/master/common/vec_env.html#vecnormalize)

---

## 4. ClipReward Wrapper vs Raw Rewards

The Atari standard (Mnih et al., 2015) clips all rewards to {-1, 0, +1} via `np.sign(reward)`. Our codebase applies this in both `AtariWrapper` and `RetroPreprocessing` by default.

| Aspect | ClipReward | Raw / Normalized |
|---|---|---|
| Stability | High -- bounded gradient magnitudes | Requires careful normalization |
| Information loss | Severe -- a +1000 event equals a +1 event | Preserved magnitude differences |
| Hyperparameter sensitivity | Low -- one less thing to tune | Higher -- learning rate, value loss scale interact with reward scale |
| Multi-game transfer | Good -- uniform scale across games | Requires per-game normalization |

**Tradeoff in practice:** Clipping is the safe default for Atari benchmarks and enables a single set of hyperparameters across games. However, for retro games where the reward structure encodes meaningful magnitudes (e.g., damage amounts in fighters, distance bonuses in racers), clipping destroys useful signal. In those cases, prefer `VecNormalize` or manual scaling.

**Reference:** Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature. [https://www.nature.com/articles/nature14236](https://www.nature.com/articles/nature14236)

---

## 5. Custom Reward Wrappers for Platformers

For platformers like Super Mario Bros, shaped rewards dramatically improve sample efficiency:

```python
class PlatformerRewardWrapper(gym.Wrapper):
    """Example: x-position delta + exploration + death penalty."""
    def __init__(self, env, x_weight=1.0, death_penalty=-50, explore_bonus=0.01):
        super().__init__(env)
        self._prev_x = 0
        self._visited = set()
        self.x_weight = x_weight
        self.death_penalty = death_penalty
        self.explore_bonus = explore_bonus

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        x_pos = info.get("x_pos", 0)
        shaped = self.x_weight * (x_pos - self._prev_x)  # rightward progress
        tile = x_pos // 16
        if tile not in self._visited:
            self._visited.add(tile)
            shaped += self.explore_bonus
        if term and info.get("life_lost", False):
            shaped += self.death_penalty
        self._prev_x = x_pos
        return obs, reward + shaped, term, trunc, info
```

**Design notes:**
- The **x-position delta** is a PBRS-compatible shaping reward (potential = x-position) and does not distort optimal policy.
- **Exploration bonuses** (count-based or tile-based) break PBRS guarantees but work well in practice for sparse-reward platformers.
- **Death penalties** should be moderate; too large and the agent becomes overly conservative, avoiding all enemies rather than learning to navigate past them.

---

## 6. Reward Scaling and Normalization Techniques

Beyond clipping and VecNormalize, other approaches include:

- **Linear scaling:** Divide all rewards by a fixed constant (e.g., max known reward). Simple, but requires domain knowledge.
- **PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets):** Adaptively normalizes value function targets. Used in IMPALA for multi-task Atari. Maintains running statistics in the value head and rescales weights when statistics update.
- **Symlog transform:** `sign(x) * log(1 + |x|)` compresses large rewards while preserving sign and relative ordering. Used in DreamerV3 for a single hyperparameter set across domains.
- **Percentile clipping:** Clip to the 1st/99th percentile of observed rewards rather than hard {-1, +1}. Preserves more information than sign clipping.

**Reference:** Hessel, M. et al. (2019). *Multi-task Deep Reinforcement Learning with PopArt.* AAAI. [https://arxiv.org/abs/1809.04474](https://arxiv.org/abs/1809.04474); Hafner, D. et al. (2023). *Mastering Diverse Domains through World Models (DreamerV3).* [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

---

## 7. Practical Recommendations by Game Type

### Atari (Dense Reward)

- **Default:** `ClipReward` (sign clipping) with standard DeepMind preprocessing. This is what GOLDS currently does and it is the right baseline.
- **If training a single game** and reward magnitudes matter (e.g., Seaquest where different enemies have different values), consider switching to `VecNormalize(gamma=0.99, clip_reward=10.0)` instead of sign clipping.
- **Multi-game training:** Keep sign clipping. It is the simplest way to ensure uniform reward scale. If you need richer signal, use symlog or PopArt.

### Platformers (Sparse-ish Reward)

- **Disable sign clipping** (`clip_reward=False` in `RetroPreprocessing`). Platform game scores are often too sparse for clipped rewards to provide useful gradient.
- **Add PBRS with x-position** as the potential function. This is the single highest-impact change for Mario-like games.
- **Use VecNormalize** on the combined (game + shaped) reward to handle scale differences across levels.
- **Consider a small time penalty** (-0.01 per step) to discourage standing still, but verify it does not cause the agent to suicide for early episode termination.

### Fighting Games (Self-Play)

- **Primary reward:** Win/loss outcome (+1 / -1 at episode end). This gives a clean learning signal for self-play (ELO-style improvement).
- **Auxiliary shaping:** Per-frame damage dealt/received as a dense supplement. Scale it small relative to win/loss (e.g., `0.001 * damage_delta`) so the agent still optimizes for winning, not just trading hits.
- **Do not clip rewards** -- the win/loss signal is already bounded, and clipping the damage component removes its magnitude information.
- **Self-play specifics:** Use opponent sampling from a pool of past checkpoints (as GOLDS already supports via `opponent_snapshot_dir`) to prevent reward cycling and maintain training stability.

---

## GOLDS-Specific Action Items

1. **Make `clip_reward` configurable per-game in YAML configs** rather than defaulting to `True` everywhere. Platformers and fighters should default to `False`.
2. **Add a `VecNormalize` option** in the training pipeline, gated by config, with proper save/load of statistics alongside model checkpoints.
3. **Implement a `PlatformerRewardWrapper`** that reads `x_pos` from retro's `info` dict and applies PBRS-style x-delta shaping.
4. **Add symlog reward transform** as an alternative to clipping for multi-game experiments: `reward = sign(r) * log(1 + |r|)`.
5. **Log raw (unshaped, unclipped) episode returns** alongside shaped returns in TensorBoard to detect reward hacking early.

---

*Last updated: 2026-03-19*
