# PPO (Proximal Policy Optimization) Advances: 2024-2025

> **Note:** This document was compiled from knowledge available through early 2025. Web search was unavailable at generation time; verify links and claims against current literature before relying on them for production decisions.

---

## 1. Learning Rate Schedules

Linear decay remains the default in most PPO Atari benchmarks, annealing the learning rate from an initial value (typically `2.5e-4`) to zero over the course of training. This is the schedule used in the original OpenAI baselines and in the "37 Implementation Details of PPO" study (Huang et al., 2022).

- **Linear decay** is the most battle-tested choice for Atari. SB3 exposes this via `learning_rate=linear_schedule(2.5e-4)`.
- **Cosine annealing** has gained traction in 2024 work influenced by LLM RLHF pipelines (e.g., OpenAI's PPO for InstructGPT). For game RL, cosine annealing can help if training length is uncertain -- it decays aggressively early, then flattens, reducing destructive late-stage updates.
- **Actionable recommendation:** Start with linear decay for Atari. Switch to cosine annealing only if you observe instability in the last 20% of training or if your total timestep budget is approximate.

**References:**
- [The 37 Implementation Details of PPO (ICLR Blog Track 2022)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [SB3 PPO Docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

---

## 2. Clip Range Annealing

The PPO clip parameter (epsilon) controls how far the new policy can deviate from the old. The original paper uses a fixed `epsilon = 0.2`, but annealing epsilon from `0.2` to `0.0` over training is a common trick.

- **Fixed clip (`0.2`)** is simpler and works well for most Atari games.
- **Linear clip annealing** (`0.2 -> 0.0`) was shown by Huang et al. (2022) to provide marginal gains on some Atari games by tightening the trust region as the policy matures.
- **Recent consensus (2024):** Clip annealing is a "nice-to-have" but not critical. The bigger win comes from learning rate annealing. If you anneal both, ensure they decay on the same schedule to avoid mismatched optimization dynamics.
- **Actionable recommendation:** Use `clip_range=linear_schedule(0.2)` in SB3 alongside a linear LR schedule. This is low-risk and occasionally improves final performance by 5-10% on score-sensitive games.

**References:**
- [Schulman et al., 2017 - PPO Paper](https://arxiv.org/abs/1707.06347)
- [Huang et al., 2022 - Implementation Details](https://arxiv.org/abs/2005.12729)

---

## 3. Entropy Coefficient Scheduling

Entropy regularization encourages exploration. The standard coefficient is `0.01` for Atari.

- **Fixed entropy (`0.01`)** is the SB3 default and works reliably for most Atari games.
- **Decayed entropy** (e.g., `0.01 -> 0.001` over training) can help in games where early exploration is critical but late-stage exploitation matters (e.g., Montezuma's Revenge, Pitfall).
- **Increased entropy (`0.02-0.05`)** helps in sparse-reward environments but risks policy collapse if set too high.
- **2024 trend:** Several papers (including PPG and DAPO work) advocate for adaptive entropy tuning similar to SAC's automatic temperature adjustment, but this remains experimental for discrete-action PPO.
- **Actionable recommendation:** Keep `ent_coef=0.01` as default. For hard-exploration Atari games, try a linear decay from `0.02` to `0.005`. Implement as a custom callback in SB3.

---

## 4. Value Function Loss Clipping Debate

PPO's original implementation clips the value function loss similarly to the policy loss. This has been contentious.

- **Clipped VF loss** (`clip_range_vf = 0.2`): Intended to prevent large value function updates. However, Engstrom et al. (2020) and Andrychowicz et al. (2021, "What Matters in On-Policy RL") found it **hurts performance** in most settings.
- **Unclipped VF loss** (`clip_range_vf = None`): Generally performs better. The value function benefits from unconstrained fitting, since its accuracy directly affects advantage estimation.
- **2024 consensus:** The community has largely moved to **no VF clipping**. SB3 defaults to `clip_range_vf = None` since v1.6+.
- **Actionable recommendation:** Use `clip_range_vf=None` (SB3 default). Do not clip the value loss unless you have a specific reason.

**References:**
- [Andrychowicz et al., 2021 - What Matters in On-Policy RL](https://arxiv.org/abs/2006.05990)
- [Engstrom et al., 2020 - Implementation Matters](https://arxiv.org/abs/2005.12729)

---

## 5. Symlog Predictions and PopArt Normalization

Handling large and variable reward scales across Atari games is a long-standing challenge for value heads.

- **PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets):** Normalizes value targets adaptively, allowing a single agent to handle diverse reward scales. Used prominently in IMPALA and Agent57.
- **Symlog transform** (introduced in DreamerV3, Hafner et al., 2023): Applies `sign(x) * log(1 + |x|)` to value predictions and targets. This compresses large values while preserving sign and handling zero well. It has become increasingly popular in 2024.
- **Practical impact:** Symlog is simpler to implement than PopArt and avoids the need for running statistics. For Atari, where rewards can range from +1 (Pong) to thousands (Breakout), symlog stabilizes training.
- **Actionable recommendation:** If training across multiple Atari games with a shared architecture, implement symlog on the value head. For single-game training, reward clipping to `[-1, 1]` (as in SB3's Atari wrappers) is sufficient and simpler.

**References:**
- [Hafner et al., 2023 - DreamerV3](https://arxiv.org/abs/2301.04104)
- [van Hasselt et al., 2016 - PopArt](https://arxiv.org/abs/1602.07714)

---

## 6. Other Recent PPO Improvements

### Phasic Policy Gradient (PPG)
- Decouples policy and value function training into separate phases, allowing the value function to train for more epochs without distorting the policy.
- Shows gains on hard Atari games like Montezuma's Revenge. Available in SB3-contrib or custom implementations.
- **Reference:** [Cobbe et al., 2021 - PPG](https://arxiv.org/abs/2009.04416)

### DAPO (Direct Advantage Policy Optimization)
- Emerged in 2024-2025 primarily in the LLM/RLHF context. Uses group-relative advantage normalization and removes the KL penalty in favor of clip-only constraints.
- Key ideas transferable to game RL: **per-minibatch advantage normalization** (instead of per-batch) can reduce variance.
- **Reference:** [DAPO, 2025](https://arxiv.org/abs/2503.14476)

### Advantage Normalization Tricks
- **Per-minibatch normalization:** Normalize advantages within each minibatch rather than across the full rollout buffer. Reduces stale-statistics effects.
- **Advantage clipping:** Clip extreme advantages (e.g., beyond 3 standard deviations) to prevent catastrophic policy updates from outlier transitions.
- **GAE lambda tuning:** The default `lambda=0.95` works well generally. For long-horizon games, `lambda=0.98` can improve credit assignment at the cost of higher variance.

### Other Notable Improvements
- **Observation normalization:** Running mean/std normalization of observations is one of the most impactful tricks per Andrychowicz et al. (2021). SB3's `VecNormalize` wrapper handles this.
- **Gradient clipping:** `max_grad_norm=0.5` (SB3 default) is robust. Increasing to `1.0` occasionally helps for complex games.

---

## 7. SB3 Best Practices for PPO on Atari

### Recommended Atari Hyperparameters (SB3)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.utils import linear_schedule

model = PPO(
    "CnnPolicy",
    env,
    learning_rate=linear_schedule(2.5e-4),
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=linear_schedule(0.1),
    clip_range_vf=None,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
)
```

### Key SB3 Tips

- **Use `n_envs=8` (or more):** Parallel environments via `SubprocVecEnv` dramatically improve sample throughput and training stability.
- **Atari wrappers matter:** Always use `make_atari_env` + `VecFrameStack(env, n_stack=4)`. These apply reward clipping, frame skipping, and episodic life resets.
- **Monitor with TensorBoard:** Track `explained_variance` (should approach 1.0), `entropy_loss` (should decay slowly, not collapse), and `clip_fraction` (should be 0.05-0.15; if >0.3, the LR or clip range is too aggressive).
- **Do not over-train epochs:** `n_epochs=4` is standard for Atari. Going higher (e.g., 10) causes overfitting to the rollout buffer and policy degradation.
- **Checkpoint frequently:** Atari training can regress. Save checkpoints every 500k steps and evaluate on 10+ episodes.

**References:**
- [SB3 RL Zoo - Atari Hyperparameters](https://github.com/DLR-RM/rl-baselines3-zoo)
- [SB3 Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

---

## Summary of Actionable Changes

| Technique | Default | Recommended Change | Expected Impact |
|---|---|---|---|
| LR schedule | Fixed | Linear decay `2.5e-4 -> 0` | Significant stability gain |
| Clip range | Fixed `0.2` | Linear decay `0.1 -> 0` | Marginal improvement |
| Entropy coef | Fixed `0.01` | Keep fixed (or decay for hard-explore games) | Low risk |
| VF loss clipping | Clipped | **Disable** (`None`) | Moderate improvement |
| Symlog value head | N/A | Add for multi-game training | Moderate improvement |
| Advantage norm | Per-batch | Per-minibatch | Small variance reduction |
| Observation norm | Off | Enable via `VecNormalize` | Significant improvement |
