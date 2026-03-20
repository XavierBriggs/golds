# Intrinsic Motivation and Exploration Methods for Reinforcement Learning

This document surveys practical exploration methods for RL in game environments, with emphasis on what is implementable within the Stable Baselines3 (SB3) ecosystem.

> **Note:** This was compiled from knowledge through early 2025. Verify links and library versions before use.

---

## 1. RND (Random Network Distillation)

**How it works:** RND uses two neural networks — a fixed, randomly initialized *target network* and a *predictor network* trained to match the target's outputs. The intrinsic reward is the prediction error: states visited frequently become predictable (low reward), while novel states produce high error (high reward). Unlike count-based methods, RND scales naturally to high-dimensional observations.

**SB3 ecosystem status:** SB3 does not include RND natively. The primary options are:
- **[sb3-contrib](https://github.com/Stable-Baselines-Contrib/stable-baselines3-contrib):** Check for community PRs; RND wrappers have been proposed but not merged into stable releases as of early 2025.
- **[stable-baselines3-contrib + custom wrapper](https://stable-baselines3.readthedocs.io/):** The most practical path is implementing RND as a reward wrapper around an SB3 PPO agent. The wrapper maintains the two networks and augments `env.step()` rewards.
- **[CleanRL](https://github.com/vwxyzjn/cleanrl):** Provides a single-file PPO+RND implementation (`ppo_rnd_envpool.py`) that is the easiest reference for porting into SB3-compatible code.

**Practical results:** RND achieved state-of-the-art on Montezuma's Revenge (~10,000 mean score) when it was published (Burda et al., 2018). It remains one of the most reliable intrinsic motivation methods for sparse-reward Atari games.

**Paper:** [Exploration by Random Network Distillation (Burda et al., 2018)](https://arxiv.org/abs/1810.12894)

---

## 2. ICM (Intrinsic Curiosity Module)

**How it works:** ICM learns a forward dynamics model in a learned feature space. It consists of: (1) a feature encoder that maps observations to embeddings, (2) an inverse model that predicts the action from consecutive state embeddings (which trains the encoder to capture action-relevant features), and (3) a forward model that predicts the next embedding given the current embedding and action. The intrinsic reward is the forward model's prediction error.

**Applicability to pixel observations:** ICM was specifically designed for pixel inputs. The inverse model forces the encoder to ignore visual noise (e.g., flickering backgrounds) that does not depend on the agent's actions, making it more robust than raw-pixel prediction.

**Pros:**
- Handles high-dimensional pixel observations well
- Filters out environment stochasticity through the inverse model
- Relatively simple to implement as a wrapper

**Cons:**
- Can get stuck in "noisy TV" scenarios where uncontrollable stochastic elements produce perpetual novelty (though less so than naive prediction methods)
- Forward model capacity limits exploration in very complex environments
- Performance degrades in environments where visual changes are subtle

**SB3 integration:** No native support. Implement as a custom wrapper or use the [curiosity-driven-exploration-pytorch](https://github.com/pathak22/noreward-rl) reference code and adapt it.

**Paper:** [Curiosity-driven Exploration by Self-Supervised Prediction (Pathak et al., 2017)](https://arxiv.org/abs/1705.05363)

---

## 3. Go-Explore

**How it works:** Go-Explore decouples exploration into two phases: (1) **Explore** — maintain an archive of visited states, select a promising state, return to it deterministically, then explore from it; (2) **Robustify** — take the discovered trajectories and train a robust policy via imitation learning or RL. The key insight is that returning to frontier states before exploring avoids the "derailment" problem where agents forget how to reach previously discovered states.

**Results:** Go-Explore solved Montezuma's Revenge and Pitfall (both notoriously hard exploration benchmarks) achieving superhuman scores. The original version required a domain-specific cell representation, but **policy-based Go-Explore** (Ecoffet et al., 2021) removed this requirement by using a learned goal-conditioned policy for the "return" phase.

**Recent developments:** Go-Explore's archive-based strategy has influenced follow-up work in open-ended learning and quality-diversity algorithms. It remains the gold standard for hard-exploration Atari games but is complex to implement from scratch.

**SB3 compatibility:** Not directly compatible. Go-Explore requires custom infrastructure (state archives, trajectory storage, deterministic return mechanisms). It is best suited as a standalone system rather than an SB3 wrapper.

**Paper:** [First return, then explore (Ecoffet et al., 2021)](https://arxiv.org/abs/2004.12919)

---

## 4. NovelD (Novelty Detection)

**How it works:** NovelD builds on RND but adds a relative novelty measure. Instead of using raw RND prediction error as the reward, it computes the *ratio* of the current state's novelty to its neighbors' novelty. This focuses exploration on truly novel transitions (crossing into new territory) rather than states that are simply at the edge of explored space.

**Key advantage:** Reduces the "couch potato" problem where agents linger near moderately novel states instead of pushing deeper into unexplored territory.

**Practical status:** NovelD showed improvements over RND on several hard-exploration games. Implementation requires an RND backbone plus an additional comparison step. No SB3 native support; build as a custom reward wrapper extending an RND implementation.

**Paper:** [NovelD: A Simple yet Effective Exploration Criterion (Zhang et al., 2021)](https://arxiv.org/abs/2106.13517)

---

## 5. RIDE (Rewarding Impact-Driven Exploration)

**How it works:** RIDE rewards the agent for taking actions that produce large changes in a learned state representation. The intrinsic reward is the L2 distance between consecutive state embeddings, weighted by an episodic state-visitation count. This combination encourages the agent to seek impactful, non-repetitive actions.

**Strengths:** RIDE explicitly penalizes revisiting states (via the count weighting), which helps in procedurally generated environments like MiniGrid where memorization is not useful.

**Results:** Strong performance on MiniGrid procedural environments and some Atari games. Less tested on complex visual environments like fighting games.

**SB3 compatibility:** Requires a custom reward wrapper with a learned embedding network and episodic count tracking. Moderate implementation effort.

**Paper:** [RIDE: Rewarding Impact-Driven Exploration (Raileanu & Rocktaschel, 2020)](https://arxiv.org/abs/2002.12292)

---

## 6. Count-Based Exploration Methods

**Classical approach:** Maintain visit counts N(s) for each state and use a bonus proportional to 1/sqrt(N(s)). This works directly in tabular or small discrete state spaces.

**Scaling to large state spaces:**
- **Hash-based counts (SimHash):** Hash high-dimensional observations into buckets and count bucket visits. Simple and effective. See [Tang et al., 2017](https://arxiv.org/abs/1611.04717).
- **Pseudo-counts via density models:** Use a density model (e.g., PixelCNN) to estimate a "pseudo-count" for each state. Theoretically elegant but computationally expensive. See [Bellemare et al., 2016](https://arxiv.org/abs/1606.01868).
- **Episodic counts (NGU / Agent57):** Track counts within an episode using an embedding space, combined with a long-term novelty signal. Agent57 (Badia et al., 2020) achieved superhuman performance on all 57 Atari games but is extremely complex.

**SB3 compatibility:** Hash-based counts are the easiest to integrate. Wrap the environment, hash observations (e.g., using a downsampled grayscale frame), and add the count bonus to the reward. This can be done in under 100 lines of wrapper code.

---

## 7. Practical Recommendations by Game Type

| Game Type | Recommended Methods | Notes |
|---|---|---|
| **Sparse-reward Atari** (Montezuma's Revenge, Pitfall, Private Eye) | RND, Go-Explore, NovelD | RND is the best effort-to-performance trade-off. Go-Explore if maximum performance is needed and implementation complexity is acceptable. |
| **Dense-reward Atari** (Breakout, Pong, Space Invaders) | Baseline PPO/DQN is usually sufficient | Intrinsic motivation can slightly help early training but is not critical. |
| **Platformers** (procedural or level-based) | ICM, RIDE, hash-based counts | ICM handles visual diversity well. RIDE excels when levels are procedurally generated. |
| **Fighting games** (Mortal Kombat, Street Fighter) | RND + shaped extrinsic reward, hash-based counts | Fighting games have dense health/score signals but sparse "strategy" discovery. Light exploration bonuses on top of reward shaping work best. Avoid heavy intrinsic signals that overwhelm the combat reward. |
| **Puzzle / navigation games** | Go-Explore, NovelD | Hard exploration with clear "progress gates" suits archive-based methods. |

### Implementation Priority for SB3 Projects

1. **Start with reward shaping + PPO** — often sufficient and zero extra infrastructure.
2. **Add hash-based count bonuses** — minimal code, meaningful improvement for sparse rewards.
3. **Implement RND wrapper** — best general-purpose intrinsic motivation; use CleanRL's implementation as reference.
4. **Consider ICM** — if pixel-based observations show high stochasticity that RND handles poorly.
5. **Go-Explore / Agent57** — only for research or when simpler methods demonstrably fail.

### Key Libraries and References

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) — core RL library
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — single-file implementations including PPO+RND
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) — environment wrappers for reward augmentation
- [MiniGrid](https://github.com/Farama-Foundation/Minigrid) — good testbed for exploration methods before scaling to Atari/ROM games
- [Agent57 paper (Badia et al., 2020)](https://arxiv.org/abs/2003.13350) — comprehensive but complex; useful as a reference for combining multiple exploration signals
