# RL Model Zoo Standards and Benchmarking Methodology

Research document covering current best practices for publishing, evaluating, and benchmarking reinforcement learning models.

---

## 1. Hugging Face RL Model Zoo (rl-baselines3-zoo)

The [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) repository is the standard reference implementation for training, evaluating, and sharing RL agents built with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). Trained models are organized by algorithm and environment:

```
rl-trained-agents/
  ppo/
    SpaceInvadersNoFrameskip-v4/
      SpaceInvadersNoFrameskip-v4.zip   # model weights
      config.yml                         # hyperparameters used
      SpaceInvadersNoFrameskip-v4/       # evaluation monitor logs
  dqn/
    BreakoutNoFrameskip-v4/
      ...
```

Each trained agent includes: the serialized policy (`.zip`), the exact hyperparameter config (`config.yml`), and optional TensorBoard logs and evaluation monitor CSVs. Models are published to the [Hugging Face Hub](https://huggingface.co/sb3) with model cards containing YAML front-matter metadata for discoverability (library name, tags, dataset, evaluation metrics). The zoo supports one-command training (`python -m rl_zoo3.train`) and evaluation (`python -m rl_zoo3.enjoy`), making results reproducible.

**Source:** [DLR-RM/rl-baselines3-zoo on GitHub](https://github.com/DLR-RM/rl-baselines3-zoo)

## 2. OpenAI Baselines Benchmarking Methodology

The [OpenAI Baselines](https://github.com/openai/baselines) project established the de facto evaluation protocol for Atari RL:

- **Training:** 10M frames (40M with frame skip of 4), using the `NoFrameskip-v4` environments with standard wrappers (episodic life, frame stacking of 4, max-pooling over last 2 frames, reward clipping to [-1, 1]).
- **Evaluation:** 100 episodes with a **deterministic** greedy policy (no epsilon-greedy exploration), reporting mean and standard deviation of undiscounted episode returns.
- **Seeds:** At minimum 3 random seeds; 5 seeds is preferred for confidence intervals. Each seed initializes both the environment and model parameters.

These conventions were formalized across the Mnih et al. (2015) DQN paper, the Schulman et al. (2017) PPO paper, and the Hessel et al. (2018) Rainbow paper.

**Sources:** [OpenAI Baselines](https://github.com/openai/baselines) | [Machado et al., 2018 - "Revisiting the Arcade Learning Environment"](https://arxiv.org/abs/1709.06009)

## 3. Human Normalized Scores for Atari

The standard Human Normalized Score (HNS) formula, introduced in Mnih et al. (2015):

$$
\text{HNS} = \frac{\text{Agent Score} - \text{Random Score}}{\text{Human Score} - \text{Random Score}}
$$

A score of 0.0 means random-level play; 1.0 means human-level. Scores above 1.0 indicate superhuman performance. The canonical human and random baselines come from the DQN paper (Mnih et al., Nature 2015) and were later refined by Machado et al. (2018).

### Reference Scores Table (Mnih et al. 2015 / Machado et al. 2018)

| Game                  | Random Score | Human Score  | DQN Score  | PPO Score (approx.) |
|-----------------------|--------------|--------------|------------|----------------------|
| Space Invaders        | 148.0        | 1,669.0      | 1,976.0    | ~1,200               |
| Breakout              | 1.7          | 31.8         | 401.2      | ~400                 |
| Pong                  | -20.7        | 9.3          | 21.0       | ~21                  |
| Q*bert                | 163.9        | 13,455.0     | 10,596.0   | ~14,000              |
| Seaquest              | 68.4         | 20,182.0     | 5,286.0    | ~1,800               |
| Asteroids             | 719.1        | 13,157.0     | 1,629.0    | ~2,100               |
| Ms. Pac-Man           | 307.3        | 15,693.0     | 2,311.0    | ~2,400               |
| Montezuma's Revenge   | 0.0          | 4,753.0      | 0.0        | ~0                   |
| Enduro                | 0.0          | 309.6        | 301.8      | ~1,200               |
| Frostbite             | 65.2         | 4,335.0      | 328.3      | ~300                 |

*Human/Random scores from Mnih et al. (2015, Nature). DQN scores from same. PPO scores are approximate from Schulman et al. (2017) and rl-baselines3-zoo benchmarks.*

**How to look up scores:** The canonical source is Table 1 in [Mnih et al. (2015)](https://www.nature.com/articles/nature14236). Updated random/human baselines using the "no-op start" and "human start" protocols are in [Machado et al. (2018)](https://arxiv.org/abs/1709.06009), Table 2.

## 4. Model Card Format and Metadata Standards

Following the [Hugging Face Model Card specification](https://huggingface.co/docs/hub/model-cards), each RL model should include a `README.md` with YAML front-matter:

```yaml
---
library_name: stable-baselines3
tags:
  - reinforcement-learning
  - ppo
  - atari
  - SpaceInvadersNoFrameskip-v4
datasets:
  - atari
model-index:
  - name: PPO-SpaceInvaders
    results:
      - task:
          type: reinforcement-learning
        dataset:
          name: SpaceInvadersNoFrameskip-v4
          type: atari
        metrics:
          - name: mean_reward
            type: mean_reward
            value: 1200.0
---
```

The free-text body should include: model description, intended use, training procedure, hyperparameters, evaluation results, and limitations. For RL specifically, document the environment version, wrapper stack, observation/action spaces, and reward structure.

**Source:** [Hugging Face Model Cards Documentation](https://huggingface.co/docs/hub/model-cards)

## 5. Reproducibility Standards

Every published RL model should log:

- **Configuration:** Full hyperparameter config (YAML or JSON), including learning rate schedule, batch size, number of environments, entropy coefficient, clip range, etc.
- **Random seeds:** All seeds used (environment seed, numpy seed, torch seed). Report results across multiple seeds.
- **Library versions:** Exact versions of `stable-baselines3`, `gymnasium`/`gym`, `torch`/`tensorflow`, `numpy`, and Python.
- **Hardware:** GPU model, number of CPUs/GPUs, total wall-clock training time.
- **Environment specification:** Full environment ID (e.g., `SpaceInvadersNoFrameskip-v4`), wrapper stack (frame skip, frame stack, reward clipping), and any preprocessing.
- **Commit hash:** Pin the exact code version via git SHA or release tag.

Use `requirements.txt` or `pyproject.toml` lock files, and consider Docker images for full environment reproducibility.

**Source:** [Pineau et al. (2021), "Improving Reproducibility in Machine Learning Research"](https://arxiv.org/abs/2003.12206)

## 6. Standard Evaluation Protocols

| Parameter                  | Standard Practice                                      |
|---------------------------|--------------------------------------------------------|
| Number of eval episodes   | **100** (minimum 30; 100 is the community standard)    |
| Policy mode               | **Deterministic** (greedy) for final evaluation        |
| Multi-seed evaluation     | **3-5 seeds** minimum; report mean +/- std             |
| Frame skip (training)     | 4 (action repeated 4 times) via `NoFrameskip-v4` + wrapper |
| Frame skip (evaluation)   | Same as training (4), to match dynamics                |
| Stochastic evaluation     | Used alongside deterministic for robustness analysis   |
| Episode termination       | On life loss (training only); full game for evaluation |
| Score aggregation         | Report **mean**, **median**, **std**, and optionally IQM |

Recent work by Agarwal et al. (2021) recommends using **Interquartile Mean (IQM)** and **performance profiles** instead of simple mean/median for more robust aggregate comparisons across games.

**Sources:** [Agarwal et al. (2021), "Deep RL at the Edge of the Statistical Precipice"](https://arxiv.org/abs/2108.13264) | [Machado et al. (2018)](https://arxiv.org/abs/1709.06009)

## 7. Published PPO Baseline Scores for Atari

PPO scores for Atari are available from several authoritative sources:

1. **Schulman et al. (2017)** -- The original PPO paper reports scores on 49 Atari games using 40M frames of training. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
2. **rl-baselines3-zoo benchmarks** -- Stable-Baselines3 maintains benchmark results for PPO (and other algorithms) at [https://github.com/DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) with results viewable on the Hugging Face Hub under the [sb3 organization](https://huggingface.co/sb3).
3. **Huang et al. (2022), CleanRL** -- Provides single-file PPO implementations with tracked benchmark results. [https://github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) | [arXiv:2208.01626](https://arxiv.org/abs/2208.01626)
4. **Hessel et al. (2018), Rainbow** -- Includes PPO comparisons in its multi-algorithm benchmark table. [arXiv:1710.02298](https://arxiv.org/abs/1710.02298)

When citing PPO Atari results, specify: the exact environment ID (e.g., `BreakoutNoFrameskip-v4`), the number of training timesteps (10M or 40M frames), the evaluation protocol (deterministic, 100 episodes), and the number of seeds.

---

## Key References

- Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540). [Link](https://www.nature.com/articles/nature14236)
- Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Machado, M. C. et al. (2018). "Revisiting the Arcade Learning Environment." [arXiv:1709.06009](https://arxiv.org/abs/1709.06009)
- Agarwal, R. et al. (2021). "Deep Reinforcement Learning at the Edge of the Statistical Precipice." [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)
- Raffin, A. et al. (2021). "Stable-Baselines3: Reliable RL Implementations." *JMLR*. [arXiv:2005.05719](https://arxiv.org/abs/2005.05719)
- Huang, S. et al. (2022). "CleanRL: High-quality Single-file Implementations of Deep RL Algorithms." [arXiv:2208.01626](https://arxiv.org/abs/2208.01626)
- Pineau, J. et al. (2021). "Improving Reproducibility in Machine Learning Research." [arXiv:2003.12206](https://arxiv.org/abs/2003.12206)
- Mitchell, M. et al. (2019). "Model Cards for Model Reporting." [arXiv:1810.03993](https://arxiv.org/abs/1810.03993)
