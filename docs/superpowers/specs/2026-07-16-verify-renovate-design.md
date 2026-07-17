# GOLDS Verify & Renovate — Design

**Date:** 2026-07-16
**Status:** Approved
**Decision:** Renovate the existing repo with verification-first sequencing. No greenfield rewrite.

## Problem

GOLDS has never produced a trusted, completed training run. Symptoms reported and confirmed:

- Agents don't learn / can't complete levels.
- Runs crash or fight the operator (video subprocess bugs, emulator limits).
- No observability: when a run goes bad there is no way to see why.
- No confidence in the PPO math, reward shaping, or model logic.

Root-cause evidence gathered 2026-07-16:

- `results.json` contains exactly one run ever: a 100k-timestep Pong smoke test (≈50x too short to learn Pong), on `device=mps`.
- Training has been running on a Mac (MPS) while a capable CUDA box (ithaca) exists.
- ithaca's root disk was 100% full (609 MB free of 98 GB) — sufficient alone to cause crashed runs, corrupt checkpoints, and failed video writes.
- The repo was already overhauled once (v2, 2026-03) and upgraded again (v3, 2026-06); both left the core pains intact. Rewrites do not fix this class of problem.

Conclusion: the bottleneck is **validation and observability**, not code architecture. The harness (~5.9k LOC, pydantic configs, factory pattern, tests) is structurally fine.

## North star

**Agents that beat games.** First target: **Sonic the Hedgehog, GreenHillZone Act 1** completion. The harness is a means, not the product.

## Scope boundaries

- **Verified core path:** config → env factory → PPO → eval → results storage. Only this path is load-bearing.
- **Quarantined (kept, marked unvalidated, not deleted):** RND exploration, self-play + Elo, MK2/SF2 configs, and all game configs other than Breakout and Sonic.
- **Focus configs:** `breakout.yaml`, `sonic_the_hedgehog.yaml` (Pong skipped by decision — Breakout is the more sensitive single validator).
- **Training host:** ithaca (Ryzen 9 5900X 24T, 64 GB RAM, RTX 3080 10 GB, Ubuntu 24.04). Mac is for dev only.

## Phases and gates

Each phase has a machine-checkable gate. No phase starts until the previous gate passes.

### Phase 0 — Ops floor (ithaca)

1. Diagnose the full root disk (investigate before deleting; 93 GB used, owner unknown).
2. Install uv; clone GOLDS; install CUDA torch + stable-retro; import the Sonic ROM.
3. `golds doctor` passes with `device=cuda`.
4. Long-run hygiene: runs execute inside tmux with periodic checkpointing so SSH drops never kill a run.

**Gate:** a 100k-timestep smoke run completes on ithaca on GPU and writes results entry + checkpoint + TensorBoard logs.

### Phase 1 — Trust ladder (Breakout baseline reproduction)

Train Breakout PPO ~10M timesteps on ithaca.

- Published reference range: ~274 (Baselines PPO) to ~398 (SB3 reference).
- Breakout validates preprocessing (fire-on-reset, episodic-life, reward clipping), PPO hyperparameters, the vec pipeline, the eval protocol, and results math in one shot.
- A miss localizes a specific bug to debug systematically — no more diffuse distrust.

**Gate:** mean eval reward ≥ 250 over 100 evaluation episodes (deterministic eval env, episodic-life off, unclipped rewards), recorded in `results.json` with config hash + git SHA.

### Phase 1b — Rigorous PPO math & model-logic validation (parallel with Phase 1)

- **Wrapper math unit tests with hand-computed expected values:** reward-shaping deltas, frame-stack ordering, action-set mappings, sticky-action probabilities.
- **PPO invariant checks instrumented on the live Breakout run:** clip fraction ∈ (0, 0.3); approx-KL bounded; explained variance rising over training; advantages zero-mean/unit-std per batch after normalization; LR and clip schedules decay exactly as configured.
- **Verification notebook:** derive the PPO clipped surrogate objective and GAE in LaTeX; recompute both in numpy on a captured rollout batch; cross-check against SB3's computed losses to numerical tolerance. This is the definitive "the math is right" artifact.
- **Model-logic audit:** NatureCNN input normalization (0–255 vs 0–1), observation dtype/shape through the full wrapper chain, eval-vs-train env parity.

**Gate:** notebook cross-check matches SB3 within tolerance; all invariant checks pass on the Breakout run; wrapper unit tests green.

### Phase 2 — Observability (W&B hub)

- Weights & Biases (free tier) as primary dashboard. Every run logs: full config + git SHA, PPO health metrics (episode return, entropy, explained variance, approx-KL, clip fraction, value/policy loss), game-progress metrics (Sonic max-x per episode, level completion rate), periodic gameplay videos, and system stats (GPU util, env FPS).
- Run-comparison views: any two runs comparable side-by-side.
- TensorBoard remains the local fallback.
- `golds diagnose <run>`: reads a run's logs and prints a plain-English health report (e.g., "entropy collapsed at 2M steps — premature convergence").
- Telegram alerts (existing) hook milestones and failures.

**Gate:** a deliberately broken run (zero-reward env) is correctly flagged by `golds diagnose`; two real runs are comparable side-by-side in W&B.

### Phase 3 — Sonic campaign

GreenHillZone Act 1 via the OpenAI Retro Contest recipe:

- Reward: x-position delta + level-completion bonus (unit-tested per Phase 1b).
- Discrete joint-button action set (v3 `DiscreteActionWrapper`), frame skip, sticky actions per config.
- Budget: 20–50M timesteps per run on the 3080.
- Iteration discipline: every failed run must yield a stated hypothesis (from Phase 2 diagnostics) before the next run launches.

**Gate:** agent completes GHZ Act 1 in ≥ 80% of eval episodes, with recorded video proof.

## Testing & correctness

- Existing pytest suite stays green throughout all phases.
- All reward/wrapper math carries unit tests with hand-computed expected values.
- Every run records config hash + git SHA; results are attributable and reproducible.

## Explicitly out of scope

- Greenfield rewrite / new repo (decided against; revisit only if the existing code actively blocks Phases 1–2, in which case fall back to a strangler rewrite of the core loop only).
- MK2/SF2 self-play, RND, and the other 14 games (quarantined until the trust ladder reaches them).
- Any new algorithm work before the Sonic gate passes.
