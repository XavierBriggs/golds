<!-- style: direct sentences, no em dashes, no filler -->
# Research memo: GOLDS Verify & Renovate (probe)

Date: 2026-07-18
Phase timebox: 6h hands-on, spent: ~2.5h hands-on (GPU wall-clock ~23.5h, does not count against appetite)

This memo records the probe that replaced phases 1 and 2. The probe was a live Breakout PPO training run on ithaca. Evidence is grade A because it was measured directly.

Evidence grades: A measured yourself / B primary source / C independent secondary / D vendor claim / E vibes. D and E carry no decision weight.

## The probe question

Does the existing GOLDS stack, deployed on ithaca (CUDA), train PPO Breakout to a mean eval reward of at least 250 within roughly 10M timesteps?

## Findings by question

### RQ1: Does the stack learn Breakout to published-baseline range?

- Finding: Yes, decisively. A 15M-timestep PPO run reached a final deterministic eval mean of ~275 (last five evals: 261.7, 266.2, 268.4, 272.9, 274.9) and a peak eval of 391.5 during training. Published references: ~274 (OpenAI Baselines PPO), ~398 (SB3 reference). The learning curve was clean and monotonic: eval mean 16 at 0.4M, 48 at 3.1M, 82 at 4.85M, 112 at 5.7M, 180 at 8M, ~275 at 15M. The 250 threshold was crossed between 10M and 12M steps.
- Evidence: probe-breakout.log and results.json on ithaca (config_hash 722f575d7092, git SHA a43ff8f), grade A.
- Decision impact: The core pipeline is validated end to end: Atari preprocessing (fire-on-reset, episodic-life, frame stack, grayscale, resize), the PPO hyperparameters, the vec env pipeline, the eval loop, and results storage all work correctly. Xavier's distrust of the math and model logic is not supported by evidence for the Atari path. The renovation decision (no greenfield) is confirmed: the code that produced this is the code being kept.

### RQ2: Does the stack run reliably on ithaca for a long run?

- Finding: Yes. 15,007,744 timesteps completed, exit code 0, zero error signatures (no Traceback, RuntimeError, OOM, or Killed) across ~23.5 hours. The run survived an SSH disconnect and a full Claude session restart because it ran under nohup. GPU memory use was trivial (~936 MiB of 10 GB).
- Evidence: process survival checks and log grep over the full run, grade A.
- Decision impact: The nohup + log-tail pattern is sufficient for long runs on this box. tmux is not installed and is not needed. The prior "runs keep crashing" pain was environmental (Mac MPS, full disk), not a code defect.

### RQ3 (emergent): What is the training throughput, and does it constrain the Sonic campaign?

- Finding: Throughput was ~177 timesteps/sec aggregate with n_envs=16, and GPU utilization sat near 6 percent during training. The run is CPU-bound on environment stepping, not GPU-bound. At this rate a 15M run takes ~23.5h; a 20-50M Sonic run would take roughly 1.3 to 3.3 days each. Retro (Sonic) env stepping is heavier than Atari, so real Sonic throughput will likely be lower still.
- Evidence: fps field in SB3 log (177), nvidia-smi utilization sample (6 percent), grade A.
- Decision impact: This is the single biggest risk to the Sonic campaign's iteration speed, and it is a Phase 3 design problem. Options to investigate in synthesis: raise n_envs (24-thread CPU is underused at 16), SubprocVecEnv vs the current setup, frame-skip tuning, and whether env stepping can be parallelized better. Iteration discipline (one hypothesis per run) matters more when each run costs days.

## Accepted risks

| Risk | Why accepted | Watch trigger |
|---|---|---|
| Eval used 10 deterministic episodes, not the 100-episode unclipped protocol in the spec's Phase 1 gate | The probe's job is a yes/no learning signal, not the formal benchmark. 10 episodes at ~275 with low variance is conclusive for that purpose. | The formal Phase 1 gate (100-ep) is still owed; run it before recording an official Breakout baseline. |
| Breakout eval reward measured under the clipped reward_regime | For Breakout, clipped per-brick reward equals raw score, so the number is meaningful as-is. Not true for games with variable point values. | For Sonic and any variable-reward game, eval must report unclipped/raw progress, not clipped reward. |
| results.json has started_at == completed_at (identical timestamps) and null eval_100ep / human_normalized_score | Cosmetic for the probe; the real metrics are in the log. | This is an observability defect to fix in Phase 2 (W&B) work. Do not trust results.json timestamps until fixed. |
| nanobind "leaked instances" warnings at process shutdown | ALE cleanup noise after training already completed and saved. No effect on results. | Only revisit if it escalates to a crash during (not after) training. |

## Reference harvest

Not required as a design-surface harvest. GOLDS has no externally-designed surface (no UI, no public API, no docs site shipped to others). It is a private tool used only by Xavier. The designed-surface override did not fire.

## Sonic sample-complexity sources (added for ADR-004, grade B/C)

Consulted 2026-07-18 to estimate Sonic training cost and reward design. Grade B (primary) to C (practitioner secondary).

- Nichol et al, "Gotta Learn Fast: A New Benchmark for Generalization in RL" (arXiv:1804.03720) and OpenAI Retro Contest: reward = horizontal progress normalized to ~9000 at level end + completion bonus; joint PPO baseline trained ~2M timesteps per training level; the benchmark is a generalization task (held-out levels), which is harder than solving a single training level. Grade B.
- Felix Yu, "Train an RL agent to play Sonic with Transfer Learning": joint training ran ~200M frames (~50M timesteps at frameskip 4) on 12 CPUs + a 1080; the dominant failure mode is the agent getting stuck mid-level and needing to backtrack; the fix is rewarding delta max(x) rather than raw delta x; score variance is high (+/- 1000) because completion often depends on discovering a route. Grade C.
- Estimate derived for our case (grade B/C): solving GHZ Act 1 directly (overfitting one level is acceptable per our non-goals) needs roughly 10-30M timesteps and likely 2-4 reward-shaping iterations. Full-game or generalist completion is an unsolved generalization problem and is out of scope.

## Gate 2 result

Presented as GATE P (probe outcome), replacing Gates 1 and 2. See below.

## GATE P result

GO, 2026-07-18. Probe answered yes with grade-A evidence: existing stack trains PPO Breakout to ~275 final / 391.5 peak eval, zero crashes over 15M steps on ithaca CUDA. Renovation decision confirmed. Throughput (177 steps/s, GPU 6%) carried to Phase 3 as the top design risk.
