<!-- style: direct sentences, no em dashes, no filler -->
# Synthesis: GOLDS Verify & Renovate

Date: 2026-07-18
Phase timebox: 30 min

## Options considered

1. **Thin renovation.** Fix throughput, add minimal W&B logging, shape the Sonic reward, run it. Skip the formal 100-episode Breakout gate and the PPO math-derivation notebook, on the grounds that the probe already validated the math empirically (grade A). Fastest path to a Sonic run. Risk: no formal baseline on record, and Xavier's stated wish to understand the math goes unmet.

2. **Rigorous trust ladder.** Formal Breakout 100-episode gate, a LaTeX PPO-derivation notebook cross-checked against SB3, live invariant checks, wrapper unit tests, then Sonic. Highest confidence and best learning artifact. Risk: the math-derivation notebook's main justification (distrust) was just dissolved by the probe, so a chunk of this work now buys less than when the spec was written.

3. **Throughput-led (chosen).** Treat throughput as its own first milestone because it multiplies every downstream run, then run a formal Breakout gate and the Sonic campaign on top of a W&B observability layer, keeping math validation lightweight (invariant checks + wrapper unit tests) with the full derivation notebook deferred to an optional learning task. Sequences the measured top risk first and keeps correctness checks that are cheap while cutting the one that the probe made redundant.

Chosen: Option 3. The probe proved correctness is not the bottleneck; iteration speed is. Sequence the throughput fix first, keep only the cheap correctness checks, and spend the saved time on Sonic reward iterations.

## Goals

| Goal | Success criterion (number or binary test) |
|---|---|
| Measure and raise retro throughput | First record a grade-A retro throughput baseline (Sonic config, n_envs=24). Then set and hit a target defined as a multiple of that measured baseline, not the Atari 177. SubprocVecEnv is already on, so the levers are n_envs, frameskip, preprocessing cost. If retro caps near 200 steps/sec, that is a recorded finding that re-plans the Sonic budget. |
| Build the Sonic completion instrument | Net-new: delta-max(x) reward, per-level end-x threshold, level-completion detector, and a raw-progress eval path. All carry unit tests with hand-computed expected values. The north star cannot be measured until this exists; today's wrapper is raw delta-x with no completion detection. |
| Record a formal Breakout baseline | Mean eval reward >= 250 over 100 deterministic episodes, unclipped, written to results.json with config hash + git SHA. |
| Observability that explains failures | A deliberately broken run (zero-reward env) is flagged by `golds diagnose` (capped at that binary flag, not a dashboard); every run streams PPO health metrics, game-progress metrics, and periodic video (via the retro video-subprocess path) to W&B; any two runs are comparable side by side. |
| Cheap correctness checks | Wrapper math unit tests (reward deltas, frame-stack order, action-set mappings) pass; live PPO invariant checks (clip fraction in (0,0.3), bounded approx-KL, rising explained variance, zero-mean/unit-std normalized advantages) pass on a real run. |
| Sonic GHZ Act 1 completion | Agent completes Green Hill Zone Act 1 in >= 80 percent of a 100-episode deterministic eval, with recorded video proof. Sonic config eval must first be fixed from its current eval_episodes=3 / deterministic=false to the 100-episode deterministic protocol. North star. Proves the stack can learn one level end-to-end (overfitting the trained level is acceptable per non-goals); it does not prove cross-level generalization. |

## North star metric

**Green Hill Zone Act 1 completion rate over a 100-episode deterministic eval.** Target >= 80 percent. Measured at the end of each Sonic training run and logged to W&B. One number. When it clears 80 it says the stack can learn a single level end-to-end, which is the milestone this overhaul is scoped to; it does not claim cross-level generalization, which the non-goals exclude.

## Non-goals (restated)

- No greenfield rewrite. The probe confirmed the existing code produces published-baseline results; it is kept. (ADR-001)
- No full-game Sonic completion and no generalist cross-level agent. Research shows single-agent full-game completion is an unsolved generalization problem; out of scope for this overhaul.
- No quarantined subsystems until the Sonic gate passes: RND (and the Montezuma/Frostbite hard-exploration games it serves), self-play/Elo (and MK2/SF2), and the other non-Sonic non-Breakout game configs.
- No new algorithm work before the Sonic gate. RecurrentPPO tuning and exploration research wait.
- No full LaTeX PPO-derivation notebook in the critical path. Deferred to an optional learning task; the probe already dissolved the distrust it was meant to cure. (ADR-005)

## Code standards / engineering principles

- Adopt the installed `code-standards` skill bar by reference. Project-specific deltas below.
- Python 3.12, uv-managed, existing pydantic-config + factory structure preserved.
- Every training run records config hash + git SHA; a result is not trusted unless reproducible from those.
- Reward and wrapper math carries unit tests with hand-computed expected values. This is the one place correctness is asserted in code, not just observed in a curve.
- Prefer fixing the existing module over adding a parallel one. Two prior overhauls already grew the surface; this one shrinks the untrusted part by quarantine, not by addition.
- Eval reward for any variable-point game (Sonic and beyond) must report unclipped raw progress, never clipped reward. (ADR-004)

## Architecture

No structural change. The validated core path stays: YAML config -> pydantic schema -> env factory -> PPO (SB3) -> eval -> results store. Three renovations bolt onto that path without reshaping it:

1. **Throughput:** SubprocVecEnv is already the default; measure the retro baseline, then raise via n_envs / frameskip / preprocessing, tuned by measurement. (ADR-002)
2. **Observability:** a W&B callback wraps the existing SB3 callback chain; TensorBoard stays as local fallback; a new `golds diagnose` command reads a run's logs and prints a binary health verdict; video routes through the retro video-subprocess path. (ADR-003)
3. **Sonic reward/eval (net-new):** replace the current raw delta-x `PlatformerRewardWrapper` with delta-max(x) progress + completion bonus, and build a level-completion detector + raw-progress eval path. None of this exists today. (ADR-004)

Everything else (RND, self-play, other games) is untouched and quarantined.

ADRs: docs/inception/adr/001-renovate-not-greenfield.md, 002-env-throughput.md, 003-observability-wandb.md, 004-sonic-reward-eval.md, 005-math-validation-depth.md

## Second pre-mortem

We built exactly this and it failed. The stories:

1. Throughput fix hit a wall: retro emulator stepping does not parallelize well under SubprocVecEnv (single-emulator-per-process limits, the same class of limit that forced the video-subprocess workaround in git history), and we never got past ~200 steps/s. Sonic runs stayed multi-day and iteration crawled. Mitigation: throughput is goal 1 with a hard number, measured before any Sonic run; if it stalls, that is a visible gate failure, not a surprise mid-campaign.
2. Sonic completion never reached 80 percent. The agent completed the level sometimes (lucky-route variance) but not robustly, and reward-shaping iterations chewed through the appetite. Mitigation: budget 2-4 shaping iterations explicitly; if the north star is not cleared inside appetite, the framing forces an extend/ship/kill decision rather than an open-ended grind.
3. W&B became a time sink: dashboard building expanded to fill the time it was given. Mitigation: observability success is a binary (broken run is flagged; two runs comparable), not a polish target.
4. The quarantine leaked: someone re-enabled RND or a self-play path to "help" Sonic, and an unvalidated subsystem silently corrupted results. Mitigation: quarantine is a stated non-goal; results stay attributable via config hash + git SHA so a leak is detectable.

## Red team

Run in a fresh-context subagent 2026-07-18. It checked claims against the code; the three load-bearing ones were re-verified by hand before acting.

| Finding | Disposition (fixed / accepted: why / killed direction) |
|---|---|
| ADR-002's named fix (switch to SubprocVecEnv) is already the default in the code; the 177 was already measured under subproc. The milestone was built on a stale picture. Verified: factory.py:46, both makers default use_subproc=True. | Fixed. ADR-002 rewritten to measure-first and pull the real remaining levers (n_envs, frameskip, preprocessing). |
| The >=400 steps/s and >40 percent GPU targets are invented, extrapolated the wrong way (Atari->retro, when retro is slower), and mutually incoherent. | Fixed. Targets removed; replaced with a grade-A retro baseline measurement, then a target set as a multiple of it. |
| The north star's instrument does not exist: no completion detector, no delta-max(x), no raw-progress eval. Shipped wrapper is raw delta-x (the rejected option). Verified in retro/wrappers.py:135-205. | Fixed. Added a first-class "build the Sonic completion instrument" goal; ADR-004 Harder note corrected to say this is net-new code. |
| North star over-claimed ("says the system works"); it measures single-level memorization. Sonic config is eval_episodes=3 / deterministic=false, which cannot produce the stated 100-ep number. Verified in config. | Fixed (wording) + accepted (scope). Reworded to "can learn one level end-to-end"; config fix to 100-ep deterministic added as an explicit build task. Single-level scope is intentional per non-goals. |
| ADR-004 cited a "literature scan (grade B/C)" that the research memo said was never done. | Fixed. Added the actual Sonic sources (Gotta Learn Fast, Retro Contest, Felix Yu) to the memo with honest grades; ADR-004 now points to them. |
| ADR-005's "the probe dissolved the distrust" is outcome-based, not a derivation check, and is used to defer the understanding deliverable Xavier explicitly asked for, inferring his priorities. | Surfaced to user. The cheap checks stay. Whether to keep a small PPO/GAE derivation notebook or defer it is Xavier's call, not an inference; asked at Gate 3. |
| "Periodic video to W&B" collides with retro's single-emulator-per-process limit (same constraint ADR-002 faces), unacknowledged. | Fixed. ADR-003 now requires video to route through the existing video-subprocess path. |
| `golds diagnose` as a bespoke plain-English verdict is mild gold-plating vs the binary "broken run is flagged" requirement. | Accepted with cap. Kept but explicitly capped at the binary flag in ADR-003 and the goals table; not to grow into a dashboard. |

## Gate 3 result

GO, 2026-07-18. Xavier confirmed deferring the PPO-derivation notebook (ADR-005 stands; cheap correctness checks kept). All red-team findings dispositioned; three load-bearing ones verified against code and fixed.
