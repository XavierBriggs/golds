<!-- style: direct sentences, no em dashes, no filler -->
# Spec: GOLDS Verify & Renovate

Date: 2026-07-18
Status: LIVING. This spec is expected to change. See Living spec below.
Last re-sync: 2026-07-18, after M2 R1/R3-R5/R10 (retro throughput measured ~2000 steps/s; Sonic ROM verified)

## Problem

GOLDS has never produced a trusted training run: agents fail to complete levels, runs crashed on infrastructure, no observability explained failures, and the PPO math was not trusted. Root causes were environmental (Mac MPS, full disk on the real training box) plus missing observability, not a broken architecture. A probe on ithaca settled it: the existing stack trains PPO Breakout to published-baseline scores. This overhaul renovates the validated stack to reach the real goal, an agent that completes Sonic Green Hill Zone Act 1. Full context: 00-framing.md. History and decision: docs/superpowers/specs/2026-07-16-verify-renovate-design.md.

## Goals and north star

| Goal | Success criterion |
|---|---|
| G1 Measure and raise retro throughput | Record a grade-A retro throughput baseline (Sonic config, n_envs=24), then hit a target set as a multiple of it. Levers: n_envs, frameskip, preprocessing. A ~200 steps/sec cap is a recorded finding that re-plans the Sonic budget. |
| G2 Build the Sonic completion instrument | delta-max(x) reward, per-level end-x threshold, completion detector, raw-progress eval path. All unit-tested with hand-computed values. |
| G3 Formal Breakout baseline | Mean eval reward >= 250 over 100 deterministic unclipped episodes, in results.json with config hash + git SHA. |
| G4 Observability | Broken (zero-reward) run flagged by `golds diagnose`; runs stream PPO health + game-progress + periodic video to W&B; any two runs comparable. |
| G5 Cheap correctness checks | Wrapper math unit tests pass; live PPO invariant checks pass on a real run. |
| G6 Sonic GHZ Act 1 completion (north star) | Completes GHZ Act 1 in >= 80 percent of a 100-episode deterministic eval, with video proof. |

**North star:** GHZ Act 1 completion rate over a 100-episode deterministic eval, target >= 80 percent, logged to W&B. It proves the stack can learn one level end-to-end; it does not claim cross-level generalization.

## Non-goals

- No greenfield rewrite. (ADR-001)
- No full-game Sonic completion and no generalist cross-level agent (unsolved generalization problem).
- No quarantined subsystems until the Sonic gate passes: RND (and Montezuma/Frostbite), self-play/Elo (and MK2/SF2), other non-Sonic non-Breakout configs.
- No new algorithm work before the Sonic gate.
- No PPO-derivation notebook in the critical path (deferred by Xavier's decision, 2026-07-18). (ADR-005)

## Requirements

| # | Requirement (testable) | Traces to goal |
|---|---|---|
| R1 | DONE 2026-07-18: Sonic config at n_envs=24 measured ~2000 steps/sec (1887-2062), grade A, W&B run dy2qpwrz. | G1 |
| R2 | SATISFIED (no engineering needed): retro ~2000 steps/sec is on par with Atari ~2200, so there is no throughput deficit to fix. The only ops action is disabling ithaca sleep (M0). A 20-50M Sonic run is ~3-7 GPU-hours. | G1 |
| R3 | A delta-max(x) reward replaces the raw delta-x in PlatformerRewardWrapper, unit-tested against hand-computed expected values. | G2, G5 |
| R4 | A level-completion detector and per-level end-x threshold exist for GHZ Act 1, unit-tested. | G2 |
| R5 | A raw-progress eval path reports unclipped completion (reached level-end x), unit-tested. | G2 |
| R6 | The Sonic config eval is fixed to 100 episodes deterministic (from eval_episodes=3 / deterministic=false). | G6 |
| R7 | DONE 2026-07-19: Breakout 15M run, raw (unclipped) eval 323-343, well above the >=250 gate and in the published PPO band (~274-400). Clean tree (git_dirty=False), eval-field code populated. Config e3d674a. | G3, G4 |
| R8 | A W&B callback logs config + git provenance, PPO health metrics, game-progress metrics, GPU/FPS, and periodic video (via the retro video-subprocess path) for every run. Its done-condition is asserted programmatically via the wandb API (run exists, expected keys present), not by eyeballing a dashboard. | G4 |
| R9 | `golds diagnose <run>` reads the run's local results.json row and prints a binary health verdict. Broken predicate is explicit: best_eval_reward is None or <= epsilon flags broken. Capped at the flag, not a dashboard. No W&B dependency. | G4 |
| R10 | DONE 2026-07-18: PPOInvariantCallback built (clip fraction, approx-KL, EV trend, finite non-degenerate advantages; advantage-normalization limitation documented honestly). Surfaced no violations on the real Sonic probe run. Config-flag gated. | G5 |
| R11 | The agent completes GHZ Act 1 in >= 80 percent of a 100-episode deterministic eval, with recorded video. | G6 |
| R12 | TrainingResult carries git_sha and git_dirty; a result recorded on a dirty tree is marked git_dirty=true, so the reproducibility claim is honest during iteration. Schema change + capture code. | G3, G4 |

## Architecture

No structural change. Validated path stays: YAML config -> pydantic schema -> env factory -> PPO (SB3) -> eval -> results store. Three renovations bolt on: throughput (measure-first), a W&B observability callback + diagnose command, and a net-new Sonic reward/completion/eval path. Everything else quarantined. Detail in 03-synthesis.md.

ADRs: adr/001-renovate-not-greenfield, adr/002-env-throughput, adr/003-observability-wandb, adr/004-sonic-reward-eval, adr/005-math-validation-depth.

Designed-surface constraints: none. GOLDS has no external designed surface; the reference harvest did not fire. Binding technical constraint carried from the memo: retro is single-emulator-per-process, so both env parallelism (R2) and video logging (R8) must respect it; video routes through the existing subprocess recorder.

## Milestones

### M0: Human-provisioned preconditions (Xavier, not a build agent)

Things only Xavier can do; build sessions cannot. Must exist before the milestone that needs them.

- W&B account created and `WANDB_API_KEY` set on ithaca. (Needed for M1b, not M1a.)
- Sonic Genesis ROM imported on ithaca (`golds rom import`). Not redistributable; not in the repo. (Needed for M2 retro probe and M3; R3-R5 unit tests do NOT need it.)
- ithaca reachable: Tailscale VPN up, `xbriggs` creds working. Documented in the repo CLAUDE.md.
- ithaca sleep/suspend DISABLED before any long run (`sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`). Evidence (2026-07-18): the probe's 177 steps/sec was wall-clock-contaminated by the box suspending mid-run; a clean short run clocked ~2200 steps/sec on the same config. Long runs must not sleep.

### M1a: Walking skeleton (fully local, zero external deps) [DONE 2026-07-18]

Verified end to end on ithaca: 50k Breakout run wrote a results.json row with distinct started_at/completed_at, real git_sha (d883df4), git_dirty=True, and `golds diagnose breakout` returned HEALTHY (best_eval_reward=2.2). Code committed d883df4, 113 tests green.

The thinnest slice that actually runs end to end with no SaaS and no human verification: fix the results.json timestamp bug, add git provenance capture (R12), run a short Breakout run on ithaca that writes a valid results.json row (correct timestamps, config hash, git_sha, git_dirty), and have `golds diagnose` read that row and return a health verdict. Every done-condition is machine-checkable locally. Uses Breakout (already validated) so M1a tests plumbing, not RL outcome. This is the walking skeleton: real training path + real results store + real diagnose, no credentials.

### M1b: Observability layer (needs M0 W&B key) [DONE 2026-07-18]

Verified live: a short Breakout run streamed to W&B (project golds, entity xbriggss) with the PPO health metrics plotting on the dashboard. Code committed 13f5fde, 120 tests green. Graceful degradation (config-flag gated, W&B failures never crash training). W&B video still deferred to M3.

### M2: Throughput baseline + retro readiness

Measure retro throughput (R1, R2). Build and unit-test the Sonic reward/completion/eval instrument (R3, R4, R5, R6), which is four requirements and G2 in full, decomposed into tasks at the M1 re-sync, not left as one line. Invariant checks running (R10). Formal Breakout 100-ep baseline recorded (R7).

### M3: Sonic campaign

Iterate Sonic training toward the north star (R11): shape reward, read W&B diagnostics, state a hypothesis per run, retrain. Budget 2-4 iterations.

## Tasks

### M1a (autonomous, local)

| # | Task | Done when |
|---|---|---|
| T1 | Fix results.json timestamp bug: capture started_at in `_on_training_start` (currently both started_at and completed_at are `datetime.now()` at end, callbacks.py:644-645). | Unit test asserts started_at < completed_at on a run; distinct wall-clock values. |
| T2 | Add git_sha + git_dirty to TrainingResult (schema.py) and capture them at run start (R12). No field exists today. | Unit test asserts a recorded result carries a valid SHA and correct dirty flag on a dirty vs clean tree. |
| T3 | Add `golds diagnose <run>` reading the local results.json row, predicate: best_eval_reward is None or <= epsilon => broken (R9). | `golds diagnose` on a normal row says healthy; on a stub row with best_eval_reward=None says broken. Machine-checkable. |
| T4 | Run the M1a Breakout skeleton end to end on ithaca. | results.json row valid (timestamps, hash, git_sha, git_dirty); `golds diagnose` reads it and returns healthy. No human eyeballing. |

### M1b (needs M0 W&B key)

| # | Task | Done when |
|---|---|---|
| T5 | Add W&B callback: config + git provenance + PPO health + GPU/FPS. Video deferred to when retro runs (M3). | A short Breakout run's W&B entry is confirmed via the wandb API (run exists, expected keys present) in a script, not by opening the dashboard. |

(M2 and M3 tasks are decomposed at each milestone's re-sync, since their scope depends on the M1 throughput and instrument reality.)

## Risks

| Risk | Source | Owner watches for |
|---|---|---|
| Retro throughput caps near 200 steps/sec despite the levers; Sonic runs stay multi-day. | memo RQ3, pre-mortem 1 | Xavier: R1 measurement result; if capped, re-plan M3 budget before starting it. |
| Sonic completion stalls at lucky-route variance, never robust to 80 percent. | Sonic literature, pre-mortem 2 | Xavier: completion rate plateau across 2-4 iterations; then extend/ship/kill. |
| Quarantine leaks (RND or self-play re-enabled to "help"). | pre-mortem 4 | Xavier: results attributable via config hash + git SHA; a leak shows as an unexplained config. |
| W&B / diagnose expands into a dashboard time sink. | pre-mortem 3 | Xavier: any diagnose work beyond the binary flag. |
| Completion detector wrong because Sonic x-position loops/wraps. | red-team finding 3 | Xavier: unit tests on hand-computed level-end cases must pass before trusting a completion number. |

## Kill criteria

- If M1a (local Breakout skeleton) is not running end to end within the first appetite third (~17 of 50 hands-on hours), kill or rescope.
- If the retro throughput baseline (R1) shows no path above ~200 steps/sec AND a single Sonic iteration would exceed 3 GPU-days, stop and re-plan M3 rather than launch multi-day runs blind.
- If after 4 Sonic reward-shaping iterations the completion rate is below 50 percent, kill the Sonic target and write a kill-log entry; do not open-endedly grind.
- If cumulative hands-on time hits the 50-hour appetite, stop and make an explicit extend/ship/kill decision.

## Open questions

| Question | Resolve when |
|---|---|
| What is the actual retro throughput ceiling on ithaca? | R1 probe, start of M2. |
| Does GHZ Act 1 need frame-skip or sticky-action tuning beyond the config defaults to complete? | First Sonic iteration in M3. |
| Should the deferred PPO-derivation notebook ever be built? | Optional; Xavier's call, no trigger. Parked per ADR-005. |

## Living spec

This spec is the prompt for build sessions. It is wired into the project CLAUDE.md; build sessions read it before structural changes.

Re-sync ritual, run at the end of every milestone before starting the next:
1. Diff implementation reality against this spec: requirements, architecture, ADR consequences.
2. Each divergence: update the spec (and ADR if a decision changed), or log it in Deviations below as accepted with one line of why.
3. Update "Last re-sync" in the header.

## Deviations

| Date | What diverged | Accepted because |
|---|---|---|
| 2026-07-19 | First Breakout baseline attempt ran at fps 17 and was killed after ~22h at 1.4M/15M steps: eval_episodes=100 was shared with the periodic in-training eval (every 50k steps), so eval ate ~99% of wall-clock. Fixed by decoupling periodic_eval_episodes (default 10) from eval_episodes (final 100), commit 74f4ef8. Relaunched clean at ~2000 steps/sec. | The R7 subagent had flagged this exact coupling; the lesson is to treat a subagent's specific overhead warning as a hard constraint on configs, not just a note. Now structurally prevented + regression-tested. |
| 2026-07-18 | ithaca's repo working tree was dirty during the M1a run (git_dirty=True). | Accepted for the skeleton (it validated the flag working). Ops item: clean or explain ithaca's working tree before M2/M3 real runs, since the "trusted result = reproducible from git SHA" principle needs git_dirty=False for a baseline to count. |
| 2026-07-18 | The probe's 177 steps/sec (grade A, the number driving ADR-002's "throughput is the top risk") was wall-clock-contaminated: ithaca suspended mid-run. A clean M1b run measured ~2200 steps/sec on the same Atari config, so real compute throughput is ~12x the probed figure. | Reframes G1/R1/R2 and ADR-002: Atari throughput is healthy; the "problem" was the box sleeping. Retro (Sonic) still needs its own measurement (R1) since it is heavier, but the throughput milestone is now likely a much smaller lift. Disable-sleep elevated to an M0 precondition. |

## Red team

Fresh-context subagent, 2026-07-18. Code-grounded (file:line). Load-bearing claims re-verified by hand before acting.

| Finding | Disposition |
|---|---|
| S1: M1 gated on human W&B signup + API key, verified by eyeballing a dashboard; a build agent can do neither. Verified: wandb not in pyproject.toml. | Fixed. W&B provisioning moved to M0 (Xavier precondition); W&B done-condition (T5) asserted via wandb API, not eyeball. |
| S2: M1 was not the thinnest slice; it front-loaded a SaaS onto the first milestone when a fully-local skeleton exists (results write path is local: store.add_result). | Fixed. Split into M1a (local skeleton, zero deps) and M1b (W&B layer). |
| S3: T2 hid 5-6 sub-tasks behind one name; git SHA is captured nowhere and needs a schema change. Verified: no git_sha/rev-parse in codebase. | Fixed. git provenance is its own task (T2/R12); W&B logging is T5; PPO-health overlap with R10 noted. |
| S4: reproducibility claim ("trusted = reproducible from git SHA") is decorative during dirty-tree iteration. | Fixed. R12 adds git_sha + git_dirty; a dirty-tree result is marked, not silently trusted. |
| S5: T1 conflated a trivial timestamp fix with unbuilt eval-field population. Verified: both timestamps are datetime.now() at end (callbacks.py:644-645); nothing populates eval_100ep. | Fixed. Timestamp fix is M1a/T1; eval-field population moved to R7/M2 where a 100-ep run exists. |
| S6: `golds diagnose` had no defined input or broken-run predicate. | Fixed. R9/T3 pin it to the local results.json row with predicate best_eval_reward None or <= epsilon. |
| S7: M2's "build the Sonic completion instrument" scoped as one line despite ADR-004 calling it net-new. | Accepted with flag. M2 decomposition happens at the M1 re-sync; spec now states G2 is four requirements and must not stay a one-liner. R3-R5 unit tests are ROM-free (test_wrappers.py pattern exists). |
| S8: Sonic ROM not obtainable by a build agent. | Fixed. ROM-on-ithaca is an M0 Xavier precondition; R3-R5 tests noted ROM-free. |
| S9: ithaca access is Tailscale-bound, undocumented in the project. | Fixed. Added to repo CLAUDE.md (VPN-must-be-up + creds) and M0. |
| Clean axes: traceability (every R traces to a goal, every goal has an R); kill criteria have real firing numbers. R8 was a compound requirement, split via T5. | Accepted. |

## Gate 4 result

GO to build, 2026-07-18. Inception complete. M1 restructured to M1a (local, autonomous) + M1b (W&B) after red-team; all 9 findings dispositioned. Spec is the prompt for build sessions; build starts at M1a.
