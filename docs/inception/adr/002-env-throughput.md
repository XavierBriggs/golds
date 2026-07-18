<!-- style: direct sentences, no em dashes, no filler -->
# ADR-002: Measure retro throughput first, then raise it; do not assume a target

Date: 2026-07-18
Status: accepted (revised after red-team)

## Context

The probe measured 177 aggregate timesteps/sec on Atari with n_envs=16 and GPU utilization near 6 percent: CPU-bound on environment stepping. A red-team check of the code corrected the original framing of this ADR. SubprocVecEnv is already the default in factory.py and both makers (use_subproc defaults to True), and the 177 number was already measured under SubprocVecEnv. So "switch to SubprocVecEnv" was never an available lever; it is already applied. The box is a Ryzen 9 5900X: 12 physical cores, 24 threads, so n_envs=16 already exceeds physical cores and the headroom from 16 to 24 is hyperthread headroom with steep diminishing returns on a CPU-bound stepping loop. Git history also shows retro has a single-emulator-per-process constraint (commit 9324e39), so retro parallelism is a known risk, and retro stepping is heavier than Atari, so retro throughput will likely be below 177, not above it. There is no measured retro throughput number yet. The original 400 steps/s and 40 percent GPU targets were invented, not derived, and were mutually incoherent (2.3x steps cannot imply 7x GPU work).

## Options

- Keep the invented 400/40 percent target: grade-E numbers dressed as a hard gate; the Atari-to-retro extrapolation runs the wrong direction. Rejected.
- Assume throughput is already maxed and skip the milestone: gives up real levers (n_envs to 24, frameskip, preprocessing cost) without measuring them. Rejected.
- Measure a retro baseline first, then set an evidence-based target as a multiple of it: honest, and it tells us early whether the Sonic iteration budget is even feasible. Chosen.

## Decision

Before any Sonic training run, run a short retro throughput probe (a few hundred k timesteps of the Sonic config at n_envs=24, off-appetite GPU wall-clock) and record steps/sec and GPU utilization as grade-A. Then set the throughput target as a multiple of that measured retro baseline, not of the Atari number. Pull the remaining real levers in priority order and measure each: n_envs toward 24, frameskip, and per-env preprocessing cost. If retro tops out near 200 steps/sec despite these, that is a finding that forces re-planning the Sonic iteration budget, not a number to force.

## Consequences

- Easier: the throughput decision rests on a retro measurement instead of an Atari extrapolation; the Sonic feasibility question gets answered in GPU-hours, before GPU-days are spent.
- Harder: adds a measure-first step before the first real Sonic run; the single-emulator constraint may cap retro parallelism below hopes.
- Foreclosed: nothing permanent; a low ceiling just means fewer, longer runs and tighter hypothesis discipline, logged as a throughput finding rather than hidden.
