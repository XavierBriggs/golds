<!-- style: direct sentences, no em dashes, no filler -->
# Framing: GOLDS Verify & Renovate

Date: 2026-07-16
Tier: STANDARD, classifier score 4/8, overrides: Q4=2 (genuinely unclear) makes the probe mandatory
Appetite: 50 hands-on hours. Includes this inception run. GPU wall-clock does not count. Hitting the limit forces an explicit extend / ship / kill decision.
Phase timeboxes: P0 0.5h / P1+P2 (probe) 6h hands-on across one day / P3 2h / P4 2h

## Problem statement

GOLDS has never produced a trusted training run. Agents fail to complete levels, runs crash on infrastructure bugs, no observability exists to explain failures, and Xavier does not trust the PPO math or reward logic. Evidence: one recorded run ever (a 100k-step Pong smoke test, 50x too short to learn), training on Mac MPS while an RTX 3080 box (ithaca) sits idle with a 100% full root disk. Two prior overhauls (v2 2026-03, v3 2026-06) rewrote the harness without fixing any of this. If nothing changes, the project keeps burning time on machinery that has never been validated end to end.

## User

Xavier only. Personal RL project. No other users, no revenue.

## Non-goals

- No greenfield rewrite. The existing repo is renovated, not replaced. Decided 2026-07-16, see docs/superpowers/specs/2026-07-16-verify-renovate-design.md.
- No work on quarantined subsystems until the Sonic gate passes: RND, self-play/Elo, MK2/SF2, and the 14 game configs other than Breakout and Sonic.
- No new algorithms (no RecurrentPPO tuning, no exploration research) before a Breakout baseline reproduces and Sonic GHZ Act 1 is beaten.
- No multi-machine or cloud training infrastructure. One box, ithaca.

## Working backwards (optional)

GOLDS trains a PPO agent that completes Green Hill Zone Act 1 in at least 80% of eval episodes, with video proof and a W&B dashboard showing exactly how it learned. Every number in results.json is reproducible from a config hash and git SHA. When a run fails, one command explains why.

## Probe check

Would a one-day throwaway prototype answer the core question faster than research? Yes. Q4 scored 2, so the probe is mandatory, and the core risk is "does this stack learn at all when given real hardware and a real budget", which is measurable, not researchable.
The one question the probe must answer: does the existing GOLDS stack, deployed on ithaca (CUDA), train PPO Breakout to a mean eval reward of at least 250 over 100 episodes within roughly 10M timesteps? Prerequisite ops (disk cleanup, deploy) count as probe work. Timebox: one day, at most 6 hands-on hours; the GPU run itself may finish overnight.

## Gate 0 result

GO, 2026-07-16. Probe path activated: Breakout baseline on ithaca replaces phases 1 and 2.
