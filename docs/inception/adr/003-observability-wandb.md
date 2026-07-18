<!-- style: direct sentences, no em dashes, no filler -->
# ADR-003: Weights & Biases as primary observability, with a diagnose command

Date: 2026-07-18
Status: accepted

## Context

"I can't see what's happening" was one of the four core pains. The probe confirmed it in a small way: results.json wrote identical started_at and completed_at timestamps and left eval_100ep and human_normalized_score null, so the real signal lived only in a raw log. Xavier explicitly asked for more than TensorBoard. Runs are multi-hour to multi-day, so failures must be legible without watching a terminal. The repo already has a Telegram notifier and TensorBoard logging.

## Options

- TensorBoard only: local, no run-comparison across machines, no hosted video, no plain-English verdict. Already present and insufficient. Rejected as primary.
- Build a custom dashboard: reinvents W&B; violates the "shrink, do not add surface" principle; becomes a time sink. Rejected.
- Weights & Biases (free tier) as primary, TensorBoard as local fallback, plus a `golds diagnose` command: hosted metrics/video/run-comparison for free, minimal code via an SB3 callback, and a scriptable health verdict. Chosen.

## Decision

Add a W&B callback that wraps the existing SB3 callback chain and logs config + git SHA, PPO health metrics, game-progress metrics (Sonic max-x, completion rate), periodic video, and GPU/FPS system stats. Keep TensorBoard as local fallback. Add `golds diagnose <run>` that reads a run's logs and prints a plain-English health verdict. Fix the results.json timestamp and null-metric defects as part of this work. Success is binary: a broken (zero-reward) run is flagged by diagnose, and any two runs are comparable in W&B.

## Video capture constraint

Retro cannot run a second emulator in-process (single-emulator-per-process, commit 9324e39 in git history forced video recording into a subprocess). Periodic video during a SubprocVecEnv retro run must reuse that existing video-subprocess path, not spin up an in-process render env. ADR-002's parallelism and this ADR's video logging contend for the same scarce resource; the W&B video callback must go through the subprocess recorder or the "periodic video" criterion will fail mid-campaign.

## Consequences

- Easier: failures become legible remotely; run-to-run comparison is a first-class view; the throughput and Sonic goals are measurable from the same place.
- Harder: adds a W&B account dependency and an API key to manage on ithaca; video logging must route through the retro video-subprocess path.
- Foreclosed: nothing; TensorBoard fallback means a W&B outage does not stop training. `golds diagnose` is capped at the binary broken-run flag, not grown into a dashboard.
