# CLAUDE.md — GOLDS

GOLDS trains RL agents (SB3 PPO) on classic games and benchmarks them. Renovation in progress; the design is spec-driven.

## Active spec (read before structural changes)

This project is under an approved inception spec. **Before any structural change, read `docs/inception/spec.md` and the relevant ADRs in `docs/inception/adr/`.** The spec is LIVING and is the prompt for build sessions.

- Spec: `docs/inception/spec.md` (goals, requirements R1-R11, milestones M1-M3, kill criteria)
- Decisions: `docs/inception/adr/001-005`
- Probe evidence + Sonic sources: `docs/inception/02-research-memo.md`

**Re-sync ritual:** at the end of every milestone, diff implementation against the spec, update it (and any ADR whose decision changed) or log the divergence in the spec's Deviations table, and date-stamp the header. The ritual lives in the spec itself.

Current north star: Sonic GHZ Act 1 completion >= 80% over a 100-episode deterministic eval.

## Training box: ithaca

Runs execute on **ithaca** (SSH host alias `ithaca`, via Tailscale). Ryzen 9 5900X (12c/24t), 64GB RAM, RTX 3080 10GB, Ubuntu 24.04, Python 3.12. No tmux (use `nohup` for long runs). uv at `~/.local/bin`. Repo at `~/Development/golds`.

- **Access requires the Tailscale VPN up** (ithaca is at Tailscale IP 100.116.147.65, host alias `ithaca`, user `xbriggs`). A build agent cannot reach ithaca without it.
- Tailscale SSH auth expires periodically; a re-auth link must be opened in a browser. Normal, not a failure.
- W&B and the Sonic Genesis ROM are human-provisioned preconditions (spec M0): `WANDB_API_KEY` on ithaca, and `golds rom import` for the non-redistributable Sonic ROM.
- Root partition (/) is separate from /home and has filled via Timeshift snapshots before. Check disk before long runs.
- Long runs: `nohup uv run golds train run -c <config> > run.log 2>&1 &`, then tail the log.

## Quarantined (do NOT build on until the Sonic gate passes)

RND exploration, self-play/Elo, MK2/SF2 fighting configs, and all game configs other than Breakout and Sonic are unvalidated. Do not enable them to "help" a Sonic run; a leak corrupts result attribution.

## Verified facts (probe, 2026-07-18)

The existing PPO stack trains Breakout to ~275 final / 391.5 peak eval (published range) on ithaca CUDA, zero crashes over 15M steps. The architecture is sound; correctness is not the bottleneck, iteration speed is. SubprocVecEnv is already the env default. The Sonic `PlatformerRewardWrapper` is currently raw delta-x with no completion detector (both must be built, per ADR-004).
