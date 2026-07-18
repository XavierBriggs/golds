<!-- style: direct sentences, no em dashes, no filler -->
# ADR-001: Renovate the existing repo, do not greenfield

Date: 2026-07-18
Status: accepted

## Context

GOLDS had never produced a trusted training run, and Xavier considered a full greenfield rewrite. Two prior overhauls (v2 2026-03, v3 2026-06) had already rewritten the harness without fixing the core pains, which argued against a third rewrite. The probe settled it with grade-A evidence: the existing code trained PPO Breakout to ~275 final / 391.5 peak eval (published range ~274-398) with zero crashes over 15M steps on ithaca. The failures Xavier attributed to the code were environmental (Mac MPS, full disk).

## Options

- Full greenfield rewrite: discards code the probe just proved correct; costs weeks before training resumes; does not itself fix RL outcomes. Rejected.
- Strangler hybrid (rebuild core loop inside repo): unnecessary now that the core loop is validated as-is. Held only as an escape hatch if renovation is blocked.
- Renovate in place: keep the validated path, quarantine the unvalidated parts, fix the measured problems. Chosen.

## Decision

Renovate the existing repository. Keep the config -> factory -> PPO -> eval -> results path unchanged. Fix the measured problems (throughput, observability, Sonic reward). Quarantine unvalidated subsystems rather than deleting or rewriting them.

## Consequences

- Easier: training resumes immediately; every fix builds on a known-good baseline.
- Harder: must resist the urge to "clean up" quarantined code that is not on the critical path.
- Foreclosed: the clean-slate architecture a greenfield would have allowed. Accepted, because the current architecture is not the problem.
