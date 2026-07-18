<!-- style: direct sentences, no em dashes, no filler -->
# ADR-005: Keep cheap correctness checks, defer the full PPO-derivation notebook

Date: 2026-07-18
Status: accepted

## Context

In the original brainstorm Xavier said the PPO math and model logic should be rigorously validated, and the design spec included a LaTeX notebook that re-derives the PPO clipped objective and GAE in numpy and cross-checks them against SB3's computed losses. That notebook's primary justification was distrust of the math. The probe dissolved that justification with grade-A evidence: the stack reached published Breakout scores on the first real run, which is only possible if the objective, advantage computation, and preprocessing are correct. So the notebook's value dropped from "verify the math is right" to "help Xavier understand the math", which is real but secondary to the north star (agents that beat games), and Xavier chose that north star over the learning-vehicle framing.

## Options

- Build the full derivation notebook in the critical path: high effort, and its correctness-verification purpose is now redundant. Rejected for the critical path.
- Drop all math validation: loses cheap checks that catch real wrapper and normalization bugs the probe's single curve would not localize. Rejected.
- Keep the cheap checks (wrapper unit tests + live PPO invariant checks), defer the derivation notebook to an optional learning task: retains bug-catching value at low cost, and preserves the notebook as an explicit later option rather than silently cutting what Xavier asked for. Chosen.

## Decision

In the critical path, keep only: wrapper math unit tests with hand-computed expected values, and live PPO invariant checks (clip fraction in (0,0.3), bounded approx-KL, rising explained variance, zero-mean/unit-std normalized advantages) asserted on a real run. Defer the full LaTeX PPO-derivation notebook to an optional, out-of-critical-path learning task Xavier can pick up when he wants the understanding for its own sake.

## Consequences

- Easier: correctness stays asserted where bugs actually hide (wrappers, normalization) at low cost; appetite is spent on Sonic instead.
- Harder: if a subtle PPO-core bug ever does appear, there is no numpy cross-check on record to localize it; the invariant checks are the safety net instead.
- Foreclosed: nothing; the notebook remains a documented option, explicitly deferred, not cancelled.
