<!-- style: direct sentences, no em dashes, no filler -->
# ADR-004: Sonic reward uses delta-max(x), eval reports unclipped completion

Date: 2026-07-18
Status: accepted

## Context

Prior work, cited in the research memo (Gotta Learn Fast / OpenAI Retro Contest, Felix Yu's practitioner writeup), is consistent on two Sonic-specific facts (grade B/C). First, the dominant failure mode is the agent getting stuck mid-level and needing to backtrack, and the standard fix is rewarding delta max(x) (furthest-right progress) rather than raw delta x, so the agent can explore backward without being punished. Second, the Retro Contest reward normalization is horizontal progress scaled to ~9000 at the level end plus a completion bonus. Separately, the probe showed Breakout eval was measured under the clipped reward regime; that is harmless for Breakout (clipped per-brick reward equals raw score) but wrong for a variable-progress game like Sonic, where the north star is level completion, not clipped reward.

## Options

- Raw delta-x reward: punishes backtracking; agent fixates and stalls at traps. Rejected per literature.
- Clipped reward for Sonic eval: decouples the eval number from actual level progress; makes the 80 percent completion gate unmeasurable. Rejected.
- delta-max(x) progress + completion bonus for training, and a raw-progress eval that reports actual completion: matches the proven recipe and makes the north star directly measurable. Chosen.

## Decision

The Sonic retro reward wrapper uses delta max(x) progress plus a level-completion bonus. Evaluation runs on a raw-progress path that reports unclipped completion (did the agent reach the level-end x), and the north star is the completion rate over 100 deterministic eval episodes. Reward and progress wrappers carry unit tests with hand-computed expected values.

## Consequences

- Easier: the north star (completion rate) is measured directly; the known backtracking trap is mitigated by construction.
- Harder: this is net-new code, not a tweak. Today's `PlatformerRewardWrapper` computes raw `scale * (x_new - x_old)` (the rejected option); delta-max(x), the level-end x threshold per level, the completion detector, and the raw-progress eval path all have to be built and unit-tested. Sonic x-position can loop, so robust end-detection is fiddly. This is large enough to be its own build goal, not a clause here.
- Foreclosed: nothing; the delta-max(x) + completion pattern generalizes to other retro platformers when the trust ladder reaches them.
