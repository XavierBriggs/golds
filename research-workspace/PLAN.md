# Research Plan: Completing Sonic GHZ Act 1 with RL

## Query (restated)
How to train a PPO/SB3 agent on stable-retro to reliably COMPLETE Sonic GHZ Act 1, overcoming the failure where the deterministic agent freezes at an obstacle around 38-51% of the level. Our setup: delta-max(x) reward, frame_skip 4, sticky 0.25, 20M timesteps x2 runs, both 0% completion. RND available but quarantined.

## Query type
Depth-first (one core question, multiple angles: benchmarks, failure-mode/exploration, reward/scale).

## Subagent tasks
1. Retro Contest + Gotta Learn Fast + winners: techniques that COMPLETED Sonic training levels, official baselines, training scale/frames. (Q1, Q4)
2. Deterministic-freeze failure + action-space (spin-dash/GHZ obstacles) + exploration bonuses (RND/curiosity) on Sonic + sticky/frameskip. (Q2, Q3)
3. Practitioner single-level GHZ1 completions (blogs/github) + reward shaping (anti-stall, curriculum) + realistic training scale. (Q4, Q5)

## Answer format
Decision-grade memo with graded evidence + ranked recommendation for iteration 3 (RND vs reward shaping vs action-set vs scale), with GPU cost and rough probability.
