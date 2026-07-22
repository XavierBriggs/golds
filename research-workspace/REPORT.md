# Iteration-3 Decision Memo: Getting Our PPO Agent Past the GHZ Act 1 Stall

Date: 2026-07-21. Author: research synthesis (3 parallel subagents + verification).

## Summary (the direct answer)

Our agent stalls at ~38% (deterministic) / ~51% (stochastic) of Green Hill Zone Act 1, 0% completion after two 20M-step runs. **The evidence says our problem is a config/tuning gap against the well-established Sonic PPO recipe — not a need for exotic exploration (RND) or more compute.** This corrects two things I told you earlier: (1) 20M is NOT too short for a single fixed level, and (2) RND is NOT the evidence-backed first move. The highest-leverage, cheapest iteration is to align our config with what actually works on Sonic, with behavioral cloning from a demo as the stronger fallback.

## What the literature actually shows

- **Standard single-level PPO does not reliably complete Sonic levels without care.** The Gotta Learn Fast benchmark: per-level PPO from scratch scored ~1489 of the ~9000 needed to finish; even joint-trained PPO on training levels reached only ~5500 (arXiv:1804.03720, primary). Contest winners' top score was 4692/10000 (openai.com/index/retro-contest-results).
- **Training scale is NOT our issue.** For a single fixed level, practitioners succeeded with 5M steps (uvipen demo GIFs) to ~12h on a laptop (torinrettig). The 200M-frame figure was for training a *generalist across many levels*. Our 20M for one level is adequate-to-generous. (github.com/uvipen/Sonic-PPO-pytorch; github.com/torinrettig/Sonic-Deep-Reinforcement-Learning)
- **RND/curiosity is weakly supported for our exact problem.** The RND paper (arXiv:1810.12894) never tested Sonic. Curiosity was tested on Sonic in the Large-Scale Curiosity study but as a sole reward, not a bonus. Two independent practitioners tried curiosity on Sonic and abandoned it for no improvement (flyyufelix). One repo (jakegrigsby/supersonic) pairs PPO+RND on GHZ1 but published no results. Verdict: RND is not a confident bet for a localized stuck point.
- **The fix that worked for the closest analog was better tuning, not RND.** torinrettig's vanilla PPO/A2C stalled on the loop + spike field for 10+ hours; switching to Borghi's 3rd-place *tuned* PPO variant got the agent past the obstacles to a claimed completion (screenshot-verified, so treat as plausible not proven).
- **Strongest completion evidence overall = behavioral cloning from a human demo.** AurelianTactics cleared 8 of 11 Sonic levels by shaping reward from human demonstrations; precise/patient maneuvers are near-impossible to hit via pure exploration.

## Our config vs the proven Sonic recipe

| Aspect | Ours | Proven Sonic recipe | Evidence | Matters? |
|---|---|---|---|---|
| Reward scale | 0.1 | 0.01 | "incredibly important, affects performance drastically" (sonic_util.py, primary); "crucial to get PPO to work" (flyyufelix) | Likely yes |
| Frame skip | fixed 4 | stochastic {2,3,4} | Sonic benchmark standard; timing noise breaks deterministic freeze/oscillation | Likely yes — targets our exact failure |
| Progress reward | delta-max(x) | delta-max(x) | matches — the most-cited un-sticking mechanism | Already right |
| Anti-stall | none | terminate on prolonged idle | uvipen terminates when last N actions identical | Probably helps |
| Action set | platformer (incl. DOWN+B) | 7-combo incl. RIGHT+B | DOWN+B is a DEAD action in Sonic 1 (no spin-dash); run+jump (RIGHT+B) is what clears GHZ | Verify RIGHT+B present |
| Sticky/eval | sticky 0.25 train, deterministic eval | benchmark uses stochastic frameskip as its noise | pure-deterministic argmax freezing is partly an eval artifact | Reconsider eval |
| Training steps | 20M | 5-20M adequate for one level | not the bottleneck | No |

## Ranked recommendation for iteration 3

**Tier 1 — config alignment (do first; ~1 run, ~3 GPU-hours; moderate probability of clearing).**
One run matching the proven recipe:
1. Switch fixed frame_skip 4 to stochastic {2,3,4} (StochasticFrameSkip). Directly attacks the deterministic-freeze via timing noise. Highest-signal single change.
2. Reward scale 0.1 to 0.01 (or confirm PPO reward-normalization is on; literature strongly favors 0.01).
3. Add anti-stall episode termination on prolonged zero max-x progress.
4. Verify the "platformer" action set includes RIGHT+B (run-jump); note DOWN+B is inert in Sonic 1 (harmless but useless).
5. Keep delta-max(x) and completion detection.
Also: report the completion metric under the stochastic (benchmark-style) policy alongside deterministic — pure-deterministic argmax is a harsher, somewhat artificial test that the benchmark itself does not use.

**Tier 2 — behavioral cloning warm-start (if Tier 1 falls short; strongest completion evidence; more infra).**
Record a human (or scripted) demo of GHZ Act 1, pre-train the policy by imitation, then PPO fine-tune. This is the best-evidenced route to clearing stuck points, at the cost of a demo-recording + BC pipeline we do not have yet.

**Tier 3 — RND (deprioritize; was my initial instinct, now downgraded).**
No direct Sonic evidence, two abandoned curiosity, RND paper never tested Sonic. Pursue only if Tier 1+2 fail and the stall looks like genuine hard-exploration — which it does not; it is a localized obstacle better addressed by timing noise or demonstrations.

## Confidence and caveats

- **High confidence:** 20M is enough for one level; reward-scale 0.01 and delta-max(x) are standard; RND is not well-supported on Sonic; Sonic 1 has no spin-dash (verified across multiple sources).
- **Medium confidence:** stochastic frameskip will meaningfully help our freeze (well-motivated and standard, but no controlled A/B in sources); reward-scale change matters at our 0.1 (strongly asserted, not A/B-proven).
- **Lower confidence / gaps:** No source shows a fully-verified end-to-end GHZ Act 1 completion (video + logged bonus); the strongest completion claims are under-documented (screenshot/GIF). No source proves a specific reward term (vs tuning) causes completion. Our "~40% stall" is our own measurement; the literature describes stalls by obstacle (loop, spring, spike field), not percent.

## Sources
- Nichol et al, "Gotta Learn Fast" arXiv:1804.03720
- OpenAI Retro Contest + results: openai.com/index/retro-contest/, openai.com/index/retro-contest-results/
- Burda et al RND arXiv:1810.12894; Large-Scale Curiosity arXiv:1808.04355 / pathak22.github.io/large-scale-curiosity/
- openai/retro-baselines sonic_util.py (action set, RewardScaler 0.01, AllowBacktracking)
- flyyufelix.github.io/2018/06/11/sonic-rl.html (reward 0.01, delta-max-x, curiosity abandoned)
- medium.com/aureliantactics (human-demo cleared 8/11; anti-stall; reward spec)
- github.com/torinrettig/Sonic-Deep-Reinforcement-Learning (stall then tuned-variant fix)
- github.com/uvipen/Sonic-PPO-pytorch (5M-step default, anti-stall termination)
- Sonic 1 no-spindash: sonic.fandom.com/wiki/Spin_Dash, soniczone0.com/games/sonic1/greenhill/
