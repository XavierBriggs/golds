# FINDINGS

## Subagent 1: Retro Contest / Gotta Learn Fast / winners (strong, mostly primary)

Completion = score ~9000-10000 (reward normalized to 9000 at level end + 1000 completion bonus decaying by 4500 steps). arXiv:1804.03720 (primary).

Baseline scores (Table 1, aggregate over test levels, 1M timesteps/level):
- PPO per-level from scratch: 1488.8 (of ~9000 to complete) — standard PPO does NOT complete.
- Joint PPO: 3127.9 (best baseline). Rainbow: 2748.6. JERK (scripted): 1904. Human: 7438.
- Joint PPO zero-shot on TRAINING levels: ~5500 (Fig 4) — still short of 9000 even on training levels.

Training budgets:
- Per-level fine-tune: 1M timesteps. Joint PPO meta-training: "hundreds of millions of timesteps", 188 workers, overfits after ~50M timesteps.
- Felix Yu (5th, score 5522): 200M frames joint training on 47 levels + per-zone expert pretraining → 5525. Home rig 12 CPU + 1 GTX 1080. https://flyyufelix.github.io/2018/06/11/sonic-rl.html

Winners' techniques (openai.com/index/retro-contest-results, reputable; top score 4692/10000):
- 1st "Dharmaraja": Joint PPO variant + RGB (not grayscale) + augmented action space + REWARD FOR VISITING NEW STATES (exploration bonus).
- 3rd "aborg": Joint PPO + more training levels + diff architecture + LR-tuned for stable fast fine-tuning (first ~150K steps unstable).
- Dbrain (4th): PPO + count-based exploration bonuses (pixel similarity + x-coord frequency) + test-time fine-tune + best-of-N policy selection.
- Recurring: joint/transfer PPO, exploration/novelty bonuses, augmented action+reward, LR stabilization.

Reward: delta-max-x (not raw delta-x) "gives a sizable performance boost" (§4.2). [matches our setup]

KEY IMPLICATIONS for us:
- Standard single-level PPO does not complete Sonic levels; extra technique consistently needed.
- Exploration/novelty bonuses were in the top solutions (supports RND).
- Our 20M timesteps (=80M frames) is SHORT vs the 200M+ frames these used (and even those didn't fully complete).
- No hard sourced frame-count for completing GHZ1 as a single training level (open question in the paper).

## Subagent 2: action space / freeze failure / RND-curiosity (strong, primary+reputable)

Action set (primary, openai/retro-baselines sonic_util.py): 7 actions [LEFT,RIGHT,LEFT+DOWN,RIGHT+DOWN,DOWN,DOWN+B,B]. B=jump, DOWN=crouch, DOWN+B=spindash charge.
- CAVEAT (weak/unverified but decisive): Sonic 1 / GHZ has NO spin-dash — DOWN+B is effectively a no-op in Sonic 1 (spindash is Sonic 2/3 only). GHZ obstacle-clearing = RIGHT+B (run-jump) + backtracking, NOT spindash. VERIFY THIS.

Preprocessing (primary, sonic_util.py):
- RewardScaler x0.01 "incredibly important, affects performance drastically." [WE USE scale 0.1 — 10x larger]
- AllowBacktracking / delta-max-x = most-cited un-sticking mechanism. [WE ALREADY USE delta_max_x — good]
- Stochastic frameskip {2,3,4} is the Sonic benchmark standard, NOT fixed 4. Injects timing noise that breaks freeze/oscillation loops. [WE USE FIXED frame_skip 4 + sticky 0.25]

RND / curiosity (THE key question):
- RND paper (arXiv:1810.12894) did NOT test Sonic (only 6 hard-exploration Atari). No direct RND-on-Sonic evidence.
- Curiosity tested on Sonic in Large-Scale Curiosity (arXiv:1808.04355): pure-curiosity aligns with extrinsic reward. But that's curiosity-as-sole-reward, not as-bonus.
- TWO practitioners AGAINST: flyyufelix "attempted curiosity-driven exploration... no noticeable improvement... abandoned it."
- CONCLUSION: curiosity/RND is NOT a reliable fix for a LOCALIZED stuck point. Weak evidence for our case.

Freeze-at-obstacle fixes, ranked by evidence:
1. AllowBacktracking/delta-max-x [we have it]
2. Stochastic frameskip {2,3,4} + sticky — timing noise breaks deterministic oscillation [we use fixed skip]
3. Entropy bonus (non-trivial early)
4. BEHAVIORAL CLONING / human-demo warm-start: AurelianTactics cleared "8 of 11 levels" this way; precise/patient maneuvers near-impossible via random exploration. STRONGEST fix for stuck points. https://medium.com/aureliantactics/attempting-to-beat-sonic-the-hedgehog-with-reinforcement-learning-6ca32d4fd86e
5. Curiosity/RND: weak/inconsistent for localized stuck points.

RND config template if used (primary, Burda): intrinsic reward normalized by std of returns; obs normalized clip [-5,5]; dual value heads gamma_I=0.99 gamma_E=0.999; sticky 0.25.

## Subagent 3: practitioner completions / reward shaping / scale (reputable)

Who "completed" GHZ1 (all under-verified):
- torinrettig: claims full GHZ1 clear, PPO using Borghi's 3rd-place variant, ~12h laptop, ~3000 iters. Evidence = screenshot only. STRONGEST match to "stuck then got past": vanilla PPO/A2C stuck on loop+spike field "10+h no improvement" -> switching to Borghi's TUNED variant fixed it. https://github.com/torinrettig/Sonic-Deep-Reinforcement-Learning
- uvipen/Sonic-PPO-pytorch: GHZ Act1/2 demo GIFs, PPO, default num_global_steps=5e6 (5M), lr1e-4, entropy0.01. Completion implied by GIF, unverified in text. Also: anti-stall = terminate when last N actions identical.
- jakegrigsby/supersonic: explicit GHZ1 target, PPO + RND, but NO published results.
- Edward Chen: none completed; PPO stuck at spring; Rainbow sometimes past spring.
- NO fully-verified end-to-end completion (video+logged bonus) found anywhere.

Reward: Retro Contest = 9000 at end + 1000 bonus decaying to 0 at 4500 steps, 4500-step episode cap. Reward x0.01 crucial for PPO stability [we use 0.1]. delta-max-x un-sticking [we have]. Anti-stall termination on idle helps.

TRAINING SCALE (resolves subagent-1 tension): for a SINGLE FIXED LEVEL, 20M is ADEQUATE-TO-GENEROUS, not short. uvipen 5M default sufficed; torinrettig ~12h laptop. The 200M frames was for a GENERALIST across many levels. => MORE TRAINING is likely NOT our fix. Tuning/architecture matters more.

STALL FIX (closest analog): torinrettig's fix was a BETTER-TUNED PPO reference impl (Borghi's), not RND/curiosity.
