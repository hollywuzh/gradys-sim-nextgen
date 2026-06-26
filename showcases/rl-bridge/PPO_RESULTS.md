# RLlib PPO Training Results

This note records the first PPO run that reliably beats the `nearest` baseline
in this GrADyS-SIM RL bridge.

## Key Fixes

- Added RLlib action masking for the shared-policy PPO path.
- Masked invalid candidate actions and forced busy UAVs to use the no-op action.
- Added a custom old-stack Torch model, `gradys_action_mask_model`, so PPO
  learns from the raw observation vector while RLlib receives masked logits.
- Added workload range arguments to the PPO and baseline entry points:
  `--deadline-range`, `--compute-demand-range`, and `--data-size-range`.

## Why The 12s Default Is Hard For PPO To Beat

The small default PPO smoke scenario uses:

```text
episode_duration = 12s
deadline_range   = 20-60s
```

In that setting, evaluated episodes usually have zero deadline misses. The task
therefore behaves mostly like "complete as many nearby requests as possible",
where `nearest` is a strong handcrafted heuristic. Action masking improved PPO
from about `10.66` to `13.24` average reward on the 20-episode evaluation, but
`nearest` remained ahead at `15.82`.

## First PPO-Greater-Than-Nearest Scenario

The first stable win uses tighter deadlines so the controller must trade off
service success, missed deadlines, queue pressure, and movement cost:

```bash
python showcases/rl-bridge/main_rllib_ppo_train.py \
  --iterations 100 \
  --train-batch-size 512 \
  --minibatch-size 128 \
  --num-epochs 4 \
  --rollout-fragment-length 64 \
  --eval-episodes 20 \
  --baseline-policies random,nearest,edf,slo-risk \
  --num-uavs 2 \
  --num-devices 8 \
  --episode-duration 30 \
  --candidate-limit 3 \
  --task-arrival-probability 0.10 \
  --deadline-range 10 30 \
  --compute-demand-range 20 120 \
  --output-dir outputs/rllib_ppo_train_100iter_eval20_tight_deadline \
  --quiet \
  --checkpoint
```

The saved checkpoint was then re-evaluated for 50 deterministic episodes:

```bash
python showcases/rl-bridge/main_rllib_ppo_evaluate.py \
  --run-dir outputs/rllib_ppo_train_100iter_eval20_tight_deadline \
  --eval-episodes 50 \
  --baseline-policies random,nearest,edf,slo-risk \
  --quiet
```

## 50-Episode Deterministic Evaluation

Artifacts:

```text
outputs/rllib_ppo_train_100iter_eval20_tight_deadline/comparison_deterministic_summary.csv
outputs/rllib_ppo_train_100iter_eval20_tight_deadline/comparison_deterministic_episodes.csv
outputs/rllib_ppo_train_100iter_eval20_tight_deadline/checkpoints/
```

Summary:

| policy | episodes | avg reward | success | miss | hits | misses | energy | meanQ | maxQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 50 | -12.29 | 0.375 | 0.625 | 2.58 | 4.56 | 524.69 | 9.24 | 15.92 |
| nearest | 50 | -8.91 | 0.435 | 0.565 | 3.24 | 4.32 | 572.69 | 8.97 | 15.54 |
| edf | 50 | -35.34 | 0.358 | 0.642 | 2.36 | 4.80 | 586.13 | 9.26 | 16.16 |
| slo-risk | 50 | -33.22 | 0.345 | 0.655 | 2.26 | 4.92 | 588.38 | 9.32 | 16.18 |
| rllib-ppo | 50 | 1.19 | 0.399 | 0.601 | 3.02 | 4.70 | 560.45 | 8.98 | 15.50 |

Interpretation:

- PPO beats `nearest` by `10.10` reward points on this 50-episode evaluation.
- PPO does not simply maximize raw hit count; `nearest` still has slightly more
  hits, but PPO spends less energy and keeps similar queue pressure.
- This is the first usable "PPO > nearest" milestone. It should be treated as a
  scenario-specific result, not yet a universal superiority claim.

## Retry Across Evaluation Seeds

The saved checkpoint was retried with three deterministic 50-episode evaluation
sets. This checks evaluation-seed robustness for the checkpoint; it is not yet a
multi-training-seed claim.

Artifact:

```text
outputs/rllib_ppo_train_100iter_eval20_tight_deadline_multi_seed_summary.csv
```

| policy | seed 1007 | seed 2007 | seed 3007 | mean reward |
| --- | ---: | ---: | ---: | ---: |
| random | -12.29 | -24.31 | -6.63 | -14.41 |
| nearest | -8.91 | -17.88 | -3.84 | -10.21 |
| edf | -35.34 | -39.71 | -30.36 | -35.14 |
| slo-risk | -33.22 | -36.99 | -29.19 | -33.13 |
| rllib-ppo | 1.19 | -9.05 | -0.73 | -2.87 |

PPO beats `nearest` in all three retry evaluations and improves the three-seed
mean reward by about `7.34` points.

## Next Validation Steps

1. Repeat the tight-deadline run over multiple training seeds.
2. Add confidence intervals over evaluation seeds.
3. Add a stronger handcrafted baseline that explicitly trades off distance,
   slack, and compute time.
4. Move from shared-policy PPO toward IPPO/MAPPO once the benchmark scenario is
   fixed.
