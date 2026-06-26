# Environment Notes

Use the existing `gradsim-test` conda environment for the first-stage smoke
tests.

```bash
conda activate gradsim-test
cd /Users/wupengfei/Documents/Framework4test/gradys-sim-nextgen
```

This environment already has a compatible Python version and can run the core
GrADyS showcase. The local check on 2026-06-24 found:

```text
gradsim-test: Python 3.10.13, gradysim available
RL_learning_with_python: Python 3.7.16, gymnasium available, too old for this scaffold
```

## First-stage RL Choice

The project started with Gymnasium only, then moved to an RLlib smoke path once
the GrADyS core loop was stable.

Gymnasium is used as the environment API, not as the learning algorithm. This
keeps the first experiment focused on whether the GrADyS simulation can expose
observations, actions, rewards, termination, and metrics in a standard RL shape.

RLlib is now available as the first multi-agent training scaffold. The current
RLlib setup uses one shared PPO policy for all UAV agents. It is intentionally
small so that the algorithm layer can be validated before adding centralized
critics, MAPPO-specific losses, or large parallel rollouts.

The RLlib PPO path now uses optional action masking by default. This masks
invalid candidate indices and lets PPO train on the same raw observation vector
without wasting probability mass on unavailable actions.

## Minimal Install

Install only the minimal wrapper dependency first:

```bash
python -m pip install -r showcases/rl-bridge/requirements-minimal.txt
```

Then run:

```bash
python showcases/rl-bridge/main_random.py
python showcases/rl-bridge/main_baselines.py --episodes 3
python showcases/rl-bridge/main_gym_random.py
```

## RLlib Install

Install the multi-agent training dependencies:

```bash
python -m pip install -r showcases/rl-bridge/requirements-marl.txt
```

The validated local combination on 2026-06-25 is:

```text
ray 2.49.2
gymnasium 1.1.1
pettingzoo 1.26.1
torch 2.2.2
```

Run the RLlib adapter smoke test:

```bash
python showcases/rl-bridge/main_rllib_random.py
```

Run one minimal PPO iteration:

```bash
python showcases/rl-bridge/main_rllib_ppo_smoke.py --iterations 1 --quiet
```

Run the reusable PPO training/evaluation scaffold:

```bash
python showcases/rl-bridge/main_rllib_ppo_train.py \
  --iterations 10 \
  --eval-episodes 5 \
  --baseline-policies random,nearest,edf,slo-risk \
  --output-dir outputs/rllib_ppo_train
```

Main artifacts:

```text
showcases/rl-bridge/outputs/rllib_ppo_train/training_metrics.csv
showcases/rl-bridge/outputs/rllib_ppo_train/ppo_eval_episodes.csv
showcases/rl-bridge/outputs/rllib_ppo_train/baseline_eval_episodes.csv
showcases/rl-bridge/outputs/rllib_ppo_train/comparison_summary.csv
```

In sandboxed Codex sessions, `ray.init()` may need an unsandboxed run because
Ray/psutil enumerates local processes on macOS. In a normal terminal this is
just a regular Python command.

The first checkpoint that beats `nearest` is documented in `PPO_RESULTS.md`.
The 50-episode deterministic checkpoint evaluation in the tight-deadline setup
reported PPO at `1.19` average reward and `nearest` at `-8.91`.

## Later Stages

- Stage 0: GrADyS core env + rule baselines.
- Stage 1: Gymnasium wrapper + random actions.
- Stage 2: Stable-Baselines3 PPO as a centralized single-controller baseline.
- Stage 3: RLlib shared-policy PPO smoke training.
- Stage 4: PettingZoo/RLlib multi-agent training variants.
- Stage 5: MAPPO-style training after the task/SLO model is fixed.
