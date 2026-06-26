# GrADyS RL Bridge Showcase

This showcase explores how to use GrADyS-SIM as a UAV mobility and network
simulation backend while exposing reinforcement-learning friendly interfaces.

The goal is not to provide a final MAPPO implementation. The goal is to create
a clean research scaffold where GrADyS owns the UAV/service simulation and RL
libraries own policy learning.

## Research Motivation

The target research problem is SLO-aware multi-UAV edge service orchestration:

- spatial tasks arrive stochastically at edge devices;
- UAVs move, collect/offload tasks, and spend energy;
- tasks carry end-to-end deadlines;
- a controller decides which UAV should serve which request.

This showcase is a minimal stepping stone toward the larger paper idea:

```text
GrADyS mobility/network simulation
    -> Gymnasium single-controller wrapper
    -> PettingZoo parallel multi-agent wrapper
    -> RLlib MultiAgentEnv wrapper
    -> MAPPO/IPPO/PPO training
```

## Files

- `rl_protocols.py`
  - Minimal GrADyS protocols for RL-controlled UAVs and passive edge devices.
- `gradys_uav_service_env.py`
  - Dependency-light core environment built on GrADyS-SIM.
  - Exposes `reset()` and `step()` without requiring Gymnasium/PettingZoo/RLlib.
- `workload.py`
  - Synthetic and Alibaba Cluster Data V2017 workload providers.
- `baseline_policies.py`
  - Rule-based baselines: hover, random, nearest request first, EDF, and
    SLO-risk dispatch.
- `experiment_runner.py`
  - Reusable multi-episode evaluation loop and CSV metrics writer.
- `ENVIRONMENT.md`
  - Recommended conda environment and staged RL dependency plan.
- `CLOSED_LOOP_GUIDE.md`
  - Step-by-step explanation of how GrADyS, Gymnasium, and RL data collection
    form the closed loop.
- `PPO_RESULTS.md`
  - Reproducible RLlib PPO-vs-baseline results, including the first scenario
    where PPO beats `nearest`.
- `gymnasium_adapter.py`
  - Optional centralized Gymnasium adapter.
- `pettingzoo_parallel_adapter.py`
  - Optional PettingZoo ParallelEnv adapter for multi-UAV MARL.
- `rllib_multiagent_adapter.py`
  - Optional RLlib MultiAgentEnv adapter.
- `main_random.py`
  - Smoke-test runner with a random policy.
- `main_trace_workload.py`
  - Smoke-test runner that samples task features from an Alibaba trace CSV.
- `main_baselines.py`
  - Baseline evaluation entry point for reproducible policy comparisons.
- `main_gym_random.py`
  - Minimal Gymnasium wrapper smoke test with random actions.
- `main_rllib_random.py`
  - RLlib `MultiAgentEnv` random-rollout smoke test with JSONL transition logs.
- `main_rllib_ppo_smoke.py`
  - Minimal RLlib PPO training smoke test with one shared UAV policy.
- `main_rllib_ppo_train.py`
  - Shared-policy RLlib PPO training and post-training evaluation against rule
    baselines.
- `main_visualize.py`
  - Visual trajectory runner that writes PNG/GIF artifacts under
    `showcases/rl-bridge/outputs/`.
- `requirements-rl.txt`
  - Optional RL package hints.
- `requirements-minimal.txt`
  - First-stage dependency set: Gymnasium only.
- `requirements-training.txt`
  - Later centralized PPO baseline dependencies.
- `requirements-marl.txt`
  - Later multi-agent RL dependencies.

## Quick Start

From the repository root:

```bash
conda activate gradsim-test
python showcases/rl-bridge/main_random.py
```

The recommended first-stage setup is documented in `ENVIRONMENT.md`. Start
with the core GrADyS environment and rule baselines before installing heavier
RL libraries.

Run with Alibaba Cluster Data V2017 task features:

```bash
python showcases/rl-bridge/main_trace_workload.py /path/to/batch_task.csv
```

The Alibaba trace is used only as a workload-feature source: task duration,
requested CPU, and requested memory are mapped to deadline, compute demand, and
payload-size proxies. Spatial task locations, UAV mobility, and wireless service
windows still come from the GrADyS environment.

Run deterministic and random baselines:

```bash
python showcases/rl-bridge/main_baselines.py --episodes 5
```

Run the same baselines with Alibaba-derived task features:

```bash
python showcases/rl-bridge/main_baselines.py \
  --workload alibaba_v2017 \
  --alibaba-task-table /path/to/batch_task.csv \
  --episodes 5 \
  --csv-output showcases/rl-bridge/outputs/rl-bridge-baselines.csv
```

Optional RL dependencies:

```bash
pip install -r showcases/rl-bridge/requirements-minimal.txt
python showcases/rl-bridge/main_gym_random.py
```

Visualize one episode:

```bash
python showcases/rl-bridge/main_visualize.py
```

Install and validate the RLlib path:

```bash
python -m pip install -r showcases/rl-bridge/requirements-marl.txt
python showcases/rl-bridge/main_rllib_random.py
python showcases/rl-bridge/main_rllib_ppo_smoke.py --iterations 1 --quiet
```

The RLlib PPO smoke test writes metrics to
`showcases/rl-bridge/outputs/rllib_ppo_smoke_metrics.json`.

Train and evaluate a shared-policy PPO baseline:

```bash
python showcases/rl-bridge/main_rllib_ppo_train.py \
  --iterations 10 \
  --eval-episodes 5 \
  --baseline-policies random,nearest,edf,slo-risk \
  --output-dir outputs/rllib_ppo_train
```

This writes training curves, per-episode evaluation rows, and a PPO-vs-baseline
summary under `showcases/rl-bridge/outputs/rllib_ppo_train/`.

Reproduce the first PPO-greater-than-`nearest` milestone:

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

See `PPO_RESULTS.md` for the 50-episode checkpoint evaluation. In that
tight-deadline scenario, shared-policy RLlib PPO reached `1.19` average reward
against `nearest` at `-8.91`.

The closed-loop data flow is explained in `CLOSED_LOOP_GUIDE.md`.

The core environment does not require Gymnasium. It is only needed when using
the optional Gymnasium adapter. The RLlib path is now available as a compact
multi-agent training scaffold; MAPPO-specific training should still wait until
the task/SLO model and reward design are stable.

## Design Notes

The core environment uses GrADyS-SIM through `SimulationBuilder`,
`DynamicVelocityMobilityHandler`, and `step_simulation()`. Each RL step sends a
velocity command to each UAV protocol, advances the event-based simulation for a
fixed control interval, and then computes task-completion rewards.

Actions are intentionally simple in this scaffold:

```text
0      -> hover / keep current target
1..K   -> choose one task from the local candidate list
```

The environment then moves the UAV toward the selected task location. A task is
completed when the UAV reaches the service radius and finishes the simplified
compute service. The reward penalizes missed deadlines and rewards successful
service.

This keeps the simulator-RL boundary visible. Later versions can replace the
action decoder with:

- discrete-continuous layered actions like AL-MAPPO;
- explicit resource allocation levels;
- task DAG critical-path slack;
- SLO mode switching: pre-positioning, dispatching, repositioning, batching.

## Next Research Extensions

1. Calibrate Alibaba-derived task-feature scaling against a target UAV edge
   device, e.g. Orange Pi / Jetson.
2. Add spatial hotspot models for weak-infrastructure IoT/disaster scenarios.
3. Add SLO feasibility lower bounds:

```text
earliest UAV arrival time + workflow critical path <= deadline
```

4. Add baselines:

- nearest-request-first;
- nearest-deadline-first;
- SLO-risk dispatch;
- fixed patrol/TSP-like route;
- MAPPO/IPPO/RLlib PPO.

5. Turn this showcase into a reusable benchmark for UAV-enabled edge service
   orchestration.
