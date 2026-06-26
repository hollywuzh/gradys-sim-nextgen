"""Run the GrADyS RL bridge with Alibaba Cluster Data V2017 task features."""

from __future__ import annotations

import argparse
import random

from gradys_uav_service_env import GradysUAVServiceCoreEnv, UAVServiceEnvConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task_table",
        help="Path to Alibaba Cluster Data V2017 task table, e.g. batch_task.csv.",
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = UAVServiceEnvConfig(
        num_uavs=3,
        num_devices=20,
        episode_duration=float(args.steps),
        control_interval=1.0,
        task_arrival_probability=0.05,
        workload_mode="alibaba_v2017",
        alibaba_task_table_path=args.task_table,
        seed=args.seed,
    )
    env = GradysUAVServiceCoreEnv(config)
    observations = env.reset()
    rng = random.Random(args.seed + 1)

    print("Trace workload smoke test")
    print(f"  observation_size={len(next(iter(observations.values())))}")
    print(f"  action_size={env.action_size}")

    done = False
    step = 0
    while not done:
        actions = {agent: rng.randrange(env.action_size) for agent in env.agents}
        result = env.step(actions)
        done = result.terminated or result.truncated
        print(
            f"step={step:02d} {env.render_text()} "
            f"reward_sum={sum(result.rewards.values()):.2f}"
        )
        step += 1

    env.close()


if __name__ == "__main__":
    main()
