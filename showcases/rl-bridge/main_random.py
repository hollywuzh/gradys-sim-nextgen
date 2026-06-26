"""Smoke test for the GrADyS RL bridge core environment."""

from __future__ import annotations

import random

from gradys_uav_service_env import GradysUAVServiceCoreEnv, UAVServiceEnvConfig


def main() -> None:
    config = UAVServiceEnvConfig(
        num_uavs=3,
        num_devices=20,
        episode_duration=30.0,
        control_interval=1.0,
        task_arrival_probability=0.05,
        seed=7,
    )
    env = GradysUAVServiceCoreEnv(config)
    observations = env.reset()
    rng = random.Random(11)

    print("Initial observation sizes:")
    for agent, observation in observations.items():
        print(f"  {agent}: {len(observation)}")

    step = 0
    done = False
    while not done:
        actions = {
            agent: rng.randrange(env.action_size)
            for agent in env.agents
        }
        result = env.step(actions)
        done = result.terminated or result.truncated
        reward_sum = sum(result.rewards.values())
        print(
            f"step={step:02d} {env.render_text()} "
            f"reward_sum={reward_sum:.2f} actions={actions}"
        )
        step += 1

    env.close()
    print("Random-policy smoke test finished.")


if __name__ == "__main__":
    main()
