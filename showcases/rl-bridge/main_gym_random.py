"""Minimal Gymnasium smoke test for the GrADyS RL bridge."""

from __future__ import annotations

from gradys_uav_service_env import UAVServiceEnvConfig
from gymnasium_adapter import GradysUAVServiceGymEnv


def main() -> None:
    config = UAVServiceEnvConfig(
        num_uavs=1,
        num_devices=5,
        episode_duration=10.0,
        control_interval=1.0,
        task_arrival_probability=0.08,
        candidate_limit=3,
        seed=3,
    )
    env = GradysUAVServiceGymEnv(config)
    observation, _ = env.reset(seed=3)
    print(f"Gymnasium smoke test: obs_shape={observation.shape} action_space={env.action_space}")

    done = False
    step = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(
            f"step={step:02d} reward={reward:.2f} "
            f"done={done} {env.render()}"
        )
        step += 1

    env.close()
    print("Gymnasium random-action smoke test finished.")


if __name__ == "__main__":
    main()
