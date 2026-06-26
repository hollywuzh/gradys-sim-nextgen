"""Random-policy smoke test for the uavsim4security RL bridge."""

from __future__ import annotations

import argparse
import random

from uavsim4security_core_env import UAVSim4SecurityCoreEnv, UAVSim4SecurityEnvConfig


def main() -> None:
    args = _parse_args()
    config = UAVSim4SecurityEnvConfig(
        num_uavs=args.num_uavs,
        episode_duration=args.episode_duration,
        control_interval=args.control_interval,
        candidate_limit=args.candidate_limit,
        seed=args.seed,
        attacks=tuple(args.attacks.split(",")) if args.attacks else (),
        attacker_ids=tuple(int(item) for item in args.attacker_ids.split(",") if item),
        attack_probability=args.attack_probability,
    )
    rng = random.Random(args.seed)
    env = UAVSim4SecurityCoreEnv(config)
    observations = env.reset(seed=args.seed)
    print("Initial observation sizes:")
    for agent, obs in observations.items():
        print(f"  {agent}: {len(obs)}")

    step = 0
    done = False
    total_reward = 0.0
    try:
        while not done and step < args.max_steps:
            actions = {}
            for agent in env.agents:
                legal = [
                    idx
                    for idx, value in enumerate(env.action_mask(agent))
                    if value > 0.0
                ]
                actions[agent] = rng.choice(legal) if legal else 0
            result = env.step(actions)
            reward_sum = sum(result.rewards.values())
            total_reward += reward_sum
            metrics = env.metrics_snapshot()
            print(
                f"step={step:02d} t={metrics['time']:.2f}s "
                f"generated={metrics['generated_packets']:.0f} "
                f"delivered={metrics['delivered_packets']:.0f} "
                f"pdr={metrics['packet_delivery_ratio']:.3f} "
                f"reward_sum={reward_sum:.2f} actions={actions}"
            )
            done = result.terminated or result.truncated
            step += 1
    finally:
        env.close()

    print(f"uavsim4security random-policy smoke test finished: reward={total_reward:.2f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-uavs", type=int, default=4)
    parser.add_argument("--episode-duration", type=float, default=3.0)
    parser.add_argument("--control-interval", type=float, default=0.5)
    parser.add_argument("--candidate-limit", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--attacks", default="")
    parser.add_argument("--attacker-ids", default="")
    parser.add_argument("--attack-probability", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
