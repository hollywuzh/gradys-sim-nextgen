"""RLlib MultiAgentEnv random-rollout smoke test for the GrADyS RL bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rllib_multiagent_adapter import GradysUAVServiceRLlibEnv


SHOWCASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    args = _parse_args()
    output_path = _resolve_showcase_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env_config = _env_config(args)
    env = GradysUAVServiceRLlibEnv(env_config)
    observations, _ = env.reset(seed=args.seed)

    done = False
    step = 0
    total_reward = 0.0
    records = []

    try:
        while not done and step < args.max_steps:
            actions = {
                agent: int(env.action_space.sample())
                for agent in observations
            }
            next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
            reward_sum = float(sum(rewards.values()))
            total_reward += reward_sum
            done = bool(terminateds["__all__"] or truncateds["__all__"])

            record = {
                "step": step,
                "time": env.core.time,
                "actions": actions,
                "rewards": rewards,
                "reward_sum": reward_sum,
                "terminateds": terminateds,
                "truncateds": truncateds,
                "infos": infos,
                "metrics": env.core.metrics_snapshot(),
                "active_agents": list(next_observations.keys()),
            }
            records.append(record)
            observations = next_observations
            step += 1
    finally:
        env.close()

    with output_path.open("w") as file_obj:
        for record in records:
            file_obj.write(json.dumps(_json_safe(record)) + "\n")

    final_metrics = records[-1]["metrics"] if records else env.core.metrics_snapshot()
    print("RLlib random rollout finished.")
    print(f"steps={step} total_reward={total_reward:.2f}")
    print(
        "final: "
        f"hits={final_metrics['hits']:.0f} "
        f"misses={final_metrics['misses']:.0f} "
        f"pending={final_metrics['pending_tasks']:.0f}"
    )
    print(f"Wrote rollout JSONL: {output_path.resolve()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-uavs", type=int, default=2)
    parser.add_argument("--num-devices", type=int, default=8)
    parser.add_argument("--episode-duration", type=float, default=12.0)
    parser.add_argument("--control-interval", type=float, default=1.0)
    parser.add_argument("--candidate-limit", type=int, default=3)
    parser.add_argument("--task-arrival-probability", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--output", default="outputs/rllib_random_rollout.jsonl")
    return parser.parse_args()


def _env_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_uavs": args.num_uavs,
        "num_devices": args.num_devices,
        "episode_duration": args.episode_duration,
        "control_interval": args.control_interval,
        "candidate_limit": args.candidate_limit,
        "task_arrival_probability": args.task_arrival_probability,
        "seed": args.seed,
    }


def _resolve_showcase_path(path: str) -> Path:
    output_path = Path(path)
    if output_path.is_absolute():
        return output_path
    return SHOWCASE_DIR / output_path


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
