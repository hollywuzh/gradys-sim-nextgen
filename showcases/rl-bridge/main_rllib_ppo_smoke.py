"""Minimal RLlib PPO smoke training for the GrADyS RL bridge."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("RAY_DEDUP_LOGS", "0")
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from rllib_multiagent_adapter import GradysUAVServiceRLlibEnv


SHOWCASE_DIR = Path(__file__).resolve().parent
ENV_NAME = "gradys_uav_service_rllib"
POLICY_ID = "shared_uav_policy"


def main() -> None:
    args = _parse_args()
    output_path = _resolve_showcase_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env_config = _env_config(args)
    register_env(ENV_NAME, lambda config: GradysUAVServiceRLlibEnv(dict(config)))

    reference_env = GradysUAVServiceRLlibEnv(env_config)
    observation_space = reference_env.observation_space
    action_space = reference_env.action_space
    reference_env.close()

    ray.init(
        local_mode=args.local_mode,
        include_dashboard=False,
        ignore_reinit_error=True,
        num_cpus=args.num_cpus,
        log_to_driver=not args.quiet,
    )

    algorithm = None
    results = []
    try:
        config = _ppo_config(args, env_config, observation_space, action_space)
        algorithm = config.build_algo()

        for iteration in range(args.iterations):
            result = algorithm.train()
            summary = _training_summary(iteration, result)
            results.append(summary)
            print(
                f"iter={iteration} "
                f"reward_mean={summary.get('episode_reward_mean')} "
                f"episodes={summary.get('num_episodes')} "
                f"steps={summary.get('num_env_steps_sampled_lifetime')}"
            )

        if args.checkpoint:
            checkpoint_dir = _resolve_showcase_path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = algorithm.save(checkpoint_dir)
            results[-1]["checkpoint"] = str(checkpoint)
            print(f"Wrote RLlib checkpoint: {checkpoint}")
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()

    payload = {
        "ray_version": ray.__version__,
        "env_name": ENV_NAME,
        "policy_id": POLICY_ID,
        "env_config": env_config,
        "ppo_config": {
            "api_stack": "legacy",
            "train_batch_size": args.train_batch_size,
            "minibatch_size": args.minibatch_size,
            "num_epochs": args.num_epochs,
            "rollout_fragment_length": args.rollout_fragment_length,
            "iterations": args.iterations,
        },
        "results": results,
    }
    output_path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")
    print(f"Wrote PPO smoke metrics: {output_path.resolve()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-uavs", type=int, default=2)
    parser.add_argument("--num-devices", type=int, default=8)
    parser.add_argument("--episode-duration", type=float, default=12.0)
    parser.add_argument("--control-interval", type=float, default=1.0)
    parser.add_argument("--candidate-limit", type=int, default=3)
    parser.add_argument("--task-arrival-probability", type=float, default=0.08)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--rollout-fragment-length", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--local-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--checkpoint-dir", default="outputs/rllib_checkpoints")
    parser.add_argument("--output", default="outputs/rllib_ppo_smoke_metrics.json")
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
        "rllib_team_reward": True,
    }


def _ppo_config(args, env_config, observation_space, action_space) -> PPOConfig:
    return (
        PPOConfig()
        .api_stack(
            enable_env_runner_and_connector_v2=False,
            enable_rl_module_and_learner=False,
        )
        .environment(
            env=ENV_NAME,
            env_config=env_config,
            disable_env_checking=False,
        )
        .framework("torch")
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=0,
            rollout_fragment_length=args.rollout_fragment_length,
            batch_mode="complete_episodes",
        )
        .training(
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            model={"fcnet_hiddens": [64, 64], "fcnet_activation": "tanh"},
        )
        .multi_agent(
            policies={
                POLICY_ID: PolicySpec(
                    policy_class=None,
                    observation_space=observation_space,
                    action_space=action_space,
                    config={},
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: POLICY_ID,
        )
    )


def _training_summary(iteration: int, result: dict[str, Any]) -> dict[str, Any]:
    env_runners = result.get("env_runners", {})
    return {
        "iteration": iteration,
        "episode_reward_mean": _first_present(
            result,
            env_runners,
            "episode_reward_mean",
        ),
        "episode_len_mean": _first_present(
            result,
            env_runners,
            "episode_len_mean",
        ),
        "num_episodes": _first_present(
            result,
            env_runners,
            "num_episodes",
            "episodes_this_iter",
        ),
        "num_env_steps_sampled_lifetime": _first_present(
            result,
            env_runners,
            "num_env_steps_sampled_lifetime",
            "num_env_steps_sampled",
        ),
        "timesteps_total": result.get("timesteps_total"),
        "training_iteration": result.get("training_iteration"),
        "time_total_s": result.get("time_total_s"),
    }


def _first_present(primary: dict[str, Any], secondary: dict[str, Any], *keys: str):
    for key in keys:
        if key in primary:
            return primary[key]
        if key in secondary:
            return secondary[key]
    return None


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
