"""Train and evaluate a shared-policy RLlib PPO baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, List

os.environ.setdefault("RAY_DEDUP_LOGS", "0")
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from baseline_policies import BASELINE_POLICIES
from experiment_runner import (
    EpisodeSummary,
    run_policy_episodes,
    summarize_by_policy,
    write_episode_csv,
)
from rllib_action_mask_model import ACTION_MASK_MODEL, register_action_mask_model
from rllib_multiagent_adapter import GradysUAVServiceRLlibEnv


SHOWCASE_DIR = Path(__file__).resolve().parent
ENV_NAME = "gradys_uav_service_rllib_train"
POLICY_ID = "shared_uav_policy"


def main() -> None:
    args = _parse_args()
    output_dir = _resolve_showcase_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    core_env_config = _core_env_config(args)
    rllib_env_config = _rllib_env_config(args, core_env_config)
    _write_json(
        output_dir / "config.json",
        {
            "env_name": ENV_NAME,
            "policy_id": POLICY_ID,
            "core_env_config": core_env_config,
            "rllib_env_config": rllib_env_config,
            "args": vars(args),
        },
    )

    register_env(ENV_NAME, lambda config: GradysUAVServiceRLlibEnv(dict(config)))
    if args.action_mask:
        register_action_mask_model()
    reference_env = GradysUAVServiceRLlibEnv(rllib_env_config)
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
    training_rows: List[dict[str, Any]] = []
    checkpoint_path = None
    try:
        config = _ppo_config(args, rllib_env_config, observation_space, action_space)
        algorithm = config.build_algo()

        for iteration in range(args.iterations):
            result = algorithm.train()
            summary = _training_summary(iteration, result)
            training_rows.append(summary)
            print(
                f"iter={iteration:03d} "
                f"reward_mean={_fmt(summary['episode_reward_mean'])} "
                f"episodes={_fmt(summary['num_episodes'])} "
                f"env_steps={_fmt(summary['num_env_steps_sampled_lifetime'])}"
            )

        if args.checkpoint:
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = str(algorithm.save(checkpoint_dir))
            print(f"Wrote checkpoint: {checkpoint_path}")

        ppo_rows = _evaluate_ppo(
            algorithm=algorithm,
            env_config=rllib_env_config,
            episodes=args.eval_episodes,
            seed=args.eval_seed,
            explore=args.eval_explore,
        )
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()

    baseline_rows = _run_baselines(args, core_env_config)
    comparison_rows = list(baseline_rows) + list(ppo_rows)

    _write_dict_csv(output_dir / "training_metrics.csv", training_rows)
    _write_json(
        output_dir / "training_metrics.json",
        {
            "ray_version": ray.__version__,
            "env_name": ENV_NAME,
            "policy_id": POLICY_ID,
            "checkpoint": checkpoint_path,
            "rllib_team_reward": args.team_reward,
            "rllib_action_mask": args.action_mask,
            "rllib_mask_hover_when_candidates": args.mask_hover_when_candidates,
            "training": training_rows,
        },
    )
    write_episode_csv(str(output_dir / "ppo_eval_episodes.csv"), ppo_rows)
    write_episode_csv(str(output_dir / "baseline_eval_episodes.csv"), baseline_rows)
    write_episode_csv(str(output_dir / "comparison_episodes.csv"), comparison_rows)
    _write_summary_csv(output_dir / "comparison_summary.csv", summarize_by_policy(comparison_rows))

    print(f"Wrote PPO artifacts under: {output_dir.resolve()}")
    _print_summary(summarize_by_policy(comparison_rows))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-seed", type=int, default=1007)
    parser.add_argument("--num-uavs", type=int, default=2)
    parser.add_argument("--num-devices", type=int, default=8)
    parser.add_argument("--episode-duration", type=float, default=12.0)
    parser.add_argument("--control-interval", type=float, default=1.0)
    parser.add_argument("--candidate-limit", type=int, default=3)
    parser.add_argument("--task-arrival-probability", type=float, default=0.08)
    parser.add_argument("--deadline-range", type=float, nargs=2, default=(20.0, 60.0))
    parser.add_argument("--compute-demand-range", type=float, nargs=2, default=(20.0, 200.0))
    parser.add_argument("--data-size-range", type=float, nargs=2, default=(0.5, 8.0))
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--rollout-fragment-length", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--local-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-explore", action="store_true")
    parser.add_argument("--team-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mask-hover-when-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When action masking is enabled, only allow hover if no task can be selected.",
    )
    parser.add_argument(
        "--baseline-policies",
        default="random,nearest,edf,slo-risk",
        help="Comma-separated baseline policies for post-training comparison.",
    )
    parser.add_argument("--output-dir", default="outputs/rllib_ppo_train")
    return parser.parse_args()


def _core_env_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_uavs": args.num_uavs,
        "num_devices": args.num_devices,
        "episode_duration": args.episode_duration,
        "control_interval": args.control_interval,
        "candidate_limit": args.candidate_limit,
        "task_arrival_probability": args.task_arrival_probability,
        "deadline_range": tuple(args.deadline_range),
        "compute_demand_range": tuple(args.compute_demand_range),
        "data_size_range": tuple(args.data_size_range),
        "seed": args.seed,
    }


def _rllib_env_config(args: argparse.Namespace, core_env_config: dict[str, Any]) -> dict[str, Any]:
    return {
        **core_env_config,
        "rllib_team_reward": args.team_reward,
        "rllib_auto_increment_seed": True,
        "rllib_action_mask": args.action_mask,
        "rllib_mask_hover_when_candidates": args.mask_hover_when_candidates,
    }


def _ppo_config(args, env_config, observation_space, action_space) -> PPOConfig:
    model_config = {"fcnet_hiddens": [64, 64], "fcnet_activation": "tanh"}
    if args.action_mask:
        model_config["custom_model"] = ACTION_MASK_MODEL
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
            gamma=args.gamma,
            lambda_=args.lambda_,
            model=model_config,
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


def _evaluate_ppo(
    algorithm,
    env_config: dict[str, Any],
    episodes: int,
    seed: int,
    explore: bool,
) -> List[EpisodeSummary]:
    rows: List[EpisodeSummary] = []
    env = GradysUAVServiceRLlibEnv(env_config)
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            observations, _ = env.reset(seed=episode_seed)
            done = False
            steps = 0
            total_reward = 0.0
            pending_sum = 0.0
            max_pending = 0.0

            while not done:
                actions = {
                    agent: _as_int_action(
                        algorithm.compute_single_action(
                            observation,
                            policy_id=POLICY_ID,
                            explore=explore,
                        )
                    )
                    for agent, observation in observations.items()
                }
                observations, rewards, terminateds, truncateds, _ = env.step(actions)
                total_reward += sum(rewards.values())
                metrics = env.core.metrics_snapshot()
                pending = metrics["pending_tasks"]
                pending_sum += pending
                max_pending = max(max_pending, pending)
                done = bool(terminateds["__all__"] or truncateds["__all__"])
                steps += 1

            metrics = env.core.metrics_snapshot()
            rows.append(
                EpisodeSummary(
                    policy="rllib-ppo",
                    episode=episode,
                    seed=episode_seed,
                    steps=steps,
                    total_reward=total_reward,
                    generated_tasks=metrics["generated_tasks"],
                    hits=metrics["hits"],
                    misses=metrics["misses"],
                    pending_tasks=metrics["pending_tasks"],
                    success_rate=metrics["success_rate"],
                    miss_rate=metrics["miss_rate"],
                    total_energy_used=metrics["total_energy_used"],
                    mean_pending_tasks=pending_sum / max(1, steps),
                    max_pending_tasks=max_pending,
                )
            )
    finally:
        env.close()
    return rows


def _run_baselines(args: argparse.Namespace, env_config: dict[str, Any]) -> List[EpisodeSummary]:
    selected = [name.strip() for name in args.baseline_policies.split(",") if name.strip()]
    unknown = [name for name in selected if name not in BASELINE_POLICIES]
    if unknown:
        raise ValueError(f"Unknown baseline policies: {unknown}. Available: {sorted(BASELINE_POLICIES)}")

    from gradys_uav_service_env import UAVServiceEnvConfig

    config = UAVServiceEnvConfig(**env_config)
    rows: List[EpisodeSummary] = []
    for policy_name in selected:
        rows.extend(
            run_policy_episodes(
                policy_name=policy_name,
                policy=BASELINE_POLICIES[policy_name],
                config=config,
                episodes=args.eval_episodes,
                seed=args.eval_seed,
            )
        )
    return rows


def _training_summary(iteration: int, result: dict[str, Any]) -> dict[str, Any]:
    env_runners = result.get("env_runners", {})
    return {
        "iteration": iteration,
        "episode_reward_mean": _first_present(result, env_runners, "episode_reward_mean"),
        "episode_reward_min": _first_present(result, env_runners, "episode_reward_min"),
        "episode_reward_max": _first_present(result, env_runners, "episode_reward_max"),
        "episode_len_mean": _first_present(result, env_runners, "episode_len_mean"),
        "num_episodes": _first_present(result, env_runners, "num_episodes", "episodes_this_iter"),
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


def _write_summary_csv(path: Path, summary: dict[str, dict[str, float]]) -> None:
    rows = [{"policy": policy, **metrics} for policy, metrics in summary.items()]
    _write_dict_csv(path, rows)


def _write_dict_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")


def _print_summary(summary) -> None:
    print("Evaluation summary")
    print("policy       episodes  avg_reward  success  miss     hits   misses  energy   meanQ   maxQ")
    for policy, metrics in summary.items():
        print(
            f"{policy:<11} "
            f"{metrics['episodes']:>8.0f} "
            f"{metrics['avg_reward']:>11.2f} "
            f"{metrics['avg_success_rate']:>7.3f} "
            f"{metrics['avg_miss_rate']:>7.3f} "
            f"{metrics['avg_hits']:>6.1f} "
            f"{metrics['avg_misses']:>7.1f} "
            f"{metrics['avg_energy']:>7.1f} "
            f"{metrics['avg_mean_pending']:>7.1f} "
            f"{metrics['avg_max_pending']:>6.1f}"
        )


def _resolve_showcase_path(path: str) -> Path:
    output_path = Path(path)
    if output_path.is_absolute():
        return output_path
    return SHOWCASE_DIR / output_path


def _as_int_action(action) -> int:
    if isinstance(action, tuple):
        action = action[0]
    if hasattr(action, "item"):
        return int(action.item())
    return int(action)


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__fspath__"):
        return os.fspath(value)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return str(value)
    return value


if __name__ == "__main__":
    main()
