"""Evaluate a saved RLlib PPO checkpoint against rule baselines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("RAY_DEDUP_LOGS", "0")
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from experiment_runner import summarize_by_policy, write_episode_csv
from main_rllib_ppo_train import (
    _evaluate_ppo,
    _print_summary,
    _run_baselines,
    _write_summary_csv,
)
from rllib_action_mask_model import register_action_mask_model
from rllib_multiagent_adapter import GradysUAVServiceRLlibEnv


SHOWCASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    args = _parse_args()
    run_dir = _resolve_showcase_path(args.run_dir)
    output_dir = _resolve_showcase_path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = json.loads((run_dir / "config.json").read_text())
    core_env_config = config_payload["core_env_config"]
    rllib_env_config = config_payload["rllib_env_config"]
    checkpoint_dir = _resolve_showcase_path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints"
    env_name = config_payload.get("env_name", "gradys_uav_service_rllib_train")
    register_env(env_name, lambda config: GradysUAVServiceRLlibEnv(dict(config)))
    register_action_mask_model()

    ray.init(
        local_mode=args.local_mode,
        include_dashboard=False,
        ignore_reinit_error=True,
        num_cpus=args.num_cpus,
        log_to_driver=not args.quiet,
    )

    algorithm = None
    try:
        algorithm = Algorithm.from_checkpoint(str(checkpoint_dir))
        ppo_rows = _evaluate_ppo(
            algorithm=algorithm,
            env_config=rllib_env_config,
            episodes=args.eval_episodes,
            seed=args.eval_seed,
            explore=args.explore,
        )
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()

    baseline_args = argparse.Namespace(
        baseline_policies=args.baseline_policies,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
    )
    baseline_rows = _run_baselines(baseline_args, core_env_config)
    comparison_rows = baseline_rows + ppo_rows

    suffix = "stochastic" if args.explore else "deterministic"
    write_episode_csv(str(output_dir / f"ppo_eval_{suffix}_episodes.csv"), ppo_rows)
    write_episode_csv(str(output_dir / f"comparison_{suffix}_episodes.csv"), comparison_rows)
    summary = summarize_by_policy(comparison_rows)
    _write_summary_csv(output_dir / f"comparison_{suffix}_summary.csv", summary)
    _print_summary(summary)
    print(f"Wrote checkpoint evaluation under: {output_dir.resolve()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=1007)
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--baseline-policies", default="random,nearest,edf,slo-risk")
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--local-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _resolve_showcase_path(path: str) -> Path:
    output_path = Path(path)
    if output_path.is_absolute():
        return output_path
    return SHOWCASE_DIR / output_path


if __name__ == "__main__":
    main()
