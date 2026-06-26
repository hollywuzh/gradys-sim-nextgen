"""Run rule-based baselines for the GrADyS RL bridge environment."""

from __future__ import annotations

import argparse
from typing import List

from baseline_policies import BASELINE_POLICIES
from experiment_runner import run_policy_episodes, summarize_by_policy, write_episode_csv
from gradys_uav_service_env import UAVServiceEnvConfig


def main() -> None:
    args = _parse_args()
    selected_policies = _select_policies(args.policies)
    config = UAVServiceEnvConfig(
        num_uavs=args.num_uavs,
        num_devices=args.num_devices,
        episode_duration=args.episode_duration,
        control_interval=args.control_interval,
        candidate_limit=args.candidate_limit,
        task_arrival_probability=args.task_arrival_probability,
        deadline_range=tuple(args.deadline_range),
        compute_demand_range=tuple(args.compute_demand_range),
        data_size_range=tuple(args.data_size_range),
        workload_mode=args.workload,
        alibaba_task_table_path=args.alibaba_task_table,
        alibaba_max_rows=args.alibaba_max_rows,
        seed=args.seed,
    )

    rows = []
    for policy_name in selected_policies:
        policy = BASELINE_POLICIES[policy_name]
        rows.extend(
            run_policy_episodes(
                policy_name=policy_name,
                policy=policy,
                config=config,
                episodes=args.episodes,
                seed=args.seed,
            )
        )

    summary = summarize_by_policy(rows)
    _print_summary(summary)
    if args.csv_output:
        write_episode_csv(args.csv_output, rows)
        print(f"\nWrote per-episode metrics to {args.csv_output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-uavs", type=int, default=4)
    parser.add_argument("--num-devices", type=int, default=30)
    parser.add_argument("--episode-duration", type=float, default=60.0)
    parser.add_argument("--control-interval", type=float, default=1.0)
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--task-arrival-probability", type=float, default=0.06)
    parser.add_argument("--deadline-range", type=float, nargs=2, default=(20.0, 60.0))
    parser.add_argument("--compute-demand-range", type=float, nargs=2, default=(20.0, 200.0))
    parser.add_argument("--data-size-range", type=float, nargs=2, default=(0.5, 8.0))
    parser.add_argument("--workload", choices=["synthetic", "alibaba_v2017"], default="synthetic")
    parser.add_argument("--alibaba-task-table")
    parser.add_argument("--alibaba-max-rows", type=int, default=50000)
    parser.add_argument("--csv-output")
    parser.add_argument(
        "--policies",
        default="hover,random,nearest,edf,slo-risk",
        help="Comma-separated policy names. Available: "
        + ",".join(sorted(BASELINE_POLICIES.keys())),
    )
    args = parser.parse_args()
    if args.workload == "alibaba_v2017" and not args.alibaba_task_table:
        parser.error("--alibaba-task-table is required when --workload=alibaba_v2017")
    return args


def _select_policies(policy_arg: str) -> List[str]:
    selected = [name.strip() for name in policy_arg.split(",") if name.strip()]
    unknown = [name for name in selected if name not in BASELINE_POLICIES]
    if unknown:
        raise ValueError(f"Unknown policies: {unknown}. Available: {sorted(BASELINE_POLICIES)}")
    return selected


def _print_summary(summary) -> None:
    print("Policy baseline summary")
    print(
        "policy       episodes  avg_reward  success  miss     hits   misses  "
        "energy   meanQ   maxQ"
    )
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


if __name__ == "__main__":
    main()
