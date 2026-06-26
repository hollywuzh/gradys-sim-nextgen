"""Reusable experiment runner for GrADyS RL bridge policies."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from baseline_policies import BaselinePolicy
from gradys_uav_service_env import GradysUAVServiceCoreEnv, UAVServiceEnvConfig


@dataclass(frozen=True)
class EpisodeSummary:
    policy: str
    episode: int
    seed: int
    steps: int
    total_reward: float
    generated_tasks: float
    hits: float
    misses: float
    pending_tasks: float
    success_rate: float
    miss_rate: float
    total_energy_used: float
    mean_pending_tasks: float
    max_pending_tasks: float


def run_policy_episodes(
    policy_name: str,
    policy: BaselinePolicy,
    config: UAVServiceEnvConfig,
    episodes: int,
    seed: int,
) -> List[EpisodeSummary]:
    summaries: List[EpisodeSummary] = []
    policy_rng = random.Random(seed + 100003)
    env = GradysUAVServiceCoreEnv(config)
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            summaries.append(
                run_one_episode(
                    policy_name=policy_name,
                    policy=policy,
                    env=env,
                    episode=episode,
                    seed=episode_seed,
                    policy_rng=policy_rng,
                )
            )
    finally:
        env.close()
    return summaries


def run_one_episode(
    policy_name: str,
    policy: BaselinePolicy,
    env: GradysUAVServiceCoreEnv,
    episode: int,
    seed: int,
    policy_rng: random.Random,
) -> EpisodeSummary:
    env.reset(seed=seed)
    done = False
    steps = 0
    total_reward = 0.0
    pending_sum = 0.0
    max_pending = 0.0

    while not done:
        actions = policy(env, policy_rng)
        result = env.step(actions)
        total_reward += sum(result.rewards.values())
        snapshot = env.metrics_snapshot()
        pending = snapshot["pending_tasks"]
        pending_sum += pending
        max_pending = max(max_pending, pending)
        done = result.terminated or result.truncated
        steps += 1

    metrics = env.metrics_snapshot()
    return EpisodeSummary(
        policy=policy_name,
        episode=episode,
        seed=seed,
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


def summarize_by_policy(rows: Iterable[EpisodeSummary]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[EpisodeSummary]] = {}
    for row in rows:
        grouped.setdefault(row.policy, []).append(row)

    summary: Dict[str, Dict[str, float]] = {}
    for policy, policy_rows in grouped.items():
        summary[policy] = {
            "episodes": float(len(policy_rows)),
            "avg_reward": _mean(row.total_reward for row in policy_rows),
            "avg_success_rate": _mean(row.success_rate for row in policy_rows),
            "avg_miss_rate": _mean(row.miss_rate for row in policy_rows),
            "avg_hits": _mean(row.hits for row in policy_rows),
            "avg_misses": _mean(row.misses for row in policy_rows),
            "avg_energy": _mean(row.total_energy_used for row in policy_rows),
            "avg_mean_pending": _mean(row.mean_pending_tasks for row in policy_rows),
            "avg_max_pending": _mean(row.max_pending_tasks for row in policy_rows),
        }
    return summary


def write_episode_csv(path: str, rows: Iterable[EpisodeSummary]) -> None:
    rows = list(rows)
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=list(asdict(rows[0]).keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))
