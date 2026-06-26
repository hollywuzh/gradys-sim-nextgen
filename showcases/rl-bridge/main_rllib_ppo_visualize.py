"""Visualize an RLlib PPO checkpoint against a rule baseline."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "gradys-rl-mpl"))
os.environ.setdefault("RAY_DEDUP_LOGS", "0")
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import ray
from matplotlib.animation import FuncAnimation, PillowWriter
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from baseline_policies import BASELINE_POLICIES
from gradys_uav_service_env import GradysUAVServiceCoreEnv, UAVServiceEnvConfig
from main_rllib_ppo_train import POLICY_ID
from main_visualize import Snapshot, _draw_map, _snapshot
from rllib_action_mask_model import register_action_mask_model
from rllib_multiagent_adapter import GradysUAVServiceRLlibEnv


SHOWCASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    args = _parse_args()
    run_dir = _resolve_showcase_path(args.run_dir)
    output_dir = _resolve_showcase_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = json.loads((run_dir / "config.json").read_text())
    env_name = config_payload.get("env_name", "gradys_uav_service_rllib_train")
    rllib_env_config = config_payload["rllib_env_config"]
    core_env_config = config_payload["core_env_config"]
    checkpoint_dir = _resolve_showcase_path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints"

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
        algorithm = Algorithm.from_checkpoint(str(checkpoint_dir.resolve()))
        ppo_snapshots, ppo_reward = _record_ppo_episode(
            algorithm=algorithm,
            env_config=rllib_env_config,
            seed=args.seed,
            explore=args.explore,
        )
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()

    baseline_snapshots, baseline_reward = _record_baseline_episode(
        config=UAVServiceEnvConfig(**core_env_config),
        policy_name=args.baseline_policy,
        seed=args.seed,
    )

    png_output = output_dir / f"ppo_vs_{args.baseline_policy}_seed{args.seed}.png"
    gif_output = output_dir / f"ppo_vs_{args.baseline_policy}_seed{args.seed}.gif"
    summary_output = output_dir / f"ppo_vs_{args.baseline_policy}_seed{args.seed}.json"

    _save_comparison_png(
        baseline_snapshots=baseline_snapshots,
        ppo_snapshots=ppo_snapshots,
        baseline_name=args.baseline_policy,
        baseline_reward=baseline_reward,
        ppo_reward=ppo_reward,
        config=UAVServiceEnvConfig(**core_env_config),
        output_path=png_output,
    )
    if not args.no_gif:
        _save_comparison_gif(
            baseline_snapshots=baseline_snapshots,
            ppo_snapshots=ppo_snapshots,
            baseline_name=args.baseline_policy,
            config=UAVServiceEnvConfig(**core_env_config),
            output_path=gif_output,
            fps=args.fps,
        )
    _write_summary(
        output_path=summary_output,
        seed=args.seed,
        baseline_name=args.baseline_policy,
        baseline_snapshots=baseline_snapshots,
        ppo_snapshots=ppo_snapshots,
        baseline_reward=baseline_reward,
        ppo_reward=ppo_reward,
    )

    baseline_metrics = baseline_snapshots[-1]["metrics"]
    ppo_metrics = ppo_snapshots[-1]["metrics"]
    print(
        f"{args.baseline_policy}: reward={baseline_reward:.2f} "
        f"hits={baseline_metrics['hits']:.0f} misses={baseline_metrics['misses']:.0f} "
        f"energy={baseline_metrics['total_energy_used']:.1f}"
    )
    print(
        f"rllib-ppo: reward={ppo_reward:.2f} "
        f"hits={ppo_metrics['hits']:.0f} misses={ppo_metrics['misses']:.0f} "
        f"energy={ppo_metrics['total_energy_used']:.1f}"
    )
    print(f"Wrote comparison PNG: {png_output.resolve()}")
    if not args.no_gif:
        print(f"Wrote comparison GIF: {gif_output.resolve()}")
    print(f"Wrote summary JSON: {summary_output.resolve()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="outputs/rllib_ppo_train_100iter_eval20_tight_deadline",
        help="PPO training run directory containing config.json and checkpoints/.",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=1008)
    parser.add_argument("--baseline-policy", choices=sorted(BASELINE_POLICIES), default="nearest")
    parser.add_argument("--output-dir", default="outputs/ppo_visualization")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--local-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _record_ppo_episode(
    algorithm,
    env_config: dict,
    seed: int,
    explore: bool,
) -> Tuple[List[Snapshot], float]:
    env = GradysUAVServiceRLlibEnv(env_config)
    snapshots: List[Snapshot] = []
    total_reward = 0.0
    try:
        observations, _ = env.reset(seed=seed)
        snapshots.append(_snapshot(env.core, {}, total_reward))
        done = False
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
            snapshots.append(_snapshot(env.core, actions, total_reward))
            done = bool(terminateds["__all__"] or truncateds["__all__"])
    finally:
        env.close()
    return snapshots, total_reward


def _record_baseline_episode(
    config: UAVServiceEnvConfig,
    policy_name: str,
    seed: int,
) -> Tuple[List[Snapshot], float]:
    env = GradysUAVServiceCoreEnv(config)
    policy = BASELINE_POLICIES[policy_name]
    snapshots: List[Snapshot] = []
    total_reward = 0.0
    try:
        env.reset(seed=seed)
        snapshots.append(_snapshot(env, {}, total_reward))
        done = False
        while not done:
            actions = policy(env, None)
            result = env.step(actions)
            total_reward += sum(result.rewards.values())
            snapshots.append(_snapshot(env, actions, total_reward))
            done = result.terminated or result.truncated
    finally:
        env.close()
    return snapshots, total_reward


def _save_comparison_png(
    baseline_snapshots: List[Snapshot],
    ppo_snapshots: List[Snapshot],
    baseline_name: str,
    baseline_reward: float,
    ppo_reward: float,
    config: UAVServiceEnvConfig,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5.6),
        dpi=150,
        gridspec_kw={"width_ratios": [1.25, 1.25, 0.9]},
    )
    _draw_map(axes[0], baseline_snapshots, len(baseline_snapshots) - 1, config, baseline_name, True)
    _draw_map(axes[1], ppo_snapshots, len(ppo_snapshots) - 1, config, "rllib-ppo", True)
    _draw_comparison_metrics(
        axes[2],
        baseline_name=baseline_name,
        baseline_snapshot=baseline_snapshots[-1],
        ppo_snapshot=ppo_snapshots[-1],
        baseline_reward=baseline_reward,
        ppo_reward=ppo_reward,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_comparison_gif(
    baseline_snapshots: List[Snapshot],
    ppo_snapshots: List[Snapshot],
    baseline_name: str,
    config: UAVServiceEnvConfig,
    output_path: Path,
    fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), dpi=120)
    frame_count = max(len(baseline_snapshots), len(ppo_snapshots))

    def update(frame_index: int):
        baseline_index = min(frame_index, len(baseline_snapshots) - 1)
        ppo_index = min(frame_index, len(ppo_snapshots) - 1)
        _draw_map(axes[0], baseline_snapshots, baseline_index, config, baseline_name, False)
        _draw_map(axes[1], ppo_snapshots, ppo_index, config, "rllib-ppo", False)
        return []

    animation = FuncAnimation(fig, update, frames=frame_count, interval=1000 / fps, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _draw_comparison_metrics(
    ax,
    baseline_name: str,
    baseline_snapshot: Snapshot,
    ppo_snapshot: Snapshot,
    baseline_reward: float,
    ppo_reward: float,
) -> None:
    baseline_metrics = baseline_snapshot["metrics"]
    ppo_metrics = ppo_snapshot["metrics"]
    ax.axis("off")
    rows = [
        ("reward", baseline_reward, ppo_reward),
        ("hits", baseline_metrics["hits"], ppo_metrics["hits"]),
        ("misses", baseline_metrics["misses"], ppo_metrics["misses"]),
        ("energy", baseline_metrics["total_energy_used"], ppo_metrics["total_energy_used"]),
        ("pending", baseline_metrics["pending_tasks"], ppo_metrics["pending_tasks"]),
        ("success", baseline_metrics["success_rate"], ppo_metrics["success_rate"]),
    ]
    lines = [
        "PPO Trajectory Comparison",
        "",
        f"{'metric':<10}{baseline_name:>10}{'ppo':>10}{'delta':>10}",
        "-" * 40,
    ]
    for name, baseline_value, ppo_value in rows:
        lines.append(
            f"{name:<10}"
            f"{baseline_value:>10.2f}"
            f"{ppo_value:>10.2f}"
            f"{(ppo_value - baseline_value):>10.2f}"
        )
    lines.extend(
        [
            "",
            "Map legend",
            "gray dots: edge devices",
            "circles: pending tasks",
            "squares: UAV starts",
            "triangles: final UAV positions",
            "dashed lines: active assignments",
        ]
    )
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9.5,
        family="monospace",
        linespacing=1.35,
    )


def _write_summary(
    output_path: Path,
    seed: int,
    baseline_name: str,
    baseline_snapshots: List[Snapshot],
    ppo_snapshots: List[Snapshot],
    baseline_reward: float,
    ppo_reward: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "baseline_policy": baseline_name,
        "baseline": _summary_payload(baseline_snapshots, baseline_reward),
        "rllib_ppo": _summary_payload(ppo_snapshots, ppo_reward),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def _summary_payload(snapshots: List[Snapshot], reward: float) -> dict:
    metrics = snapshots[-1]["metrics"]
    return {
        "total_reward": reward,
        "hits": metrics["hits"],
        "misses": metrics["misses"],
        "pending_tasks": metrics["pending_tasks"],
        "success_rate": metrics["success_rate"],
        "miss_rate": metrics["miss_rate"],
        "total_energy_used": metrics["total_energy_used"],
        "mean_pending_tasks": sum(
            snapshot["metrics"]["pending_tasks"] for snapshot in snapshots[1:]
        )
        / max(1, len(snapshots) - 1),
        "max_pending_tasks": max(
            (snapshot["metrics"]["pending_tasks"] for snapshot in snapshots),
            default=0.0,
        ),
        "actions_by_step": [snapshot["actions"] for snapshot in snapshots[1:]],
    }


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


if __name__ == "__main__":
    main()
