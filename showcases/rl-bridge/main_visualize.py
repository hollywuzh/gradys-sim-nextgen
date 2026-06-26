"""Visualize one GrADyS RL-bridge episode as a trajectory PNG and GIF."""

from __future__ import annotations

import argparse
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "gradys-rl-mpl"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from baseline_policies import BASELINE_POLICIES
from gradys_uav_service_env import GradysUAVServiceCoreEnv, Task, UAVServiceEnvConfig


Snapshot = Dict[str, object]
SHOWCASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    args = _parse_args()
    config = UAVServiceEnvConfig(
        num_uavs=args.num_uavs,
        num_devices=args.num_devices,
        episode_duration=args.episode_duration,
        control_interval=args.control_interval,
        task_arrival_probability=args.task_arrival_probability,
        seed=args.seed,
    )
    output_dir = _resolve_showcase_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_output = (
        _resolve_showcase_path(args.png_output)
        if args.png_output
        else output_dir / f"{args.policy}_episode.png"
    )
    gif_output = (
        _resolve_showcase_path(args.gif_output)
        if args.gif_output
        else output_dir / f"{args.policy}_episode.gif"
    )

    snapshots, total_reward = _record_episode(config, args.policy, args.seed)
    _save_summary_png(snapshots, config, args.policy, total_reward, png_output)
    if not args.no_gif:
        _save_gif(snapshots, config, args.policy, gif_output, args.fps)

    metrics = snapshots[-1]["metrics"]
    print(f"Policy: {args.policy}")
    print(
        "Final metrics: "
        f"hits={metrics['hits']:.0f} misses={metrics['misses']:.0f} "
        f"pending={metrics['pending_tasks']:.0f} success={metrics['success_rate']:.3f} "
        f"reward={total_reward:.2f}"
    )
    print(f"Wrote trajectory PNG: {png_output.resolve()}")
    if not args.no_gif:
        print(f"Wrote animation GIF: {gif_output.resolve()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=sorted(BASELINE_POLICIES), default="nearest")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-uavs", type=int, default=4)
    parser.add_argument("--num-devices", type=int, default=30)
    parser.add_argument("--episode-duration", type=float, default=60.0)
    parser.add_argument("--control-interval", type=float, default=1.0)
    parser.add_argument("--task-arrival-probability", type=float, default=0.06)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--png-output")
    parser.add_argument("--gif-output")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def _resolve_showcase_path(path: str) -> Path:
    output_path = Path(path)
    if output_path.is_absolute():
        return output_path
    return SHOWCASE_DIR / output_path


def _record_episode(
    config: UAVServiceEnvConfig,
    policy_name: str,
    seed: int,
) -> tuple[List[Snapshot], float]:
    env = GradysUAVServiceCoreEnv(config)
    rng = random.Random(seed + 101)
    policy = BASELINE_POLICIES[policy_name]
    snapshots: List[Snapshot] = []
    total_reward = 0.0

    try:
        env.reset(seed=seed)
        snapshots.append(_snapshot(env, {}, total_reward))
        done = False
        while not done:
            actions = policy(env, rng)
            result = env.step(actions)
            total_reward += sum(result.rewards.values())
            snapshots.append(_snapshot(env, actions, total_reward))
            done = result.terminated or result.truncated
    finally:
        env.close()

    return snapshots, total_reward


def _snapshot(
    env: GradysUAVServiceCoreEnv,
    actions: Dict[str, int],
    total_reward: float,
) -> Snapshot:
    state = env.visualization_snapshot()
    return {
        "time": state["time"],
        "agent_positions": state["agent_positions"],
        "device_positions": state["device_positions"],
        "pending_tasks": state["pending_tasks"],
        "targets": state["targets"],
        "actions": dict(actions),
        "metrics": env.metrics_snapshot(),
        "total_reward": total_reward,
    }


def _save_summary_png(
    snapshots: List[Snapshot],
    config: UAVServiceEnvConfig,
    policy_name: str,
    total_reward: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_map, ax_text) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        dpi=150,
        gridspec_kw={"width_ratios": [2.3, 1.0]},
    )
    _draw_map(ax_map, snapshots, len(snapshots) - 1, config, policy_name, full_history=True)
    _draw_metrics(ax_text, snapshots[-1], policy_name, total_reward)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_gif(
    snapshots: List[Snapshot],
    config: UAVServiceEnvConfig,
    policy_name: str,
    output_path: Path,
    fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)

    def update(frame_index: int):
        _draw_map(ax, snapshots, frame_index, config, policy_name, full_history=False)
        return []

    animation = FuncAnimation(fig, update, frames=len(snapshots), interval=1000 / fps, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _draw_map(
    ax,
    snapshots: List[Snapshot],
    frame_index: int,
    config: UAVServiceEnvConfig,
    policy_name: str,
    full_history: bool,
) -> None:
    snapshot = snapshots[frame_index]
    ax.clear()
    ax.set_xlim(0.0, config.area_size)
    ax.set_ylim(0.0, config.area_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e8edf3", linewidth=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"{policy_name} policy | t={snapshot['time']:.1f}s | "
        f"pending={len(snapshot['pending_tasks'])}"
    )

    _draw_devices(ax, snapshot)
    _draw_pending_tasks(ax, snapshot)
    _draw_uav_paths(ax, snapshots, frame_index, full_history)
    _draw_target_links(ax, snapshot)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)


def _draw_devices(ax, snapshot: Snapshot) -> None:
    devices = snapshot["device_positions"]
    if not devices:
        return
    xs = [position[0] for position in devices]
    ys = [position[1] for position in devices]
    ax.scatter(xs, ys, s=14, marker=".", color="#a8b0bb", alpha=0.55, label="edge devices")


def _draw_pending_tasks(ax, snapshot: Snapshot) -> None:
    tasks: List[Task] = snapshot["pending_tasks"]
    if not tasks:
        return
    now = float(snapshot["time"])
    xs = [task.location[0] for task in tasks]
    ys = [task.location[1] for task in tasks]
    slacks = [max(0.0, task.deadline - now) for task in tasks]
    sizes = [30.0 + min(95.0, task.compute_demand * 0.35) for task in tasks]
    points = ax.scatter(
        xs,
        ys,
        s=sizes,
        c=slacks,
        cmap="YlOrRd_r",
        edgecolor="#6f3b00",
        linewidth=0.6,
        alpha=0.85,
        label="pending tasks",
    )
    points.set_clim(0.0, 60.0)


def _draw_uav_paths(
    ax,
    snapshots: List[Snapshot],
    frame_index: int,
    full_history: bool,
) -> None:
    colors = plt.get_cmap("tab10")
    agents = list(snapshots[0]["agent_positions"].keys())
    start_index = 0 if full_history else max(0, frame_index - 18)
    history = snapshots[start_index : frame_index + 1]
    for index, agent in enumerate(agents):
        color = colors(index % 10)
        xs = [frame["agent_positions"][agent][0] for frame in history]
        ys = [frame["agent_positions"][agent][1] for frame in history]
        ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.82, label=f"{agent} path")
        ax.scatter(xs[0], ys[0], marker="s", s=45, color=color, edgecolor="white", linewidth=0.8)
        ax.scatter(xs[-1], ys[-1], marker="^", s=95, color=color, edgecolor="black", linewidth=0.7)
        ax.text(xs[-1] + 1.1, ys[-1] + 1.1, agent, fontsize=8, color=color)


def _draw_target_links(ax, snapshot: Snapshot) -> None:
    task_by_id = {task.task_id: task for task in snapshot["pending_tasks"]}
    for agent, task_id in snapshot["targets"].items():
        if task_id is None or task_id not in task_by_id:
            continue
        agent_position = snapshot["agent_positions"][agent]
        task = task_by_id[task_id]
        ax.plot(
            [agent_position[0], task.location[0]],
            [agent_position[1], task.location[1]],
            color="#2f343b",
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
        )


def _draw_metrics(
    ax,
    snapshot: Snapshot,
    policy_name: str,
    total_reward: float,
) -> None:
    metrics = snapshot["metrics"]
    ax.axis("off")
    lines = [
        "Episode Summary",
        "",
        f"policy: {policy_name}",
        f"time: {metrics['time']:.1f} s",
        f"generated tasks: {metrics['generated_tasks']:.0f}",
        f"pending tasks: {metrics['pending_tasks']:.0f}",
        f"hits: {metrics['hits']:.0f}",
        f"misses: {metrics['misses']:.0f}",
        f"success rate: {metrics['success_rate']:.3f}",
        f"miss rate: {metrics['miss_rate']:.3f}",
        f"energy used: {metrics['total_energy_used']:.1f}",
        f"total reward: {total_reward:.2f}",
        "",
        "Markers",
        "gray dots: edge devices",
        "triangles: UAVs",
        "squares: UAV starts",
        "orange/red circles: pending tasks",
        "dashed lines: active assignments",
    ]
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        linespacing=1.35,
    )


if __name__ == "__main__":
    main()
