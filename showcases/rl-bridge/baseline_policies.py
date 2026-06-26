"""Rule-based baselines for the GrADyS RL bridge showcase."""

from __future__ import annotations

import math
import random
from typing import Callable, Dict

from gradys_uav_service_env import AgentId, GradysUAVServiceCoreEnv, Task

ActionDict = Dict[AgentId, int]
BaselinePolicy = Callable[[GradysUAVServiceCoreEnv, random.Random], ActionDict]


def hover_policy(env: GradysUAVServiceCoreEnv, rng: random.Random) -> ActionDict:
    """Keep all UAVs idle."""
    return {agent: 0 for agent in env.agents}


def random_policy(env: GradysUAVServiceCoreEnv, rng: random.Random) -> ActionDict:
    """Choose a random candidate index for each UAV."""
    return {agent: rng.randrange(env.action_size) for agent in env.agents}


def earliest_deadline_first_policy(env: GradysUAVServiceCoreEnv, rng: random.Random) -> ActionDict:
    """Serve the locally most urgent candidate first.

    The core environment already orders candidates by deadline and then
    distance, so action 1 selects the EDF candidate when one exists.
    """
    actions: ActionDict = {}
    for agent in env.agents:
        actions[agent] = 1 if env.candidate_tasks(agent) else 0
    return actions


def nearest_request_first_policy(env: GradysUAVServiceCoreEnv, rng: random.Random) -> ActionDict:
    """Serve the nearest visible request for each UAV."""
    actions: ActionDict = {}
    for agent in env.agents:
        candidates = env.candidate_tasks(agent)
        if not candidates:
            actions[agent] = 0
            continue
        x_pos, y_pos, _ = env.agent_position(agent)
        best_index = min(
            range(len(candidates)),
            key=lambda index: _distance((x_pos, y_pos), candidates[index].location),
        )
        actions[agent] = best_index + 1
    return actions


def slo_risk_policy(env: GradysUAVServiceCoreEnv, rng: random.Random) -> ActionDict:
    """Select the candidate with the smallest estimated SLO slack.

    This baseline approximates an online schedulability test:

    remaining slack = deadline - now - travel time - compute time

    The most negative slack is the highest-risk task. It is a useful comparator
    for UAV-SLO orchestration because it blends mobility and service time.
    """
    actions: ActionDict = {}
    for agent in env.agents:
        candidates = env.candidate_tasks(agent)
        if not candidates:
            actions[agent] = 0
            continue
        x_pos, y_pos, _ = env.agent_position(agent)
        best_index = min(
            range(len(candidates)),
            key=lambda index: _estimated_slack(env, (x_pos, y_pos), candidates[index]),
        )
        actions[agent] = best_index + 1
    return actions


BASELINE_POLICIES: Dict[str, BaselinePolicy] = {
    "hover": hover_policy,
    "random": random_policy,
    "nearest": nearest_request_first_policy,
    "edf": earliest_deadline_first_policy,
    "slo-risk": slo_risk_policy,
}


def _estimated_slack(
    env: GradysUAVServiceCoreEnv,
    position,
    task: Task,
) -> float:
    speed = max(env.config.max_speed_xy, 1e-9)
    compute_rate = max(env.config.compute_rate, 1e-9)
    travel_time = _distance(position, task.location) / speed
    service_time = task.compute_demand / compute_rate
    return task.deadline - env.time - travel_time - service_time


def _distance(start, end) -> float:
    return math.hypot(end[0] - start[0], end[1] - start[1])
