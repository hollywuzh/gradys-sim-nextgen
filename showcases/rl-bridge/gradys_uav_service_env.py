"""Core GrADyS-backed environment for RL/MARL experiments.

This file intentionally avoids hard dependencies on Gymnasium, PettingZoo, or
RLlib. Those libraries can wrap this core through the adapter files in the same
directory.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from gradysim.simulator.handler.mobility import (
    DynamicVelocityMobilityConfiguration,
    DynamicVelocityMobilityHandler,
)
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration

from rl_protocols import PassiveEdgeDeviceProtocol, RLControlledUAVProtocol
from workload import AlibabaClusterV2017Workload, SyntheticWorkload, WorkloadProvider

Position2D = Tuple[float, float]
Position3D = Tuple[float, float, float]
AgentId = str


@dataclass(frozen=True)
class Task:
    task_id: int
    device_id: int
    location: Position2D
    arrival_time: float
    deadline: float
    compute_demand: float
    data_size: float


@dataclass
class UAVServiceEnvConfig:
    num_uavs: int = 4
    num_devices: int = 30
    area_size: float = 100.0
    altitude: float = 30.0
    episode_duration: float = 120.0
    control_interval: float = 1.0
    mobility_update_rate: float = 0.2
    max_speed_xy: float = 12.0
    max_speed_z: float = 4.0
    max_acc_xy: float = 8.0
    max_acc_z: float = 4.0
    candidate_limit: int = 5
    service_radius: float = 2.0
    compute_rate: float = 50.0
    task_arrival_probability: float = 0.08
    deadline_range: Tuple[float, float] = (20.0, 60.0)
    compute_demand_range: Tuple[float, float] = (20.0, 200.0)
    data_size_range: Tuple[float, float] = (0.5, 8.0)
    workload_mode: str = "synthetic"
    alibaba_task_table_path: Optional[str] = None
    alibaba_max_rows: int = 50000
    success_reward: float = 10.0
    miss_penalty: float = -10.0
    movement_penalty: float = -0.01
    wait_penalty: float = -0.02
    seed: Optional[int] = None


@dataclass
class StepResult:
    observations: Dict[AgentId, List[float]]
    rewards: Dict[AgentId, float]
    terminated: bool
    truncated: bool
    infos: Dict[AgentId, dict]


class GradysUAVServiceCoreEnv:
    """A lightweight multi-UAV service environment backed by GrADyS-SIM."""

    def __init__(self, config: Optional[UAVServiceEnvConfig] = None) -> None:
        self.config = config or UAVServiceEnvConfig()
        self.agents = [f"uav_{idx}" for idx in range(self.config.num_uavs)]
        self._rng = random.Random(self.config.seed)
        self._simulation = None
        self._uav_node_ids: Dict[AgentId, int] = {}
        self._device_positions: List[Position2D] = []
        self._tasks: List[Task] = []
        self._targets: Dict[AgentId, Optional[int]] = {}
        self._busy_until: Dict[AgentId, float] = {}
        self._energy_used: Dict[AgentId, float] = {}
        self._task_seq = 0
        self._hits = 0
        self._misses = 0
        self._last_rewards: Dict[AgentId, float] = {}
        self._workload = self._build_workload()
        self._compute_observation_scale = max(
            self.config.compute_demand_range[1],
            self._workload.max_compute_demand,
        )
        self._data_observation_scale = max(
            self.config.data_size_range[1],
            self._workload.max_data_size,
        )

    @property
    def action_size(self) -> int:
        return self.config.candidate_limit + 1

    @property
    def observation_size(self) -> int:
        # x, y, energy, busy remaining, queue length
        # plus K candidates: dx, dy, distance, deadline slack,
        # estimated service slack, compute demand, data size.
        return 5 + 7 * self.config.candidate_limit

    @property
    def time(self) -> float:
        if self._simulation is None:
            return 0.0
        return float(getattr(self._simulation, "_current_timestamp", 0.0))

    def reset(self, seed: Optional[int] = None) -> Dict[AgentId, List[float]]:
        if seed is not None:
            self._rng.seed(seed)
        elif self.config.seed is not None:
            self._rng.seed(self.config.seed)

        self._tasks = []
        self._targets = {agent: None for agent in self.agents}
        self._busy_until = {agent: 0.0 for agent in self.agents}
        self._energy_used = {agent: 0.0 for agent in self.agents}
        self._last_rewards = {agent: 0.0 for agent in self.agents}
        self._task_seq = 0
        self._hits = 0
        self._misses = 0

        self._device_positions = [
            (
                self._rng.uniform(0.0, self.config.area_size),
                self._rng.uniform(0.0, self.config.area_size),
            )
            for _ in range(self.config.num_devices)
        ]

        mobility_config = DynamicVelocityMobilityConfiguration(
            update_rate=self.config.mobility_update_rate,
            max_speed_xy=self.config.max_speed_xy,
            max_speed_z=self.config.max_speed_z,
            max_acc_xy=self.config.max_acc_xy,
            max_acc_z=self.config.max_acc_z,
            send_telemetry=True,
            telemetry_decimation=1,
        )

        builder = SimulationBuilder(
            SimulationConfiguration(
                duration=self.config.episode_duration,
                real_time=False,
                execution_logging=False,
            )
        )
        builder.add_handler(TimerHandler())
        builder.add_handler(DynamicVelocityMobilityHandler(mobility_config))

        center = self.config.area_size / 2.0
        self._uav_node_ids = {}
        for agent in self.agents:
            node_id = builder.add_node(
                RLControlledUAVProtocol,
                (center, center, self.config.altitude),
            )
            self._uav_node_ids[agent] = node_id

        for x_pos, y_pos in self._device_positions:
            builder.add_node(PassiveEdgeDeviceProtocol, (x_pos, y_pos, 0.0))

        self._simulation = builder.build()
        self._prime_simulation()
        self._generate_tasks()
        return self._observations()

    def step(self, actions: Dict[AgentId, int]) -> StepResult:
        if self._simulation is None:
            raise RuntimeError("Call reset() before step().")

        rewards = {agent: self.config.wait_penalty * len(self._tasks) for agent in self.agents}
        self._expire_tasks(rewards)
        self._assign_new_targets(actions)
        self._command_uav_velocities()

        time_before = self.time
        self._advance_until(min(time_before + self.config.control_interval, self.config.episode_duration))
        self._update_target_arrivals(rewards)
        self._generate_tasks()

        for agent in self.agents:
            rewards[agent] += self.config.movement_penalty * self._distance_moved(agent, time_before)

        terminated = self.time >= self.config.episode_duration
        infos = {
            agent: {
                "time": self.time,
                "pending_tasks": len(self._tasks),
                "hits": self._hits,
                "misses": self._misses,
                "energy_used": self._energy_used[agent],
            }
            for agent in self.agents
        }
        self._last_rewards = rewards
        return StepResult(self._observations(), rewards, terminated, False, infos)

    def close(self) -> None:
        if self._simulation is not None:
            for agent in self.agents:
                self._uav_protocol(agent).set_velocity((0.0, 0.0, 0.0))
        self._simulation = None

    def render_text(self) -> str:
        return (
            f"t={self.time:.1f}s pending={len(self._tasks)} "
            f"hits={self._hits} misses={self._misses}"
        )

    def visualization_snapshot(self) -> Dict[str, object]:
        """Return a read-only-friendly snapshot for plotting and animation."""
        return {
            "time": self.time,
            "agent_positions": {
                agent: self.agent_position(agent)
                for agent in self.agents
            },
            "device_positions": list(self._device_positions),
            "pending_tasks": list(self._tasks),
            "targets": dict(self._targets),
            "hits": self._hits,
            "misses": self._misses,
        }

    def agent_position(self, agent: AgentId) -> Position3D:
        """Return the current 3D position of one UAV agent."""
        return self._uav_position(agent)

    def metrics_snapshot(self) -> Dict[str, float]:
        """Return episode-level metrics for baselines and training callbacks."""
        completed_or_missed = self._hits + self._misses
        total_energy = sum(self._energy_used.values())
        success_rate = self._hits / completed_or_missed if completed_or_missed else 0.0
        miss_rate = self._misses / completed_or_missed if completed_or_missed else 0.0
        return {
            "time": self.time,
            "generated_tasks": float(self._task_seq),
            "pending_tasks": float(len(self._tasks)),
            "hits": float(self._hits),
            "misses": float(self._misses),
            "completed_or_missed_tasks": float(completed_or_missed),
            "success_rate": success_rate,
            "miss_rate": miss_rate,
            "total_energy_used": total_energy,
            "mean_energy_used": total_energy / max(1, len(self.agents)),
        }

    def candidate_tasks(self, agent: AgentId) -> List[Task]:
        position = self._uav_position(agent)
        assigned = {task_id for task_id in self._targets.values() if task_id is not None}
        candidates = [task for task in self._tasks if task.task_id not in assigned]
        candidates.sort(
            key=lambda task: (
                task.deadline - self.time,
                _distance_2d((position[0], position[1]), task.location),
            )
        )
        return candidates[: self.config.candidate_limit]

    def _prime_simulation(self) -> None:
        # One event is enough to initialize handlers and protocols. If no event
        # exists, the simulator has no active handlers and the environment is invalid.
        if not self._simulation.step_simulation():
            raise RuntimeError("GrADyS simulation ended during reset().")

    def _advance_until(self, target_time: float) -> None:
        while self.time < target_time and not self._simulation.is_simulation_done():
            if not self._simulation.step_simulation():
                break

    def _generate_tasks(self) -> None:
        now = self.time
        for device_id, location in enumerate(self._device_positions):
            if self._rng.random() > self.config.task_arrival_probability:
                continue
            workload = self._workload.sample(self._rng)
            task = Task(
                task_id=self._task_seq,
                device_id=device_id,
                location=location,
                arrival_time=now,
                deadline=now + workload.deadline_delta,
                compute_demand=workload.compute_demand,
                data_size=workload.data_size,
            )
            self._task_seq += 1
            self._tasks.append(task)

    def _expire_tasks(self, rewards: Dict[AgentId, float]) -> None:
        now = self.time
        expired = [task for task in self._tasks if task.deadline <= now]
        if not expired:
            return
        expired_ids = {task.task_id for task in expired}
        self._tasks = [task for task in self._tasks if task.task_id not in expired_ids]
        self._misses += len(expired)
        for agent, target_id in list(self._targets.items()):
            if target_id in expired_ids:
                self._targets[agent] = None
                rewards[agent] += self.config.miss_penalty

    def _assign_new_targets(self, actions: Dict[AgentId, int]) -> None:
        for agent in self.agents:
            if self._busy_until[agent] > self.time or self._targets[agent] is not None:
                continue
            action = int(actions.get(agent, 0))
            if action <= 0:
                continue
            candidates = self.candidate_tasks(agent)
            index = action - 1
            if index < len(candidates):
                self._targets[agent] = candidates[index].task_id

    def _command_uav_velocities(self) -> None:
        for agent in self.agents:
            protocol = self._uav_protocol(agent)
            if self._busy_until[agent] > self.time:
                protocol.set_velocity((0.0, 0.0, 0.0))
                continue
            task = self._target_task(agent)
            if task is None:
                protocol.set_velocity((0.0, 0.0, 0.0))
                continue
            velocity = self._velocity_toward(agent, task.location)
            protocol.set_velocity(velocity)

    def _update_target_arrivals(self, rewards: Dict[AgentId, float]) -> None:
        for agent in self.agents:
            if self._busy_until[agent] > self.time:
                continue
            task = self._target_task(agent)
            if task is None:
                continue
            position = self._uav_position(agent)
            if _distance_2d((position[0], position[1]), task.location) > self.config.service_radius:
                continue

            completion_time = self.time + task.compute_demand / self.config.compute_rate
            self._busy_until[agent] = completion_time
            self._targets[agent] = None
            self._tasks = [candidate for candidate in self._tasks if candidate.task_id != task.task_id]

            if completion_time <= task.deadline:
                self._hits += 1
                rewards[agent] += self.config.success_reward
            else:
                self._misses += 1
                rewards[agent] += self.config.miss_penalty

    def _observations(self) -> Dict[AgentId, List[float]]:
        return {agent: self._observation(agent) for agent in self.agents}

    def _observation(self, agent: AgentId) -> List[float]:
        x_pos, y_pos, _ = self._uav_position(agent)
        obs = [
            x_pos / self.config.area_size,
            y_pos / self.config.area_size,
            self._energy_used[agent] / 1000.0,
            max(0.0, self._busy_until[agent] - self.time) / self.config.episode_duration,
            len(self._tasks) / max(1, self.config.num_devices),
        ]
        candidates = self.candidate_tasks(agent)
        for task in candidates:
            distance = _distance_2d((x_pos, y_pos), task.location)
            travel_time = distance / max(self.config.max_speed_xy, 1e-9)
            service_time = task.compute_demand / max(self.config.compute_rate, 1e-9)
            obs.extend(
                [
                    (task.location[0] - x_pos) / self.config.area_size,
                    (task.location[1] - y_pos) / self.config.area_size,
                    distance / self.config.area_size,
                    (task.deadline - self.time) / self.config.episode_duration,
                    (task.deadline - self.time - travel_time - service_time)
                    / self.config.episode_duration,
                    task.compute_demand / self._compute_observation_scale,
                    task.data_size / self._data_observation_scale,
                ]
            )
        while len(candidates) < self.config.candidate_limit:
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            candidates.append(Task(-1, -1, (0.0, 0.0), 0.0, 0.0, 0.0, 0.0))
        return obs

    def _uav_protocol(self, agent: AgentId) -> RLControlledUAVProtocol:
        node_id = self._uav_node_ids[agent]
        return self._simulation.get_node(node_id).protocol_encapsulator.protocol

    def _uav_position(self, agent: AgentId) -> Position3D:
        node_id = self._uav_node_ids[agent]
        return tuple(self._simulation.get_node(node_id).position)

    def _target_task(self, agent: AgentId) -> Optional[Task]:
        target_id = self._targets.get(agent)
        if target_id is None:
            return None
        for task in self._tasks:
            if task.task_id == target_id:
                return task
        self._targets[agent] = None
        return None

    def _velocity_toward(self, agent: AgentId, location: Position2D) -> Position3D:
        x_pos, y_pos, _ = self._uav_position(agent)
        dx = location[0] - x_pos
        dy = location[1] - y_pos
        norm = math.hypot(dx, dy)
        if norm <= self.config.service_radius:
            return (0.0, 0.0, 0.0)
        speed = self.config.max_speed_xy
        return (speed * dx / norm, speed * dy / norm, 0.0)

    def _distance_moved(self, agent: AgentId, time_before: float) -> float:
        # Lightweight proxy for energy accounting. The detailed path is owned by
        # GrADyS; at this scaffold stage we only account for commanded motion.
        protocol = self._uav_protocol(agent)
        vx, vy, vz = protocol.commanded_velocity
        interval = max(0.0, self.time - time_before)
        distance = math.sqrt(vx * vx + vy * vy + vz * vz) * interval
        self._energy_used[agent] += distance
        return distance

    def _build_workload(self) -> WorkloadProvider:
        if self.config.workload_mode == "synthetic":
            return SyntheticWorkload(
                deadline_range=self.config.deadline_range,
                compute_demand_range=self.config.compute_demand_range,
                data_size_range=self.config.data_size_range,
            )
        if self.config.workload_mode == "alibaba_v2017":
            if not self.config.alibaba_task_table_path:
                raise ValueError(
                    "alibaba_task_table_path is required when workload_mode='alibaba_v2017'."
                )
            return AlibabaClusterV2017Workload.from_csv(
                self.config.alibaba_task_table_path,
                max_rows=self.config.alibaba_max_rows,
                deadline_range=self.config.deadline_range,
                compute_demand_range=self.config.compute_demand_range,
                data_size_range=self.config.data_size_range,
            )
        raise ValueError(f"Unknown workload_mode={self.config.workload_mode!r}.")


def flatten_agent_dict(values: Dict[AgentId, Iterable[float]], agents: List[AgentId]) -> List[float]:
    flat: List[float] = []
    for agent in agents:
        flat.extend(values[agent])
    return flat


def _distance_2d(start: Position2D, end: Position2D) -> float:
    return math.hypot(end[0] - start[0], end[1] - start[1])
