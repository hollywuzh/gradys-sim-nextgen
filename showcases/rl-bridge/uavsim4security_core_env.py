"""Core uavsim4security-backed environment for RL/MARL experiments.

The external simulator is a SimPy network simulator, not a Gym-like env. This
wrapper keeps the simulator as the system of record and exposes a small
reset/step interface that mirrors ``gradys_uav_service_env.py``.
"""

from __future__ import annotations

import copy
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

os.environ.setdefault("MPLBACKEND", "Agg")

SHOWCASE_DIR = Path(__file__).resolve().parent
UAVSIM_ROOT = SHOWCASE_DIR / "external" / "uavsim4security"
if str(UAVSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(UAVSIM_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import simpy  # noqa: E402
from entities.drone import Drone  # noqa: E402
from entities.packet import AckPacket, DataPacket  # noqa: E402
from simulator import simulator as simulator_module  # noqa: E402
from topology.virtual_force.vf_packet import VfPacket  # noqa: E402
from utils import config as uav_config  # noqa: E402


AgentId = str
DEFAULT_SENSING_RANGE = float(getattr(uav_config, "SENSING_RANGE", 750.0))


@dataclass
class UAVSim4SecurityEnvConfig:
    num_uavs: int = 4
    episode_duration: float = 3.0
    control_interval: float = 0.5
    candidate_limit: int = 3
    seed: Optional[int] = None
    routing_protocol: str = "RLLIB"
    route_policy: str = "action"
    attacks: tuple[str, ...] = ()
    attacker_ids: tuple[int, ...] = ()
    attack_probability: float = 1.0
    sensing_range: Optional[float] = None
    jammer_coords: Optional[tuple[float, float, float]] = None
    jammer_radius: float = 160.0
    jammer_power_w: float = 0.18
    enable_dataset_export: bool = False
    delivered_reward: float = 10.0
    generated_penalty: float = -0.05
    collision_penalty: float = -1.0
    security_event_penalty: float = -1.0
    delay_penalty: float = -0.001


@dataclass
class StepResult:
    observations: Dict[AgentId, List[float]]
    rewards: Dict[AgentId, float]
    terminated: bool
    truncated: bool
    infos: Dict[AgentId, dict]


@dataclass(frozen=True)
class RouteCandidate:
    drone_id: int
    distance_to_neighbor: float
    distance_to_destination: float
    progress_to_destination: float
    residual_energy: float
    is_destination: bool


_ACTIVE_CORE_ENV: Optional["UAVSim4SecurityCoreEnv"] = None
_ORIGINAL_CREATE_ROUTING_PROTOCOL = Drone._create_routing_protocol
_PATCHED = False


class UAVSim4SecurityCoreEnv:
    """A lightweight RL wrapper around the uavsim4security SimPy simulator."""

    def __init__(self, config: Optional[UAVSim4SecurityEnvConfig] = None) -> None:
        self.config = config or UAVSim4SecurityEnvConfig()
        self.agents = [f"uav_{idx}" for idx in range(self.config.num_uavs)]
        self._rng = random.Random(self.config.seed)
        self._env: Optional[simpy.Environment] = None
        self._simulator = None
        self._last_actions = {agent: 0 for agent in self.agents}
        self._last_rewards = {agent: 0.0 for agent in self.agents}
        self._route_decision_count = 0
        self._ack_timeout_count = 0
        self._recent_route_decisions: List[dict] = []
        _patch_external_simulator()

    @property
    def action_size(self) -> int:
        return self.config.candidate_limit + 1

    @property
    def observation_size(self) -> int:
        # x, y, z, residual energy, queue length, waiting length, time, has packet
        # plus K next-hop candidates:
        # valid, is destination, dx, dy, dz, distance-to-neighbor,
        # distance-to-destination, progress-to-destination.
        return 8 + 8 * self.config.candidate_limit

    @property
    def time(self) -> float:
        return self._time_us / 1e6

    @property
    def _time_us(self) -> float:
        if self._env is None:
            return 0.0
        return float(self._env.now)

    @property
    def _episode_duration_us(self) -> int:
        return int(self.config.episode_duration * 1e6)

    @property
    def _control_interval_us(self) -> int:
        return max(1, int(self.config.control_interval * 1e6))

    def reset(self, seed: Optional[int] = None) -> Dict[AgentId, List[float]]:
        global _ACTIVE_CORE_ENV
        if seed is not None:
            self._rng.seed(seed)
        elif self.config.seed is not None:
            self._rng.seed(self.config.seed)
        episode_seed = self._rng.randint(0, 2**31 - 1)
        _ACTIVE_CORE_ENV = self
        self._configure_external(episode_seed)

        self._env = simpy.Environment()
        channel_states = {
            idx: simpy.Resource(self._env, capacity=1)
            for idx in range(self.config.num_uavs)
        }
        self._last_actions = {agent: 0 for agent in self.agents}
        self._last_rewards = {agent: 0.0 for agent in self.agents}
        self._route_decision_count = 0
        self._ack_timeout_count = 0
        self._recent_route_decisions = []
        self._simulator = simulator_module.Simulator(
            seed=episode_seed,
            env=self._env,
            channel_states=channel_states,
            n_drones=self.config.num_uavs,
            total_simulation_time=self._episode_duration_us,
        )
        return self._observations()

    def step(self, actions: Dict[AgentId, int]) -> StepResult:
        if self._env is None or self._simulator is None:
            raise RuntimeError("Call reset() before step().")
        self._last_actions = {
            agent: int(actions.get(agent, 0))
            for agent in self.agents
        }

        before = self._metrics_counts()
        target_time = min(
            self._time_us + self._control_interval_us,
            self._episode_duration_us,
        )
        if target_time > self._time_us:
            self._env.run(until=target_time)
        after = self._metrics_counts()

        reward = self._reward_from_delta(before, after)
        rewards = {agent: reward for agent in self.agents}
        self._last_rewards = rewards
        terminated = self._time_us >= self._episode_duration_us
        infos = {agent: self._agent_info(agent) for agent in self.agents}
        return StepResult(self._observations(), rewards, terminated, False, infos)

    def close(self) -> None:
        self._env = None
        self._simulator = None

    def metrics_snapshot(self) -> Dict[str, float]:
        if self._simulator is None:
            return {
                "time": 0.0,
                "generated_packets": 0.0,
                "delivered_packets": 0.0,
                "packet_delivery_ratio": 0.0,
                "average_delay_ms": 0.0,
                "collision_count": 0.0,
                "security_event_count": 0.0,
                "route_decisions": 0.0,
                "ack_timeouts": 0.0,
                "mean_residual_energy": 0.0,
            }

        metrics = self._simulator.metrics
        generated = float(metrics.datapacket_generated_num)
        delivered = float(len(metrics.datapacket_arrived))
        delays = list(metrics.deliver_time_dict.values())
        pdr = delivered / generated if generated else 0.0
        residual_energy = [
            float(drone.residual_energy)
            for drone in self._simulator.drones
        ]
        return {
            "time": self.time,
            "generated_packets": generated,
            "delivered_packets": delivered,
            "packet_delivery_ratio": pdr,
            "average_delay_ms": _mean(delays) / 1e3 if delays else 0.0,
            "collision_count": float(metrics.collision_num),
            "security_event_count": float(sum(metrics.security_event_counts.values())),
            "route_decisions": float(self._route_decision_count),
            "ack_timeouts": float(self._ack_timeout_count),
            "mean_residual_energy": _mean(residual_energy),
        }

    def visualization_snapshot(self) -> Dict[str, object]:
        if self._simulator is None:
            return {"time": 0.0, "agent_positions": {}}
        return {
            "time": self.time,
            "map": {
                "length": float(uav_config.MAP_LENGTH),
                "width": float(uav_config.MAP_WIDTH),
                "height": float(uav_config.MAP_HEIGHT),
                "sensing_range": float(getattr(uav_config, "SENSING_RANGE", 0.0)),
            },
            "agent_positions": {
                agent: tuple(float(value) for value in self._drone(idx).coords)
                for idx, agent in enumerate(self.agents)
            },
            "agents": {
                agent: self._agent_visual_state(agent)
                for agent in self.agents
            },
            "attack_names": list(uav_config.ATTACK_NAMES),
            "attacker_ids": [int(item) for item in uav_config.ATTACKER_IDS],
            "jammer": self._jammer_snapshot(),
            "recent_route_decisions": list(self._recent_route_decisions[-80:]),
            "attack_events": list(getattr(self._simulator, "security_attack_events", [])[-80:]),
            "metrics": self.metrics_snapshot(),
        }

    def action_mask(self, agent: AgentId) -> List[float]:
        mask = [0.0] * self.action_size
        mask[0] = 1.0
        candidates = self.candidate_next_hops(agent)
        if not candidates:
            return [1.0] * self.action_size
        for idx in range(min(len(candidates), self.config.candidate_limit)):
            mask[idx + 1] = 1.0
        return mask

    def candidate_next_hops(self, agent: AgentId) -> List[RouteCandidate]:
        packet = self._reference_packet(agent)
        if packet is None:
            return []
        return self.candidate_next_hops_for_packet(_agent_index(agent), packet)

    def candidate_next_hops_for_packet(self, drone_id: int, packet: DataPacket) -> List[RouteCandidate]:
        if self._simulator is None:
            return []
        current = self._simulator.drones[drone_id]
        destination = packet.dst_drone
        current_to_dest = _distance_3d(current.coords, destination.coords)
        visited = set(getattr(packet, "intermediate_drones", []))
        candidates: List[RouteCandidate] = []
        for candidate in self._simulator.drones:
            if candidate.identifier == drone_id:
                continue
            if candidate.identifier in visited and candidate.identifier != destination.identifier:
                continue
            distance_to_neighbor = _distance_3d(current.coords, candidate.coords)
            if distance_to_neighbor > float(getattr(uav_config, "SENSING_RANGE", 750.0)):
                continue
            distance_to_destination = _distance_3d(candidate.coords, destination.coords)
            candidates.append(
                RouteCandidate(
                    drone_id=int(candidate.identifier),
                    distance_to_neighbor=distance_to_neighbor,
                    distance_to_destination=distance_to_destination,
                    progress_to_destination=current_to_dest - distance_to_destination,
                    residual_energy=float(candidate.residual_energy),
                    is_destination=candidate.identifier == destination.identifier,
                )
            )
        candidates.sort(
            key=lambda item: (
                not item.is_destination,
                -item.progress_to_destination,
                item.distance_to_destination,
                item.distance_to_neighbor,
            )
        )
        return candidates[: self.config.candidate_limit]

    def choose_next_hop_for_packet(self, drone_id: int, packet: DataPacket) -> Optional[int]:
        candidates = self.candidate_next_hops_for_packet(drone_id, packet)
        if not candidates:
            return None
        if self.config.route_policy.lower() == "defensive":
            return self._defensive_next_hop_for_packet(candidates)
        action = int(self._last_actions.get(f"uav_{drone_id}", 0))
        if action <= 0 or action > len(candidates):
            chosen = candidates[0]
        else:
            chosen = candidates[action - 1]
        return chosen.drone_id

    def _defensive_next_hop_for_packet(self, candidates: List[RouteCandidate]) -> Optional[int]:
        attackers = set(int(item) for item in uav_config.ATTACKER_IDS)
        jammer = self._jammer_snapshot()
        best_candidate = candidates[0]
        best_score = -float("inf")
        for candidate in candidates:
            score = (
                2.5 * candidate.progress_to_destination / max(1.0, _map_diagonal())
                - 0.7 * candidate.distance_to_neighbor / max(1.0, _map_diagonal())
                + 0.3 * candidate.residual_energy / max(1.0, uav_config.INITIAL_ENERGY)
            )
            if candidate.is_destination:
                score += 4.0
            elif candidate.drone_id in attackers:
                score -= 1.4
            if self._candidate_inside_jammer(candidate.drone_id, jammer):
                score -= 0.6
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate.drone_id

    def record_route_decision(self, drone_id: int, packet: DataPacket, next_hop_id: Optional[int]) -> None:
        self._route_decision_count += 1
        self._recent_route_decisions.append(
            {
                "time": self.time,
                "drone_id": int(drone_id),
                "packet_id": int(packet.packet_id),
                "src_id": int(packet.src_drone.identifier),
                "dst_id": int(packet.dst_drone.identifier),
                "next_hop_id": int(next_hop_id) if next_hop_id is not None else None,
                "success": next_hop_id is not None,
                "action": int(self._last_actions.get(f"uav_{drone_id}", 0)),
            }
        )
        if len(self._recent_route_decisions) > 500:
            self._recent_route_decisions = self._recent_route_decisions[-500:]
        if hasattr(self._simulator, "dataset_exporter"):
            recorder = getattr(self._simulator.dataset_exporter, "record_route_decision", None)
            if recorder is not None:
                recorder(
                    "RLLIB",
                    self._drone(drone_id),
                    packet,
                    next_hop_id is not None,
                    next_hop_id,
                    {"action": int(self._last_actions.get(f"uav_{drone_id}", 0))},
                )

    def record_ack_timeout(self) -> None:
        self._ack_timeout_count += 1

    def _configure_external(self, seed: int) -> None:
        uav_config.SIM_TIME = self._episode_duration_us
        uav_config.NUMBER_OF_DRONES = self.config.num_uavs
        uav_config.SIMULATION_SEED = seed
        uav_config.ROUTING_PROTOCOL = self.config.routing_protocol.upper()
        uav_config.ENABLE_VISUALIZATION = False
        uav_config.ENABLE_INTERACTIVE_VISUALIZATION = False
        uav_config.DATASET_EXPORT_ENABLED = bool(self.config.enable_dataset_export)
        uav_config.LOG_FILE = "running_log_uavsim4security_rllib.log"
        uav_config.ATTACK_NAMES = tuple(item.upper() for item in self.config.attacks)
        uav_config.ATTACKER_IDS = tuple(int(item) for item in self.config.attacker_ids)
        uav_config.ATTACK_PROBABILITY = float(self.config.attack_probability)
        uav_config.SENSING_RANGE = (
            float(self.config.sensing_range)
            if self.config.sensing_range is not None
            else DEFAULT_SENSING_RANGE
        )
        uav_config.ATTACK_START_US = 0
        uav_config.ATTACK_END_US = self._episode_duration_us
        uav_config.GL_ID_DATA_PACKET = 0
        uav_config.GL_ID_HELLO_PACKET = 10000
        uav_config.GL_ID_ACK_PACKET = 20000
        uav_config.GL_ID_VF_PACKET = 30000
        uav_config.GL_ID_GRAD_MESSAGE = 40000
        if "PHY_JAMMING" in uav_config.ATTACK_NAMES or "PHY_JAMMER" in uav_config.ATTACK_NAMES:
            uav_config.PHY_JAMMER_START_US = 0
            uav_config.PHY_JAMMER_END_US = self._episode_duration_us
            uav_config.PHY_JAMMER_RADIUS_M = float(self.config.jammer_radius)
            uav_config.PHY_JAMMER_POWER_W = float(self.config.jammer_power_w)
            if self.config.jammer_coords is not None:
                uav_config.PHY_JAMMER_COORDS = tuple(float(item) for item in self.config.jammer_coords)
            else:
                uav_config.PHY_JAMMER_COORDS = ()

    def _observations(self) -> Dict[AgentId, List[float]]:
        return {agent: self._observation(agent) for agent in self.agents}

    def _observation(self, agent: AgentId) -> List[float]:
        drone = self._drone(_agent_index(agent))
        reference_packet = self._reference_packet(agent)
        obs = [
            float(drone.coords[0]) / uav_config.MAP_LENGTH,
            float(drone.coords[1]) / uav_config.MAP_WIDTH,
            float(drone.coords[2]) / uav_config.MAP_HEIGHT,
            float(drone.residual_energy) / uav_config.INITIAL_ENERGY,
            float(drone.transmitting_queue.qsize()) / max(1, uav_config.MAX_QUEUE_SIZE),
            float(len(drone.waiting_list)) / max(1, uav_config.MAX_QUEUE_SIZE),
            self._time_us / max(1.0, self._episode_duration_us),
            1.0 if reference_packet is not None else 0.0,
        ]
        candidates = self.candidate_next_hops(agent)
        for candidate in candidates:
            candidate_drone = self._drone(candidate.drone_id)
            obs.extend(
                [
                    1.0,
                    1.0 if candidate.is_destination else 0.0,
                    (float(candidate_drone.coords[0]) - float(drone.coords[0])) / uav_config.MAP_LENGTH,
                    (float(candidate_drone.coords[1]) - float(drone.coords[1])) / uav_config.MAP_WIDTH,
                    (float(candidate_drone.coords[2]) - float(drone.coords[2])) / uav_config.MAP_HEIGHT,
                    candidate.distance_to_neighbor / _map_diagonal(),
                    candidate.distance_to_destination / _map_diagonal(),
                    candidate.progress_to_destination / _map_diagonal(),
                ]
            )
        while len(candidates) < self.config.candidate_limit:
            obs.extend([0.0] * 8)
            candidates.append(RouteCandidate(-1, 0.0, 0.0, 0.0, 0.0, False))
        return obs

    def _reference_packet(self, agent: AgentId) -> Optional[DataPacket]:
        if self._simulator is None:
            return None
        drone = self._drone(_agent_index(agent))
        for packet in list(getattr(drone.transmitting_queue, "queue", [])):
            if isinstance(packet, DataPacket):
                return packet
        for packet in list(drone.waiting_list):
            if isinstance(packet, DataPacket):
                return packet
        return None

    def _drone(self, drone_id: int):
        if self._simulator is None:
            raise RuntimeError("Call reset() before reading drones.")
        return self._simulator.drones[drone_id]

    def _agent_info(self, agent: AgentId) -> dict:
        metrics = self.metrics_snapshot()
        return {
            "time": metrics["time"],
            "generated_packets": metrics["generated_packets"],
            "delivered_packets": metrics["delivered_packets"],
            "packet_delivery_ratio": metrics["packet_delivery_ratio"],
            "collision_count": metrics["collision_count"],
            "route_decisions": metrics["route_decisions"],
            "ack_timeouts": metrics["ack_timeouts"],
        }

    def _agent_visual_state(self, agent: AgentId) -> dict:
        drone_id = _agent_index(agent)
        drone = self._drone(drone_id)
        return {
            "id": drone_id,
            "position": tuple(float(value) for value in drone.coords),
            "residual_energy": float(drone.residual_energy),
            "queue_length": int(drone.transmitting_queue.qsize()),
            "waiting_length": int(len(drone.waiting_list)),
            "is_attacker": drone_id in set(int(item) for item in uav_config.ATTACKER_IDS),
        }

    def _jammer_snapshot(self) -> Optional[dict]:
        if self._simulator is None:
            return None
        attack_manager = getattr(self._simulator, "attack_manager", None)
        if attack_manager is None:
            return None
        for attack in getattr(attack_manager, "attacks", []):
            jammer = getattr(attack, "jammer", None)
            if jammer is None:
                continue
            start_time = float(getattr(attack, "start_time_us", 0.0)) / 1e6
            end_time = float(getattr(attack, "end_time_us", 0.0)) / 1e6
            return {
                "id": str(jammer.identifier),
                "position": tuple(float(value) for value in jammer.coords),
                "radius": float(jammer.radius),
                "power_w": float(jammer.transmit_power),
                "active": start_time <= self.time <= end_time,
                "start_time": start_time,
                "end_time": end_time,
            }
        return None

    def _candidate_inside_jammer(self, drone_id: int, jammer: Optional[dict]) -> bool:
        if jammer is None or not jammer.get("active"):
            return False
        candidate = self._drone(drone_id)
        return _distance_3d(candidate.coords, jammer.get("position", (0.0, 0.0, 0.0))) <= float(jammer.get("radius", 0.0))

    def _metrics_counts(self) -> dict[str, float]:
        if self._simulator is None:
            return {}
        metrics = self._simulator.metrics
        return {
            "generated": float(metrics.datapacket_generated_num),
            "delivered": float(len(metrics.datapacket_arrived)),
            "collision": float(metrics.collision_num),
            "security": float(sum(metrics.security_event_counts.values())),
            "delay_us": float(sum(metrics.deliver_time_dict.values())),
        }

    def _reward_from_delta(self, before: dict[str, float], after: dict[str, float]) -> float:
        generated = after["generated"] - before["generated"]
        delivered = after["delivered"] - before["delivered"]
        collisions = after["collision"] - before["collision"]
        security_events = after["security"] - before["security"]
        delay_us = max(0.0, after["delay_us"] - before["delay_us"])
        return float(
            self.config.delivered_reward * delivered
            + self.config.generated_penalty * max(0.0, generated - delivered)
            + self.config.collision_penalty * collisions
            + self.config.security_event_penalty * security_events
            + self.config.delay_penalty * (delay_us / 1e3)
        )


class RLlibControlledRouting:
    """Routing protocol shim that delegates next-hop choice to the core env."""

    def __init__(self, core: UAVSim4SecurityCoreEnv, simulator, my_drone) -> None:
        self.core = core
        self.simulator = simulator
        self.my_drone = my_drone

    def next_hop_selection(self, packet):
        has_route = True
        enquire = False
        packet.intermediate_drones.append(self.my_drone.identifier)
        next_hop_id = self.core.choose_next_hop_for_packet(self.my_drone.identifier, packet)
        self.core.record_route_decision(self.my_drone.identifier, packet, next_hop_id)
        if next_hop_id is None or next_hop_id == self.my_drone.identifier:
            has_route = False
        else:
            packet.next_hop_id = next_hop_id
        return has_route, packet, enquire

    def packet_reception(self, packet, src_drone_id):
        if isinstance(packet, DataPacket):
            yield from self._receive_data(packet, src_drone_id)
        elif isinstance(packet, AckPacket):
            self._receive_ack(packet, src_drone_id)
        elif isinstance(packet, VfPacket):
            return

    def penalize(self, packet) -> None:
        self.core.record_ack_timeout()

    def _receive_data(self, packet, src_drone_id):
        packet_copy = copy.copy(packet)
        if packet_copy.dst_drone.identifier == self.my_drone.identifier:
            if packet_copy.packet_id not in self.simulator.metrics.datapacket_arrived:
                self.simulator.metrics.calculate_metrics(packet_copy)
        elif self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
            self.my_drone.transmitting_queue.put(packet_copy)
        else:
            return

        uav_config.GL_ID_ACK_PACKET += 1
        src_drone = self.simulator.drones[src_drone_id]
        ack_packet = AckPacket(
            src_drone=self.my_drone,
            dst_drone=src_drone,
            ack_packet_id=uav_config.GL_ID_ACK_PACKET,
            ack_packet_length=uav_config.ACK_PACKET_LENGTH,
            ack_packet=packet_copy,
            simulator=self.simulator,
            channel_id=packet_copy.channel_id,
        )
        yield self.simulator.env.timeout(uav_config.SIFS_DURATION)
        if not self.my_drone.sleep:
            ack_packet.increase_ttl()
            self.my_drone.mac_protocol.phy.unicast(ack_packet, src_drone_id)
            yield self.simulator.env.timeout(ack_packet.packet_length / uav_config.BIT_RATE * 1e6)
            self.simulator.drones[src_drone_id].receive()

    def _receive_ack(self, packet, src_drone_id) -> None:
        data_packet_acked = packet.ack_packet
        self.simulator.metrics.mac_delay.append(
            (self.simulator.env.now - data_packet_acked.first_attempt_time) / 1e3
        )
        self.my_drone.remove_from_queue(data_packet_acked)
        key = f"wait_ack{self.my_drone.identifier}_{data_packet_acked.packet_id}"
        wait_finish = self.my_drone.mac_protocol.wait_ack_process_finish
        wait_process = self.my_drone.mac_protocol.wait_ack_process_dict
        if wait_finish.get(key) == 0 and key in wait_process:
            if not wait_process[key].triggered:
                wait_finish[key] = 1
                wait_process[key].interrupt()


def flatten_agent_dict(values: Dict[AgentId, Iterable[float]], agents: List[AgentId]) -> List[float]:
    flat: List[float] = []
    for agent in agents:
        flat.extend(values[agent])
    return flat


def _patch_external_simulator() -> None:
    global _PATCHED
    if _PATCHED:
        return
    simulator_module.scatter_plot = _noop
    simulator_module.scatter_plot_with_obstacles = _noop
    plt.show = _noop
    Drone._create_routing_protocol = _patched_create_routing_protocol
    _PATCHED = True


def _patched_create_routing_protocol(self):
    if uav_config.ROUTING_PROTOCOL.upper() == "RLLIB":
        if _ACTIVE_CORE_ENV is None:
            raise RuntimeError("No active UAVSim4SecurityCoreEnv for RLLIB routing.")
        return RLlibControlledRouting(_ACTIVE_CORE_ENV, self.simulator, self)
    return _ORIGINAL_CREATE_ROUTING_PROTOCOL(self)


def _noop(*args, **kwargs):
    return None


def _agent_index(agent: AgentId) -> int:
    return int(agent.split("_", 1)[1])


def _distance_3d(start, end) -> float:
    return math.sqrt(
        (float(end[0]) - float(start[0])) ** 2
        + (float(end[1]) - float(start[1])) ** 2
        + (float(end[2]) - float(start[2])) ** 2
    )


def _map_diagonal() -> float:
    return math.sqrt(
        uav_config.MAP_LENGTH ** 2
        + uav_config.MAP_WIDTH ** 2
        + uav_config.MAP_HEIGHT ** 2
    )


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
