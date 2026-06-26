import csv
import json
import os
from copy import deepcopy

from utils import config


class DatasetExporter:
    """
    Export routing decisions and feedback events as a lightweight training dataset.

    The first version focuses on QMR/SQMR-style hop-by-hop routing decisions and
    stores one transition record per forwarding decision.
    """

    def __init__(self, simulator):
        self.simulator = simulator
        self.enabled = config.DATASET_EXPORT_ENABLED
        self.records = []
        self.pending = {}
        self.packet_generation_records = []

        self.export_dir = os.path.join(os.getcwd(), config.DATASET_EXPORT_DIR)
        self.basename = config.DATASET_EXPORT_BASENAME

    def record_packet_generated(self, packet):
        if not self.enabled:
            return

        self.packet_generation_records.append({
            "packet_id": int(packet.packet_id),
            "created_at_us": int(packet.creation_time),
            "src_id": int(packet.src_drone.identifier),
            "dst_id": int(packet.dst_drone.identifier),
            "packet_length_bits": int(packet.packet_length),
            "deadline_us": int(packet.deadline),
            "channel_id": int(packet.channel_id),
        })

    def record_route_decision(self, protocol_name, drone, packet, has_route, chosen_neighbor_id, context):
        if not self.enabled:
            return

        key = self._transition_key(drone.identifier, packet.packet_id)
        record = {
            "protocol": str(protocol_name).upper(),
            "seed": int(self.simulator.seed),
            "sim_time_us": int(self.simulator.env.now),
            "decision_key": key,
            "packet_id": int(packet.packet_id),
            "src_id": int(packet.src_drone.identifier),
            "dst_id": int(packet.dst_drone.identifier),
            "current_node_id": int(drone.identifier),
            "has_route": bool(has_route),
            "chosen_next_hop_id": None if chosen_neighbor_id is None else int(chosen_neighbor_id),
            "packet_deadline_remaining_us": int(max(0, packet.creation_time + packet.deadline - self.simulator.env.now)),
            "packet_ttl": int(packet.get_current_ttl()),
            "packet_length_bits": int(packet.packet_length),
            "channel_id": int(packet.channel_id),
            "attack_names": list(config.ATTACK_NAMES),
            "state": context,
            "outcome": {
                "ack_status": "pending",
                "forwarding_status": "unknown",
                "is_local_minimum": None,
                "reward": None,
                "max_q": None,
                "next_state": None,
                "q_value_after": None,
                "trust_score_after": None,
                "security_event": None,
                "drop_reason": None,
            },
        }

        if not has_route or chosen_neighbor_id is None:
            record["outcome"]["ack_status"] = "no_route"
            self.records.append(record)
            return

        self.pending[key] = record

    def record_route_feedback(
        self,
        protocol_name,
        current_node_id,
        packet,
        next_hop_id,
        ack_status,
        reward=None,
        max_q=None,
        is_local_minimum=None,
        next_state=None,
        q_value_after=None,
        trust_score_after=None,
        security_event=None,
        drop_reason=None,
    ):
        if not self.enabled:
            return

        key = self._transition_key(current_node_id, packet.packet_id)
        record = self.pending.get(key)
        if record is None:
            return

        outcome = record["outcome"]
        outcome["ack_status"] = ack_status
        outcome["reward"] = self._safe_float(reward)
        outcome["max_q"] = self._safe_float(max_q)
        outcome["is_local_minimum"] = is_local_minimum
        outcome["next_state"] = next_state
        outcome["q_value_after"] = self._safe_float(q_value_after)
        outcome["trust_score_after"] = self._safe_float(trust_score_after)
        outcome["security_event"] = security_event
        outcome["drop_reason"] = drop_reason
        record["sim_time_feedback_us"] = int(self.simulator.env.now)
        record["protocol"] = str(protocol_name).upper()
        record["next_hop_id_feedback"] = int(next_hop_id) if next_hop_id is not None else None

        if ack_status in {"ack_timeout", "dropped"}:
            if outcome["forwarding_status"] == "unknown":
                outcome["forwarding_status"] = "not_observed"
            self.records.append(record)
            self.pending.pop(key, None)

    def record_forwarding_observation(self, current_node_id, packet_id, observed, trust_score_after=None, q_value_after=None):
        if not self.enabled:
            return

        key = self._transition_key(current_node_id, packet_id)
        record = self.pending.get(key)
        if record is None:
            return

        record["outcome"]["forwarding_status"] = "forwarded" if observed else "not_forwarded"
        if trust_score_after is not None:
            record["outcome"]["trust_score_after"] = self._safe_float(trust_score_after)
        if q_value_after is not None:
            record["outcome"]["q_value_after"] = self._safe_float(q_value_after)

    def finalize(self):
        if not self.enabled:
            return None

        os.makedirs(self.export_dir, exist_ok=True)

        for record in list(self.pending.values()):
            if record["outcome"]["ack_status"] == "pending":
                record["outcome"]["ack_status"] = "unfinished"
            if record["outcome"]["forwarding_status"] == "unknown":
                record["outcome"]["forwarding_status"] = "not_observed"
            self.records.append(record)
        self.pending.clear()

        jsonl_path = os.path.join(self.export_dir, f"{self.basename}.jsonl")
        csv_path = os.path.join(self.export_dir, f"{self.basename}.csv")
        packets_path = os.path.join(self.export_dir, f"{self.basename}_packets.json")
        meta_path = os.path.join(self.export_dir, f"{self.basename}_meta.json")
        transitions_jsonl_path = os.path.join(self.export_dir, f"{self.basename}_transitions.jsonl")
        transitions_csv_path = os.path.join(self.export_dir, f"{self.basename}_transitions.csv")

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        csv_rows = [self._flatten_record(record) for record in self.records]
        if csv_rows:
            fieldnames = list(csv_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)

        training_samples = [self._build_training_sample(record) for record in self.records]
        with open(transitions_jsonl_path, "w", encoding="utf-8") as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        if training_samples:
            transition_rows = [self._flatten_training_sample(sample) for sample in training_samples]
            fieldnames = list(transition_rows[0].keys())
            with open(transitions_csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(transition_rows)

        with open(packets_path, "w", encoding="utf-8") as f:
            json.dump(self.packet_generation_records, f, ensure_ascii=False, indent=2)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "protocol": config.ROUTING_PROTOCOL,
                "seed": self.simulator.seed,
                "sim_time_us": self.simulator.total_simulation_time,
                "attack_names": list(config.ATTACK_NAMES),
                "attackers": list(config.ATTACKER_IDS),
                "num_records": len(self.records),
                "num_generated_packets": len(self.packet_generation_records),
                "num_training_samples": len(training_samples),
            }, f, ensure_ascii=False, indent=2)

        return {
            "jsonl_path": jsonl_path,
            "csv_path": csv_path,
            "packets_path": packets_path,
            "meta_path": meta_path,
            "transitions_jsonl_path": transitions_jsonl_path,
            "transitions_csv_path": transitions_csv_path,
            "num_records": len(self.records),
        }

    @staticmethod
    def _transition_key(current_node_id, packet_id):
        return f"{current_node_id}:{packet_id}"

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None
        return float(value)

    def _flatten_record(self, record):
        state = deepcopy(record["state"])
        outcome = deepcopy(record["outcome"])
        flat = {
            "protocol": record["protocol"],
            "seed": record["seed"],
            "sim_time_us": record["sim_time_us"],
            "sim_time_feedback_us": record.get("sim_time_feedback_us"),
            "packet_id": record["packet_id"],
            "src_id": record["src_id"],
            "dst_id": record["dst_id"],
            "current_node_id": record["current_node_id"],
            "chosen_next_hop_id": record["chosen_next_hop_id"],
            "has_route": record["has_route"],
            "packet_deadline_remaining_us": record["packet_deadline_remaining_us"],
            "packet_ttl": record["packet_ttl"],
            "packet_length_bits": record["packet_length_bits"],
            "channel_id": record["channel_id"],
            "attack_names": ",".join(record["attack_names"]),
            "ack_status": outcome["ack_status"],
            "forwarding_status": outcome["forwarding_status"],
            "reward": outcome["reward"],
            "max_q": outcome["max_q"],
            "q_value_after": outcome["q_value_after"],
            "trust_score_after": outcome["trust_score_after"],
            "is_local_minimum": outcome["is_local_minimum"],
            "security_event": outcome["security_event"],
            "drop_reason": outcome["drop_reason"],
            "neighbor_count": state.get("neighbor_count"),
            "candidate_count": len(state.get("candidate_neighbors", [])),
            "sub_candidate_count": len(state.get("sub_candidate_neighbors", [])),
            "chosen_snapshot_json": json.dumps(state.get("chosen_neighbor_snapshot"), ensure_ascii=False),
            "state_json": json.dumps(state, ensure_ascii=False),
            "next_state_json": json.dumps(outcome["next_state"], ensure_ascii=False),
        }
        return flat

    def _build_training_sample(self, record):
        state = deepcopy(record["state"])
        next_state = deepcopy(record["outcome"].get("next_state"))
        chosen_snapshot = deepcopy(state.get("chosen_neighbor_snapshot") or {})
        next_chosen_snapshot = deepcopy((next_state or {}).get("chosen_neighbor_snapshot") or {})
        packet_events = self._get_packet_attack_events(record["packet_id"])

        reward = record["outcome"].get("reward")
        if reward is None:
            reward = self._fallback_reward(record["outcome"].get("ack_status"))

        sample = {
            "protocol": record["protocol"],
            "seed": record["seed"],
            "transition_id": record["decision_key"],
            "packet_id": record["packet_id"],
            "src_id": record["src_id"],
            "dst_id": record["dst_id"],
            "current_node_id": record["current_node_id"],
            "time_us": record["sim_time_us"],
            "feedback_time_us": record.get("sim_time_feedback_us"),
            "action": {
                "chosen_next_hop_id": record["chosen_next_hop_id"],
                "has_route": record["has_route"],
            },
            "reward": self._safe_float(reward),
            "terminal": record["outcome"].get("ack_status") in {"ack_timeout", "dropped", "no_route", "unfinished"},
            "attack_context": {
                "attack_names": list(record["attack_names"]),
                "attack_active": bool(record["attack_names"]),
                "packet_attack_event_count": len(packet_events),
                "packet_attack_events": packet_events,
                "watchdog_forwarded": record["outcome"].get("forwarding_status") == "forwarded",
                "watchdog_failed": record["outcome"].get("forwarding_status") == "not_forwarded",
                "drop_reason": record["outcome"].get("drop_reason"),
                "security_event": record["outcome"].get("security_event"),
            },
            "outcome": {
                "ack_status": record["outcome"].get("ack_status"),
                "forwarding_status": record["outcome"].get("forwarding_status"),
                "is_local_minimum": record["outcome"].get("is_local_minimum"),
                "max_q": self._safe_float(record["outcome"].get("max_q")),
                "q_value_after": self._safe_float(record["outcome"].get("q_value_after")),
                "trust_score_after": self._safe_float(record["outcome"].get("trust_score_after")),
            },
            "state_features": self._build_feature_block(
                record,
                state,
                chosen_snapshot,
                state_prefix="state",
            ),
            "next_state_features": self._build_feature_block(
                record,
                next_state or {},
                next_chosen_snapshot,
                state_prefix="next_state",
            ),
            "raw_state": state,
            "raw_next_state": next_state,
        }
        return sample

    def _build_feature_block(self, record, state, chosen_snapshot, state_prefix):
        chosen_snapshot = chosen_snapshot or {}
        return {
            "neighbor_count": state.get("neighbor_count"),
            "candidate_count": len(state.get("candidate_neighbors", [])),
            "sub_candidate_count": len(state.get("sub_candidate_neighbors", [])),
            "required_velocity": self._safe_float(state.get("required_velocity")),
            "packet_deadline_remaining_us": record["packet_deadline_remaining_us"],
            "packet_ttl": record["packet_ttl"],
            "chosen_next_hop_id": record["chosen_next_hop_id"],
            "chosen_lq": self._safe_float(chosen_snapshot.get("lq")),
            "chosen_k_factor": self._safe_float(chosen_snapshot.get("k_factor")),
            "chosen_q_value": self._safe_float(chosen_snapshot.get("q_value")),
            "chosen_trust_score": self._safe_float(chosen_snapshot.get("trust_score", 1.0)),
            "chosen_delay_us": self._safe_float(chosen_snapshot.get("delay_us")),
            "chosen_remain_energy": self._safe_float(chosen_snapshot.get("remain_energy")),
            "chosen_actual_velocity": self._safe_float(chosen_snapshot.get("actual_velocity")),
            "chosen_is_candidate": chosen_snapshot.get("is_candidate"),
            "chosen_is_sub_candidate": chosen_snapshot.get("is_sub_candidate"),
            "lq_bucket": self._bucket_lq(chosen_snapshot.get("lq")),
            "delay_bucket": self._bucket_delay_us(chosen_snapshot.get("delay_us")),
            "trust_bucket": self._bucket_trust(chosen_snapshot.get("trust_score", 1.0)),
            "energy_bucket": self._bucket_energy(chosen_snapshot.get("remain_energy")),
            "velocity_bucket": self._bucket_velocity(chosen_snapshot.get("actual_velocity")),
            "attack_indicator": 1 if record["attack_names"] else 0,
            "watchdog_failure_indicator": 1 if record["outcome"].get("forwarding_status") == "not_forwarded" else 0,
            "watchdog_success_indicator": 1 if record["outcome"].get("forwarding_status") == "forwarded" else 0,
            "state_label": state_prefix,
        }

    def _flatten_training_sample(self, sample):
        state_features = deepcopy(sample["state_features"])
        next_state_features = deepcopy(sample["next_state_features"])
        row = {
            "protocol": sample["protocol"],
            "seed": sample["seed"],
            "transition_id": sample["transition_id"],
            "packet_id": sample["packet_id"],
            "src_id": sample["src_id"],
            "dst_id": sample["dst_id"],
            "current_node_id": sample["current_node_id"],
            "time_us": sample["time_us"],
            "feedback_time_us": sample["feedback_time_us"],
            "action_next_hop_id": sample["action"]["chosen_next_hop_id"],
            "action_has_route": sample["action"]["has_route"],
            "reward": sample["reward"],
            "terminal": sample["terminal"],
            "ack_status": sample["outcome"]["ack_status"],
            "forwarding_status": sample["outcome"]["forwarding_status"],
            "is_local_minimum": sample["outcome"]["is_local_minimum"],
            "max_q": sample["outcome"]["max_q"],
            "q_value_after": sample["outcome"]["q_value_after"],
            "trust_score_after": sample["outcome"]["trust_score_after"],
            "attack_names": ",".join(sample["attack_context"]["attack_names"]),
            "attack_active": sample["attack_context"]["attack_active"],
            "packet_attack_event_count": sample["attack_context"]["packet_attack_event_count"],
            "watchdog_forwarded": sample["attack_context"]["watchdog_forwarded"],
            "watchdog_failed": sample["attack_context"]["watchdog_failed"],
            "drop_reason": sample["attack_context"]["drop_reason"],
            "security_event": sample["attack_context"]["security_event"],
            "raw_state_json": json.dumps(sample["raw_state"], ensure_ascii=False),
            "raw_next_state_json": json.dumps(sample["raw_next_state"], ensure_ascii=False),
        }
        for key, value in state_features.items():
            row[f"s_{key}"] = value
        for key, value in next_state_features.items():
            row[f"sp_{key}"] = value
        return row

    def _get_packet_attack_events(self, packet_id):
        events = []
        for event in getattr(self.simulator, "security_attack_events", []):
            if event.get("packet_id") == packet_id:
                events.append({
                    "time_us": int(event.get("time_us", 0)),
                    "attack_name": event.get("attack_name"),
                    "drone_id": event.get("drone_id"),
                })
        return events

    @staticmethod
    def _fallback_reward(ack_status):
        if ack_status == "ack_received":
            return 1.0
        if ack_status == "no_route":
            return -2.0
        if ack_status == "ack_timeout":
            return -10.0
        if ack_status == "dropped":
            return -10.0
        return 0.0

    @staticmethod
    def _bucket_lq(value):
        if value is None:
            return "unknown"
        value = float(value)
        if value < 0.4:
            return "poor"
        if value < 0.75:
            return "fair"
        return "good"

    @staticmethod
    def _bucket_delay_us(value):
        if value is None:
            return "unknown"
        value = float(value)
        if value < 2e5:
            return "small"
        if value < 5e5:
            return "medium"
        return "large"

    @staticmethod
    def _bucket_trust(value):
        if value is None:
            return "unknown"
        value = float(value)
        if value < 0.35:
            return "low"
        if value < 0.75:
            return "medium"
        return "high"

    @staticmethod
    def _bucket_energy(value):
        if value is None:
            return "unknown"
        ratio = float(value) / float(config.INITIAL_ENERGY)
        if ratio < 0.3:
            return "low"
        if ratio < 0.7:
            return "medium"
        return "high"

    @staticmethod
    def _bucket_velocity(value):
        if value is None:
            return "unknown"
        value = float(value)
        if value < 0:
            return "negative"
        if value < 30:
            return "weak"
        return "strong"
