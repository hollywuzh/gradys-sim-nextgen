import logging

from routing.qmr.qmr_table import QMRTable
from routing.sqmr import sqmr_config

logger = logging.getLogger("network_routing")


class SQMRTable(QMRTable):
    def add_new_neighbor_entry(self, drone_id):
        super().add_new_neighbor_entry(drone_id)
        entry = self.table[drone_id]
        entry["trust_score"] = sqmr_config.initial_trust
        entry["trust_success_count"] = 0
        entry["trust_failure_count"] = 0

    def update_trust_on_success(self, neighbor_id):
        if neighbor_id not in self.table:
            return

        entry = self.table[neighbor_id]
        entry["trust_score"] = min(
            sqmr_config.max_trust,
            entry["trust_score"] + sqmr_config.success_reward_step,
        )
        entry["trust_success_count"] += 1

    def update_trust_on_failure(self, neighbor_id):
        if neighbor_id not in self.table:
            return

        entry = self.table[neighbor_id]
        entry["trust_score"] = max(
            sqmr_config.min_trust,
            entry["trust_score"] - sqmr_config.failure_penalty_step,
        )
        entry["trust_failure_count"] += 1

    def route_decision_qmr(self, packet, destination):
        cur_time = self.env.now
        cur_neighbor_ids = self.table.keys()

        candidate_neighbors, sub_candidate_neighbors, actual_velocity_dict, min_velocity = (
            self.filter_space_of_exploration(packet, destination, cur_time))

        if len(candidate_neighbors) > 0:
            chosen_neighbor_id = max(
                candidate_neighbors,
                key=lambda x: x[1] * self.table[x[0]]["q_value"] * self.table[x[0]].get("trust_score", 1.0),
            )[0]

        elif len(sub_candidate_neighbors) > 0:
            chosen_neighbor_id = max(
                sub_candidate_neighbors,
                key=lambda x: x[1] * self.table[x[0]].get("trust_score", 1.0),
            )[0]

        else:
            if len(cur_neighbor_ids) == 0:
                chosen_neighbor_id = self.my_drone.identifier
            else:
                chosen_neighbor_id = max(
                    cur_neighbor_ids,
                    key=lambda x: self.table[x]["q_value"] * self.table[x].get("trust_score", 1.0),
                )

        return chosen_neighbor_id
