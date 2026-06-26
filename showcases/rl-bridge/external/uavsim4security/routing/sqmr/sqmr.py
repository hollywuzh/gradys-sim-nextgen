import copy
import random

from entities.packet import DataPacket
from simulator.log import logger
from routing.qmr.history_packets_recorder import HistoryPacketsRecorder
from routing.qmr.qmr_packet import QMRHelloPacket, QMRAckPacket
from routing.sqmr import sqmr_config
from routing.sqmr.sqmr_table import SQMRTable
from utils import config


class SQMR:
    """
    Secure QMR: QMR enhanced with lightweight forwarding-trust defense.

    The core idea is to watch whether the chosen next hop actually forwards
    the packet afterwards. Missing forwarding evidence decreases trust and
    future route preference for that neighbor.
    """

    def __init__(self, simulator, my_drone):
        self.simulator = simulator
        self.my_drone = my_drone

        self.table = SQMRTable(simulator.env, my_drone)
        self.history_packet_recorder = HistoryPacketsRecorder(self.simulator.n_drones)
        self.rng_routing = random.Random(self.my_drone.identifier + self.my_drone.simulator.seed + 10)

        self.eps = 0.8
        self.hello_interval = 0.5 * 1e6
        self.pending_forward_checks = {}

        self.simulator.register_transmission_observer(self.observe_unicast_transmission)
        self.simulator.env.process(self.broadcast_hello_packet_periodically())
        self.simulator.env.process(self.check_waiting_list())
        self.simulator.env.process(self.update_discounted_factor())
        self.simulator.env.process(self.update_eps())
        self.simulator.env.process(self.monitor_forwarding_watchdog())

    def broadcast_hello_packet(self, my_drone):
        config.GL_ID_HELLO_PACKET += 1
        channel_id = self.my_drone.channel_assigner.channel_assign()
        cur_time = self.simulator.env.now
        received_hello_packet_tuple = self.history_packet_recorder.get_all_active_received_hello_packet_count(cur_time)

        hello_pkt = QMRHelloPacket(
            src_drone=my_drone,
            creation_time=cur_time,
            id_hello_packet=config.GL_ID_HELLO_PACKET,
            hello_packet_length=config.HELLO_PACKET_LENGTH,
            received_hello_packet_count=received_hello_packet_tuple,
            simulator=self.simulator,
            channel_id=channel_id
        )
        hello_pkt.transmission_mode = 1

        logger.info('At time: %s (us) ---- UAV: %s has a hello packet to broadcast',
                    self.simulator.env.now, self.my_drone.identifier)

        self.history_packet_recorder.add_sent_hello_packet(hello_pkt)
        self.simulator.metrics.control_packet_num += 1
        self.my_drone.transmitting_queue.put(hello_pkt)

    def broadcast_hello_packet_periodically(self):
        while True:
            self.broadcast_hello_packet(self.my_drone)
            jitter = self.rng_routing.randint(1000, 2000)
            yield self.simulator.env.timeout(self.hello_interval + jitter)

    def next_hop_selection(self, packet):
        enquire = False
        has_route = True

        self.table.purge()
        dst_drone = packet.dst_drone
        packet.intermediate_drones.append(self.my_drone.identifier)
        next_hop_id = self.table.make_route_decision(packet, dst_drone, self.eps)

        if next_hop_id == self.my_drone.identifier:
            has_route = False
        else:
            packet.next_hop_id = next_hop_id

        self._record_route_decision(packet, has_route, next_hop_id if has_route else None)

        if has_route:
            self.history_packet_recorder.add_sent_data_packet(packet)

        return has_route, packet, enquire

    def update_discounted_factor(self):
        while True:
            self.table.update_discounted_factor()
            yield self.simulator.env.timeout(sqmr_config.discount_factor_update_interval)

    def packet_reception(self, packet, src_drone_id):
        cur_time = self.simulator.env.now

        if isinstance(packet, QMRHelloPacket):
            self.table.update_neighbor(packet, cur_time)
            self.history_packet_recorder.add_received_hello_packet(packet)

        elif isinstance(packet, DataPacket):
            packet_copy = copy.copy(packet)
            packet_copy.previous_drone = self.simulator.drones[src_drone_id]
            queuing_delay = packet_copy.transmitting_start_time - packet_copy.waiting_start_time

            config.GL_ID_ACK_PACKET += 1
            src_drone = self.simulator.drones[src_drone_id]
            max_q = self.table.get_max_q()
            is_local_minimum = self.table.check_local_minimum(packet.dst_drone)

            ack_packet = QMRAckPacket(
                creation_time=cur_time,
                src_drone=self.my_drone,
                dst_drone=src_drone,
                ack_packet_id=config.GL_ID_ACK_PACKET,
                ack_packet_length=config.ACK_PACKET_LENGTH,
                ack_packet=packet_copy,
                transmitting_start_time=cur_time,
                queuing_delay=queuing_delay,
                max_q=max_q,
                is_local_minimum=is_local_minimum,
                source_packet_backoff_start_time=packet.first_attempt_time,
                simulator=self.simulator,
                channel_id=packet_copy.channel_id
            )
            yield self.simulator.env.timeout(config.SIFS_DURATION)

            if not self.my_drone.sleep:
                ack_packet.increase_ttl()
                self.my_drone.mac_protocol.phy.unicast(ack_packet, src_drone_id)
                yield self.simulator.env.timeout(ack_packet.packet_length / config.BIT_RATE * 1e6)
                self.simulator.drones[src_drone_id].receive()

            if packet_copy.dst_drone.identifier == self.my_drone.identifier:
                if packet_copy.packet_id not in self.simulator.metrics.datapacket_arrived:
                    self.simulator.metrics.calculate_metrics(packet_copy)
            else:
                if self.my_drone.transmitting_queue.qsize() < config.MAX_QUEUE_SIZE:
                    logger.info('At time: %s (us) ---- Data packet: %s is received by next hop UAV: %s',
                                self.simulator.env.now, packet_copy.packet_id, self.my_drone.identifier)
                    self.my_drone.transmitting_queue.put(packet_copy)

        elif isinstance(packet, QMRAckPacket):
            original_packet = packet.ack_packet
            cur_time = self.simulator.env.now
            mac_delay = cur_time - packet.source_packet_backoff_start_time
            self.simulator.metrics.mac_delay.append(mac_delay / 1e3)
            self.table.add_mac_delay(mac_delay, cur_time, packet.src_drone.identifier)
            self.history_packet_recorder.add_received_ack_packet(packet)
            self.update(packet, src_drone_id)

            key2 = f"wait_ack{self.my_drone.identifier}_{original_packet.packet_id}"
            if self.my_drone.mac_protocol.wait_ack_process_finish[key2] == 0:
                if not self.my_drone.mac_protocol.wait_ack_process_dict[key2].triggered:
                    logger.info('At time: %s, the wait_ack process (id: %s) of UAV: %s is interrupted by UAV: %s',
                                self.simulator.env.now, key2, self.my_drone.identifier, src_drone_id)
                    self.my_drone.mac_protocol.wait_ack_process_finish[key2] = 1
                    self.my_drone.mac_protocol.wait_ack_process_dict[key2].interrupt()

    def update(self, packet, next_hop_id):
        origin_data_packet = packet.ack_packet
        dst_drone = origin_data_packet.dst_drone
        max_q = packet.max_q
        f = 1 if next_hop_id == dst_drone.identifier else 0
        self.table.update_q_value(f, max_q, next_hop_id, packet.is_local_minimum, dst_drone)
        self._record_route_feedback(
            packet=origin_data_packet,
            next_hop_id=next_hop_id,
            ack_status="ack_received",
            max_q=max_q,
            is_local_minimum=packet.is_local_minimum,
        )

    def check_waiting_list(self):
        while True:
            if not self.my_drone.sleep:
                yield self.simulator.env.timeout(0.6 * 1e6)
                for waiting_pkd in list(self.my_drone.waiting_list):
                    if self.simulator.env.now > waiting_pkd.creation_time + waiting_pkd.deadline:
                        self.my_drone.waiting_list.remove(waiting_pkd)
                    else:
                        dst_drone = waiting_pkd.dst_drone
                        best_next_hop_id = self.table.make_route_decision(waiting_pkd, dst_drone, self.eps)
                        if best_next_hop_id != self.my_drone.identifier:
                            self.my_drone.transmitting_queue.put(waiting_pkd)
                            self.my_drone.waiting_list.remove(waiting_pkd)
            else:
                break

    def penalty_for_ack_loss(self, packet):
        next_hop_id = packet.next_hop_id
        f = 1 if next_hop_id == packet.dst_drone.identifier else 0
        q_max = self.table.get_last_max_q_value_of_neighbor(next_hop_id)
        is_penalty = True
        self.table.update_q_value(f, q_max, next_hop_id, is_penalty)
        self._record_route_feedback(
            packet=packet,
            next_hop_id=next_hop_id,
            ack_status="ack_timeout",
            max_q=q_max,
            is_local_minimum=is_penalty,
            drop_reason="ack_timeout",
        )

    def update_eps(self):
        while True:
            yield self.simulator.env.timeout(self.hello_interval)
            self.eps = max(sqmr_config.eps_decay * self.eps, 0.2)

    def penalize(self, packet):
        self.penalty_for_ack_loss(packet)

    def observe_unicast_transmission(self, packet, src_drone_id, dst_drone_id, transmit_time_us):
        if not isinstance(packet, DataPacket):
            return

        pending_check = self.pending_forward_checks.get(packet.packet_id)
        if (
            pending_check is not None
            and pending_check["next_hop_id"] == src_drone_id
            and not pending_check["confirmed"]
        ):
            pending_check["confirmed"] = True
            self.table.update_trust_on_success(src_drone_id)
            trust_score = self.table.table[src_drone_id]["trust_score"] if src_drone_id in self.table.table else None
            q_value = self.table.table[src_drone_id]["q_value"] if src_drone_id in self.table.table else None
            self.simulator.dataset_exporter.record_forwarding_observation(
                current_node_id=self.my_drone.identifier,
                packet_id=packet.packet_id,
                observed=True,
                trust_score_after=trust_score,
                q_value_after=q_value,
            )
            self.simulator.metrics.record_security_event("sqmr_forward_success_count")
            logger.info(
                'At time: %s (us) ---- SQMR trust confirms forwarding of packet: %s by UAV: %s at UAV: %s',
                self.simulator.env.now,
                packet.packet_id,
                src_drone_id,
                self.my_drone.identifier,
            )
            return

        if src_drone_id != self.my_drone.identifier:
            return

        if dst_drone_id == packet.dst_drone.identifier:
            return

        self.pending_forward_checks[packet.packet_id] = {
            "next_hop_id": dst_drone_id,
            "expire_time": transmit_time_us + sqmr_config.forwarding_watchdog_timeout,
            "confirmed": False,
        }

    def monitor_forwarding_watchdog(self):
        while True:
            yield self.simulator.env.timeout(sqmr_config.watchdog_check_interval)

            expired_packet_ids = []
            for packet_id, check in self.pending_forward_checks.items():
                if check["confirmed"]:
                    expired_packet_ids.append(packet_id)
                    continue

                if self.simulator.env.now >= check["expire_time"]:
                    next_hop_id = check["next_hop_id"]
                    self.table.update_trust_on_failure(next_hop_id)
                    if next_hop_id in self.table.table:
                        self.table.table[next_hop_id]["q_value"] = max(
                            0.05,
                            self.table.table[next_hop_id]["q_value"] * 0.6,
                        )
                        trust_score = self.table.table[next_hop_id]["trust_score"]
                        q_value = self.table.table[next_hop_id]["q_value"]
                    else:
                        trust_score = None
                        q_value = None
                    self.simulator.dataset_exporter.record_forwarding_observation(
                        current_node_id=self.my_drone.identifier,
                        packet_id=packet_id,
                        observed=False,
                        trust_score_after=trust_score,
                        q_value_after=q_value,
                    )
                    self.simulator.metrics.record_security_event("sqmr_forward_failure_count")
                    logger.info(
                        'At time: %s (us) ---- SQMR trust penalizes UAV: %s for missing forwarding evidence of packet: %s at UAV: %s',
                        self.simulator.env.now,
                        next_hop_id,
                        packet_id,
                        self.my_drone.identifier,
                    )
                    expired_packet_ids.append(packet_id)

            for packet_id in expired_packet_ids:
                self.pending_forward_checks.pop(packet_id, None)

    def _record_route_decision(self, packet, has_route, chosen_neighbor_id):
        context = self.table.build_serializable_routing_snapshot(
            packet,
            packet.dst_drone,
            chosen_neighbor_id=chosen_neighbor_id,
        )
        self.simulator.dataset_exporter.record_route_decision(
            protocol_name="SQMR",
            drone=self.my_drone,
            packet=packet,
            has_route=has_route,
            chosen_neighbor_id=chosen_neighbor_id,
            context=context,
        )

    def _record_route_feedback(self, packet, next_hop_id, ack_status, max_q=None, is_local_minimum=None, drop_reason=None):
        next_state = self.table.build_serializable_routing_snapshot(
            packet,
            packet.dst_drone,
            chosen_neighbor_id=next_hop_id,
        )
        reward = None
        trust_score_after = None
        if next_hop_id in self.table.table:
            reward = self.table.get_reward(
                1 if next_hop_id == packet.dst_drone.identifier else 0,
                ack_status != "ack_received",
                next_hop_id,
            )
            q_value_after = self.table.table[next_hop_id]["q_value"]
            trust_score_after = self.table.table[next_hop_id].get("trust_score")
        else:
            q_value_after = None

        self.simulator.dataset_exporter.record_route_feedback(
            protocol_name="SQMR",
            current_node_id=self.my_drone.identifier,
            packet=packet,
            next_hop_id=next_hop_id,
            ack_status=ack_status,
            reward=reward,
            max_q=max_q,
            is_local_minimum=is_local_minimum,
            next_state=next_state,
            q_value_after=q_value_after,
            trust_score_after=trust_score_after,
            drop_reason=drop_reason,
        )
