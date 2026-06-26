from entities.packet import DataPacket
from simulator.log import logger

from security.attacks.base_attack import BaseAttack


class AckSpoofingAttack(BaseAttack):
    name = "ACK_SPOOFING"

    def __init__(self, manager):
        super().__init__(manager)
        self.spoofed_packets = set()

    def on_wait_ack_started(self, drone, packet, wait_ack_key):
        if not isinstance(packet, DataPacket):
            return

        if not self.is_attack_window_active():
            return

        if not self.manager.has_attackers():
            return

        if not self.sample_attack():
            return

        spoof_source_id = min(self.manager.context.attacker_ids)
        self.simulator.env.process(
            self._deliver_forged_ack(drone, packet, wait_ack_key, spoof_source_id)
        )

    def should_drop_received_packet(self, drone, packet, src_drone_id):
        acked_packet = getattr(packet, "ack_packet", None)
        if acked_packet is None:
            return False

        key = (drone.identifier, acked_packet.packet_id)
        if key not in self.spoofed_packets:
            return False

        self.spoofed_packets.remove(key)
        logger.info(
            'At time: %s (us) ---- SECURITY drops real ACK for packet: %s at UAV: %s after forged ACK already succeeded',
            self.simulator.env.now,
            acked_packet.packet_id,
            drone.identifier,
        )
        self.manager.record_security_event("ack_spoof_real_ack_suppressed_count")
        return True

    def _deliver_forged_ack(self, drone, packet, wait_ack_key, spoof_source_id):
        yield self.simulator.env.timeout(1)

        if drone.mac_protocol.wait_ack_process_finish.get(wait_ack_key) != 0:
            return

        self.spoofed_packets.add((drone.identifier, packet.packet_id))
        if packet.first_attempt_time is not None:
            drone.simulator.metrics.mac_delay.append((self.simulator.env.now - packet.first_attempt_time) / 1e3)

        drone.remove_from_queue(packet)
        drone.mac_protocol.wait_ack_process_finish[wait_ack_key] = 1

        wait_ack_process = drone.mac_protocol.wait_ack_process_dict.get(wait_ack_key)
        logger.info(
            'At time: %s (us) ---- SECURITY forged ACK for packet: %s at UAV: %s by attacker UAV: %s',
            self.simulator.env.now,
            packet.packet_id,
            drone.identifier,
            spoof_source_id,
        )
        self.manager.record_security_event("ack_spoof_success_count")

        if wait_ack_process is not None and not wait_ack_process.triggered:
            wait_ack_process.interrupt()
