from simulator.log import logger
from routing.q_routing.q_routing_packet import QRoutingAckPacket
from routing.qfanet.qfanet_packet import QFanetAckPacket
from routing.qmr.qmr_packet import QMRAckPacket
from routing.qgeo.qgeo_packet import QGeoAckPacket
from utils import config

from security.attacks.base_attack import BaseAttack


class AckPoisoningAttack(BaseAttack):
    name = "ACK_POISONING"

    def mutate_outgoing_packet(self, drone, packet, transmission_type, next_hop_id):
        if transmission_type != "unicast":
            return packet

        if not self.is_attack_window_active():
            return packet

        if not self.manager.is_attacker(drone.identifier):
            return packet

        if not self.sample_attack():
            return packet

        if isinstance(packet, QRoutingAckPacket):
            packet.queuing_delay = 0
            packet.min_q = 0
            self.manager.record_security_event("ack_poison_qrouting_count")
        elif isinstance(packet, QFanetAckPacket):
            packet.void_area_flag = 0
            packet.reward = max(packet.reward, getattr(drone.routing_protocol, "r_max", 100))
            packet.sinr_eta = 1.0
            self.manager.record_security_event("ack_poison_qfanet_count")
        elif isinstance(packet, QMRAckPacket):
            packet.max_q = max(packet.max_q, 1000.0)
            packet.is_local_minimum = False
            self.manager.record_security_event("ack_poison_qmr_count")
        elif isinstance(packet, QGeoAckPacket):
            packet.void_area_flag = 0
            packet.reward = max(packet.reward, getattr(drone.routing_protocol, "r_max", 10))
            packet.max_q = max(packet.max_q, getattr(drone.routing_protocol, "r_max", 10))
            self.manager.record_security_event("ack_poison_qgeo_count")
        else:
            return packet

        logger.info(
            'At time: %s (us) ---- SECURITY poisons ACK packet: %s at UAV: %s',
            self.simulator.env.now,
            packet.packet_id,
            drone.identifier,
        )
        self.manager.record_security_event("ack_poison_count")
        return packet
