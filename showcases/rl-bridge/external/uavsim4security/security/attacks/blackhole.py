from entities.packet import DataPacket
from simulator.log import logger

from security.attacks.base_attack import BaseAttack


class BlackholeAttack(BaseAttack):
    name = "BLACKHOLE"

    def should_drop_forwarded_packet(self, drone, packet):
        if not isinstance(packet, DataPacket):
            return False

        if not self.is_attack_window_active():
            return False

        if not self.manager.is_attacker_for(self.name, drone.identifier):
            return False

        # Blackhole applies to relayed traffic, not locally generated traffic or final delivery.
        if packet.src_drone.identifier == drone.identifier:
            return False

        if packet.dst_drone.identifier == drone.identifier:
            return False

        if not self.manager.sample_attack_with_probability(self.manager.get_probability_for(self.name)):
            return False

        logger.info(
            'At time: %s (us) ---- SECURITY blackhole drops packet: %s at UAV: %s',
            self.simulator.env.now,
            packet.packet_id,
            drone.identifier,
        )
        self.manager.record_attack_event(self.name, drone.identifier, packet.packet_id)
        self.manager.record_security_event("blackhole_drop_count")
        return True
