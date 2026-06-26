from simulator.log import logger
from utils import config

from security.attacks.base_attack import BaseAttack


class LocationSpoofingAttack(BaseAttack):
    name = "LOCATION_SPOOFING"

    def mutate_outgoing_packet(self, drone, packet, transmission_type, next_hop_id):
        if transmission_type not in {"broadcast", "unicast"}:
            return packet

        if not self.is_attack_window_active():
            return packet

        if not self.manager.is_attacker(drone.identifier):
            return packet

        if not self.sample_attack():
            return packet

        mutated = False
        spoofed_position = self._spoof_position(drone.coords)
        spoofed_velocity = self._spoof_velocity(drone.velocity)

        if hasattr(packet, "cur_position"):
            packet.cur_position = spoofed_position
            mutated = True

        if hasattr(packet, "cur_velocity"):
            packet.cur_velocity = spoofed_velocity
            mutated = True

        if hasattr(packet, "src_coords"):
            packet.src_coords = spoofed_position
            mutated = True

        if hasattr(packet, "src_velocity"):
            packet.src_velocity = spoofed_velocity
            mutated = True

        if hasattr(packet, "src_energy"):
            packet.src_energy = config.INITIAL_ENERGY
            mutated = True

        if hasattr(packet, "remain_energy"):
            packet.remain_energy = config.INITIAL_ENERGY
            mutated = True

        if mutated:
            logger.info(
                'At time: %s (us) ---- SECURITY location spoofing mutates packet: %s at UAV: %s',
                self.simulator.env.now,
                packet.packet_id,
                drone.identifier,
            )
            self.manager.record_security_event("location_spoof_count")

        return packet

    @staticmethod
    def _spoof_position(real_position):
        center = (
            config.MAP_LENGTH / 2,
            config.MAP_WIDTH / 2,
            config.MAP_HEIGHT / 2,
        )
        ratio = config.LOCATION_SPOOF_RATIO
        return tuple(
            max(0.0, min(bound, coord + ratio * (target - coord)))
            for coord, target, bound in zip(
                real_position,
                center,
                (config.MAP_LENGTH, config.MAP_WIDTH, config.MAP_HEIGHT),
            )
        )

    @staticmethod
    def _spoof_velocity(real_velocity):
        ratio = config.LOCATION_SPOOF_RATIO
        return [component * (1.0 + ratio) for component in real_velocity]
