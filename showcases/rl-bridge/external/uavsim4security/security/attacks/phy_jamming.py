from dataclasses import dataclass

from simulator.log import logger
from utils import config
from utils.util_function import euclidean_distance_3d

from security.attacks.base_attack import BaseAttack


@dataclass
class StaticJammer:
    identifier: str
    coords: tuple[float, float, float]
    transmit_power: float
    radius: float


class PhyJammingAttack(BaseAttack):
    name = "PHY_JAMMING"

    def __init__(self, manager):
        super().__init__(manager)
        self.jammer = None
        self.start_time_us = config.PHY_JAMMER_START_US
        self.end_time_us = min(config.PHY_JAMMER_END_US, config.SIM_TIME)

    def initialize(self):
        if config.PHY_JAMMER_COORDS and len(config.PHY_JAMMER_COORDS) == 3:
            coords = tuple(config.PHY_JAMMER_COORDS)
        else:
            coords = self._compute_network_centroid()

        self.jammer = StaticJammer(
            identifier="JAMMER_0",
            coords=coords,
            transmit_power=config.PHY_JAMMER_POWER_W,
            radius=config.PHY_JAMMER_RADIUS_M,
        )

        logger.info(
            'At time: %s (us) ---- SECURITY initializes PHY jammer at %s with power %.3f W, radius %.1f m, window (%s, %s) us',
            self.simulator.env.now,
            self.jammer.coords,
            self.jammer.transmit_power,
            self.jammer.radius,
            self.start_time_us,
            self.end_time_us,
        )

    def get_external_interferers(self, receiver, channel_id):
        if self.jammer is None:
            self.initialize()

        if not self.is_attack_window_active():
            return []

        if not self.sample_attack():
            return []

        distance = euclidean_distance_3d(receiver.coords, self.jammer.coords)
        if distance > self.jammer.radius:
            return []

        self.manager.record_security_event("phy_jamming_interference_count")
        return [self.jammer]

    def is_attack_window_active(self):
        return self.start_time_us <= self.simulator.env.now <= self.end_time_us

    def _compute_network_centroid(self):
        if not self.simulator.drones:
            return (
                config.MAP_LENGTH / 2,
                config.MAP_WIDTH / 2,
                config.MAP_HEIGHT / 2,
            )

        x_mean = sum(drone.coords[0] for drone in self.simulator.drones) / len(self.simulator.drones)
        y_mean = sum(drone.coords[1] for drone in self.simulator.drones) / len(self.simulator.drones)
        z_mean = sum(drone.coords[2] for drone in self.simulator.drones) / len(self.simulator.drones)
        return (x_mean, y_mean, z_mean)
