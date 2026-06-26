import random

from security.attacks.ack_poisoning import AckPoisoningAttack
from security.attack_context import AttackContext
from security.attacks.ack_spoofing import AckSpoofingAttack
from security.attacks.blackhole import BlackholeAttack
from security.attacks.grayhole import GrayholeAttack
from security.attacks.location_spoofing import LocationSpoofingAttack
from security.attacks.phy_jamming import PhyJammingAttack
from utils import config


class AttackManager:
    ATTACK_REGISTRY = {
        "ACK_POISON": AckPoisoningAttack,
        "ACK_POISONING": AckPoisoningAttack,
        "BLACKHOLE": BlackholeAttack,
        "GRAYHOLE": GrayholeAttack,
        "ACK_SPOOF": AckSpoofingAttack,
        "ACK_SPOOFING": AckSpoofingAttack,
        "LOCATION_SPOOF": LocationSpoofingAttack,
        "LOC_SPOOF": LocationSpoofingAttack,
        "PHY_JAMMING": PhyJammingAttack,
        "PHY_JAMMER": PhyJammingAttack,
    }

    def __init__(self, simulator):
        self.simulator = simulator
        self.context = AttackContext(
            attack_names=tuple(config.ATTACK_NAMES),
            attacker_ids=frozenset(config.ATTACKER_IDS),
            start_time_us=config.ATTACK_START_US,
            end_time_us=config.ATTACK_END_US,
            probability=config.ATTACK_PROBABILITY,
        )
        self.rng = random.Random(self.simulator.seed + 7000)
        self.attacks = []
        self._load_attacks()

    @property
    def enabled(self):
        return bool(self.attacks)

    def _load_attacks(self):
        loaded_names = set()
        for attack_name in self.context.attack_names:
            attack_cls = self.ATTACK_REGISTRY.get(attack_name)
            if attack_cls is None:
                continue

            canonical_name = attack_cls.name
            if canonical_name in loaded_names:
                continue

            self.attacks.append(attack_cls(self))
            loaded_names.add(canonical_name)

    def initialize(self):
        for attack in self.attacks:
            attack.initialize()

    def is_attack_window_active(self):
        return self.context.start_time_us <= self.simulator.env.now <= self.context.end_time_us

    def is_attacker(self, drone_id):
        return drone_id in self.context.attacker_ids

    def has_attackers(self):
        return bool(self.context.attacker_ids)

    def sample_attack(self):
        return self.rng.random() <= self.context.probability

    def sample_attack_with_probability(self, probability):
        return self.rng.random() <= probability

    def get_attacker_ids_for(self, attack_name):
        attack_name = attack_name.upper()
        if attack_name == "BLACKHOLE" and config.BLACKHOLE_ATTACKER_IDS:
            return frozenset(config.BLACKHOLE_ATTACKER_IDS)
        if attack_name == "GRAYHOLE" and config.GRAYHOLE_ATTACKER_IDS:
            return frozenset(config.GRAYHOLE_ATTACKER_IDS)

        return self.context.attacker_ids

    def is_attacker_for(self, attack_name, drone_id):
        return drone_id in self.get_attacker_ids_for(attack_name)

    def get_probability_for(self, attack_name):
        attack_name = attack_name.upper()
        if attack_name == "BLACKHOLE":
            return config.BLACKHOLE_DROP_PROBABILITY
        if attack_name == "GRAYHOLE":
            return config.GRAYHOLE_DROP_PROBABILITY

        return self.context.probability

    def record_attack_event(self, attack_name, drone_id, packet_id):
        if not hasattr(self.simulator, "security_attack_events"):
            self.simulator.security_attack_events = []

        self.simulator.security_attack_events.append(
            {
                "time_us": self.simulator.env.now,
                "attack_name": attack_name,
                "drone_id": drone_id,
                "packet_id": packet_id,
            }
        )

    def record_security_event(self, event_name, increment=1):
        self.simulator.metrics.record_security_event(event_name, increment)

    def should_drop_forwarded_packet(self, drone, packet):
        if not self.enabled:
            return False

        for attack in self.attacks:
            if attack.should_drop_forwarded_packet(drone, packet):
                return True

        return False

    def should_drop_received_packet(self, drone, packet, src_drone_id):
        if not self.enabled:
            return False

        for attack in self.attacks:
            if attack.should_drop_received_packet(drone, packet, src_drone_id):
                return True

        return False

    def on_wait_ack_started(self, drone, packet, wait_ack_key):
        if not self.enabled:
            return

        for attack in self.attacks:
            attack.on_wait_ack_started(drone, packet, wait_ack_key)

    def mutate_outgoing_packet(self, drone, packet, transmission_type, next_hop_id):
        if not self.enabled:
            return packet

        mutated_packet = packet
        for attack in self.attacks:
            mutated_packet = attack.mutate_outgoing_packet(
                drone,
                mutated_packet,
                transmission_type,
                next_hop_id,
            )

        return mutated_packet

    def get_external_interferers(self, receiver, channel_id):
        if not self.enabled:
            return []

        interferers = []
        for attack in self.attacks:
            interferers.extend(attack.get_external_interferers(receiver, channel_id))

        return interferers
