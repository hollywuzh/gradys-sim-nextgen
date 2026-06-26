class BaseAttack:
    name = "BASE"

    def __init__(self, manager):
        self.manager = manager
        self.simulator = manager.simulator

    def initialize(self):
        return None

    def should_drop_forwarded_packet(self, drone, packet):
        return False

    def should_drop_received_packet(self, drone, packet, src_drone_id):
        return False

    def on_wait_ack_started(self, drone, packet, wait_ack_key):
        return None

    def mutate_outgoing_packet(self, drone, packet, transmission_type, next_hop_id):
        return packet

    def get_external_interferers(self, receiver, channel_id):
        return []

    def is_attack_window_active(self):
        return self.manager.is_attack_window_active()

    def sample_attack(self):
        return self.manager.sample_attack()
