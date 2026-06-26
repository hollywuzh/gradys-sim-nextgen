"""Small GrADyS protocols used by the RL bridge showcase."""

from __future__ import annotations

from typing import Optional, Tuple

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.mobility import SetVelocityMobilityCommand
from gradysim.protocol.messages.telemetry import Telemetry

Position3D = Tuple[float, float, float]
Velocity3D = Tuple[float, float, float]


class RLControlledUAVProtocol(IProtocol):
    """Protocol whose mobility commands are issued by an external RL wrapper."""

    def __init__(self) -> None:
        self.node_id: Optional[int] = None
        self.current_position: Optional[Position3D] = None
        self.current_velocity: Velocity3D = (0.0, 0.0, 0.0)
        self.commanded_velocity: Velocity3D = (0.0, 0.0, 0.0)

    def initialize(self) -> None:
        self.node_id = self.provider.get_id()
        node = getattr(self.provider, "node", None)
        position = getattr(node, "position", None)
        if position is not None:
            self.current_position = tuple(position)

    def set_velocity(self, velocity: Velocity3D) -> None:
        self.commanded_velocity = velocity
        self.provider.send_mobility_command(SetVelocityMobilityCommand(*velocity))

    def handle_timer(self, timer: str) -> None:
        pass

    def handle_packet(self, message: str) -> None:
        pass

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self.current_position = tuple(telemetry.current_position)
        self.current_velocity = tuple(getattr(telemetry, "current_velocity", (0.0, 0.0, 0.0)))

    def finish(self) -> None:
        self.set_velocity((0.0, 0.0, 0.0))


class PassiveEdgeDeviceProtocol(IProtocol):
    """Passive node used only to show edge-device positions in GrADyS."""

    def initialize(self) -> None:
        pass

    def handle_timer(self, timer: str) -> None:
        pass

    def handle_packet(self, message: str) -> None:
        pass

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def finish(self) -> None:
        pass
