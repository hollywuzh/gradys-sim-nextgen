from dataclasses import dataclass


@dataclass(frozen=True)
class AttackContext:
    attack_names: tuple[str, ...]
    attacker_ids: frozenset[int]
    start_time_us: int
    end_time_us: int
    probability: float
