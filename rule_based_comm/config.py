"""Tunable assumptions for the initial, pre-robot cost model."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MessageSizes:
    E: int = 120
    N: int = 40
    C: int = 40
    I: int = 30_720

    def size_of(self, item: str) -> int:
        try:
            return getattr(self, item)
        except AttributeError as exc:
            raise ValueError(f"Unknown message item: {item}") from exc


@dataclass(frozen=True)
class CostWeights:
    transmission_per_kib: float = 0.1
    request_roundtrip: float = 1.0
    wrong_action: float = 12.0
    reobservation: float = 2.0
    unresolved: float = 10.0


@dataclass(frozen=True)
class OutcomeParameters:
    """Probabilities to replace with measured/calibrated values later."""

    crop_match_success: float = 0.95
    # Scenario ④ splits the remaining 5% into about 2.3% wrong and
    # 2.7% unresolved: 2.3 / 5.0 = 0.46.
    crop_wrong_fraction_on_failure: float = 0.46
    appearance_match_success: float = 0.90
    wrong_fraction_on_match_failure: float = 0.30
    unresolved_after_reobservation: float = 0.10
    default_position_success: float = 0.50


@dataclass(frozen=True)
class PolicyConfig:
    message_sizes: MessageSizes = field(default_factory=MessageSizes)
    costs: CostWeights = field(default_factory=CostWeights)
    outcome: OutcomeParameters = field(default_factory=OutcomeParameters)

