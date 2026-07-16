from dataclasses import asdict, dataclass


@dataclass
class EpisodeMetrics:
    reward: float
    safety_cost: float
    hazard_recall: float
    coverage: float
    success: bool
    steps: int

    def to_dict(self) -> dict[str, float | bool | int]:
        return asdict(self)