from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class MotionState(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    UNCERTAIN = "uncertain"
    STATIONARY = "stationary"
    MOVING = "moving"


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    score: float
    group: str


@dataclass(frozen=True)
class Track:
    track_id: int
    group: str
    xyxy: tuple[float, float, float, float]
    score: float
    detector_confirmed: bool
    motion_state: MotionState = MotionState.NOT_APPLICABLE


@dataclass(frozen=True)
class Latency:
    detector_ms: float
    tracker_ms: float
    awareness_ms: float
    total_ms: float


@dataclass(frozen=True)
class AwarenessResult:
    schema_version: str
    frame_index: int
    timestamp_seconds: float
    person_present: bool
    person_count: int
    animal_present: bool
    animal_count: int
    moving_vehicle_present: bool
    moving_vehicle_count: int
    relevant_present: bool
    tracks: tuple[Track, ...]
    latency: Latency

    def to_dict(self, include_tracks: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if not include_tracks:
            result.pop("tracks")
        return result

