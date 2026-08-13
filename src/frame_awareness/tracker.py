from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.oc_sort import OCSORT

from .detector import GROUPS
from .types import Detection, Track


@dataclass
class _BridgeState:
    xyxy: np.ndarray
    velocity: np.ndarray
    score: float
    last_frame: int


class OCSortTracker:
    """One OC-SORT instance per awareness group with globally unique output IDs."""

    def __init__(self, config: Any, update_fps: float) -> None:
        self.config = config
        self.update_fps = float(update_fps)
        self.maximum_age_frames = max(1, round(float(config.maximum_age_seconds) * update_fps))
        self.bridge_frames = int(config.emitted_prediction_frames)
        self._make_trackers()

    def _make_trackers(self) -> None:
        confidence = self.config.confidence
        association = self.config.association
        args = SimpleNamespace(
            tracker_type="ocsort",
            track_high_thresh=float(confidence.high),
            track_low_thresh=float(confidence.low),
            new_track_thresh=float(confidence.new_track),
            track_buffer=self.maximum_age_frames,
            match_thresh=float(association.match_threshold),
            fuse_score=bool(association.fuse_detection_score),
            delta_t=int(association.delta_t),
            inertia=float(association.inertia),
            use_byte=bool(association.use_byte_association),
        )
        self.trackers = {group: OCSORT(args) for group in GROUPS}
        self.states: dict[tuple[str, int], _BridgeState] = {}
        self.id_map: dict[tuple[str, int], int] = {}
        self.next_id = 1

    def update(
        self, frame_index: int, frame_shape: tuple[int, ...], detections: list[Detection]
    ) -> list[Track]:
        observations = []
        seen = set()
        for group_index, group in enumerate(GROUPS):
            rows = [
                [*item.xyxy, item.score, group_index]
                for item in detections
                if item.group == group
            ]
            boxes = Boxes(
                np.asarray(rows, dtype=np.float32).reshape(-1, 6),
                (int(frame_shape[0]), int(frame_shape[1])),
            )
            for row in self.trackers[group].update(boxes):
                local_id = int(row[4])
                key = (group, local_id)
                seen.add(key)
                if key not in self.id_map:
                    self.id_map[key] = self.next_id
                    self.next_id += 1
                current = np.asarray(row[:4], dtype=float)
                previous = self.states.get(key)
                elapsed = max(1, frame_index - previous.last_frame) if previous else 1
                velocity = (current - previous.xyxy) / elapsed if previous else np.zeros(4)
                score = float(row[5])
                self.states[key] = _BridgeState(current, velocity, score, frame_index)
                observations.append(
                    Track(self.id_map[key], group, tuple(map(float, current)), score, True)
                )
        observations.extend(self._bridge(frame_index, seen))
        return observations

    def _bridge(self, frame_index: int, seen: set[tuple[str, int]]) -> list[Track]:
        output = []
        expired = []
        for key, state in self.states.items():
            if key in seen:
                continue
            age = frame_index - state.last_frame
            if age > self.bridge_frames:
                expired.append(key)
                continue
            predicted = state.xyxy + state.velocity * age
            output.append(
                Track(
                    self.id_map[key],
                    key[0],
                    tuple(map(float, predicted)),
                    state.score,
                    False,
                )
            )
        for key in expired:
            del self.states[key]
        return output

    def reset(self) -> None:
        self._make_trackers()
