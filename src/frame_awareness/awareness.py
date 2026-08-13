from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from .types import AwarenessResult, Detection, Latency, MotionState, Track


class VehicleMotionClassifier:
    def __init__(self, config: Any, fps: float) -> None:
        self.fps = float(fps)
        self.config = config
        self.history: dict[int, deque[tuple[int, np.ndarray]]] = defaultdict(
            lambda: deque(maxlen=int(config.history.maximum_observations))
        )
        self.states: dict[int, MotionState] = {}
        self.last_moving: dict[int, int] = {}

    def classify(
        self,
        track_id: int,
        frame_index: int,
        xyxy: tuple[float, ...],
        confirmed: bool,
        frame_shape: tuple[int, ...],
    ) -> MotionState:
        if confirmed:
            self.history[track_id].append((frame_index, np.asarray(xyxy, dtype=float)))
            self.states[track_id] = self._calculate(track_id, frame_shape)
            if self.states[track_id] == MotionState.MOVING:
                self.last_moving[track_id] = frame_index
        state = self.states.get(track_id, MotionState.UNCERTAIN)
        last = self.last_moving.get(track_id)
        if (
            state != MotionState.MOVING
            and last is not None
            and frame_index - last <= int(self.config.history.moving_grace_frames)
        ):
            return MotionState.MOVING
        return state

    def _calculate(self, track_id: int, frame_shape: tuple[int, ...]) -> MotionState:
        history = self.history[track_id]
        if len(history) < int(self.config.history.minimum_observations):
            return MotionState.UNCERTAIN
        boxes = np.asarray([item[1] for item in history])
        times = np.asarray([item[0] / self.fps for item in history])
        widths = np.maximum(1.0, boxes[:, 2] - boxes[:, 0])
        heights = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
        centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))
        vx = _median_slope(times, centers[:, 0])
        vy = _median_slope(times, centers[:, 1])
        pixels_per_second = math.hypot(vx, vy)
        box_diagonals_per_second = pixels_per_second / float(np.median(np.hypot(widths, heights)))
        image_diagonals_per_second = pixels_per_second / math.hypot(frame_shape[1], frame_shape[0])
        log_heights = np.log(heights)
        scale_velocity = _median_slope(times, log_heights)
        differences = np.diff(log_heights)
        nonzero = differences[np.abs(differences) > 1e-4]
        consistency = (
            float(np.mean(np.sign(nonzero) == np.sign(scale_velocity)))
            if len(nonzero) and scale_velocity != 0
            else 0.0
        )
        translation = self.config.translation
        scale = self.config.scale
        translation_moving = (
            box_diagonals_per_second >= float(translation.moving_box_diagonals_per_second)
            and pixels_per_second >= float(translation.minimum_pixels_per_second)
            and image_diagonals_per_second >= float(translation.minimum_image_diagonals_per_second)
        )
        scale_moving = (
            abs(scale_velocity) >= float(scale.moving_log_height_per_second)
            and consistency >= float(scale.minimum_direction_consistency)
        )
        if translation_moving or scale_moving:
            return MotionState.MOVING
        if (
            box_diagonals_per_second <= float(translation.stationary_box_diagonals_per_second)
            and abs(scale_velocity) <= float(scale.stationary_log_height_per_second)
        ):
            return MotionState.STATIONARY
        return self.states.get(track_id, MotionState.UNCERTAIN)

    def reset(self) -> None:
        self.history.clear()
        self.states.clear()
        self.last_moving.clear()


class TemporalAwareness:
    def __init__(self, config: Any, thresholds: Mapping[str, float]) -> None:
        self.config = config
        self.thresholds = thresholds
        size = int(config.window_frames)
        self.track_frames: deque[tuple[int, tuple[Track, ...]]] = deque(maxlen=size)
        self.detection_frames: deque[tuple[int, tuple[Detection, ...]]] = deque(maxlen=size)

    def decide(
        self,
        frame_index: int,
        timestamp_seconds: float,
        tracks: Iterable[Track],
        detections: Iterable[Detection],
        latency: Latency,
    ) -> AwarenessResult:
        tracks, detections = tuple(tracks), tuple(detections)
        self.track_frames.append((frame_index, tracks))
        self.detection_frames.append((frame_index, detections))
        person_present = self._detection_presence("person", self.config.person)
        animal_present = self._detection_presence("animal", self.config.animal)
        person_count = self._count("person", self.config.person)
        animal_count = self._count("animal", self.config.animal)
        moving_vehicle_count = self._count(
            "vehicle", self.config.moving_vehicle, MotionState.MOVING
        )
        stationary_vehicle_count = self._count(
            "vehicle", self.config.stationary_vehicle, MotionState.STATIONARY
        )
        moving_vehicle_present = moving_vehicle_count > 0
        return AwarenessResult(
            "1.1",
            frame_index,
            timestamp_seconds,
            person_present,
            person_count,
            animal_present,
            animal_count,
            moving_vehicle_present,
            moving_vehicle_count,
            stationary_vehicle_count,
            person_present or animal_present or moving_vehicle_present,
            tracks,
            latency,
        )

    def _detection_presence(self, group: str, rule: Any) -> bool:
        required = int(rule.presence_detection_frames)
        threshold = float(self.thresholds[group])
        return sum(
            any(item.group == group and item.score >= threshold for item in detections)
            for _, detections in self.detection_frames
        ) >= required

    def _count(
        self,
        group: str,
        rule: Any,
        required_motion: MotionState | None = None,
    ) -> int:
        track_frames: dict[int, set[int]] = defaultdict(set)
        confirmations: dict[int, int] = defaultdict(int)
        threshold = float(self.thresholds[group])
        for frame_index, tracks in self.track_frames:
            for track in tracks:
                if track.group != group:
                    continue
                if required_motion is not None and track.motion_state != required_motion:
                    continue
                track_frames[track.track_id].add(frame_index)
                confirmations[track.track_id] += int(
                    track.detector_confirmed and track.score >= threshold
                )
        minimum_frames = int(_value(rule, "counting_track_frames", "track_frames"))
        required_confirmations = int(
            _value(rule, "counting_detector_confirmations", "detector_confirmations")
        )
        return sum(
            len(frames) >= minimum_frames and confirmations[track_id] >= required_confirmations
            for track_id, frames in track_frames.items()
        )

    def reset(self) -> None:
        self.track_frames.clear()
        self.detection_frames.clear()


def apply_motion(
    tracks: Iterable[Track],
    classifier: VehicleMotionClassifier,
    frame_index: int,
    frame_shape: tuple[int, ...],
) -> list[Track]:
    output = []
    for track in tracks:
        state = (
            classifier.classify(
                track.track_id,
                frame_index,
                track.xyxy,
                track.detector_confirmed,
                frame_shape,
            )
            if track.group == "vehicle"
            else MotionState.NOT_APPLICABLE
        )
        output.append(replace(track, motion_state=state))
    return output


def _median_slope(times: np.ndarray, values: np.ndarray) -> float:
    slopes = []
    for left in range(len(times) - 1):
        elapsed = times[left + 1 :] - times[left]
        valid = elapsed > 0
        slopes.extend(((values[left + 1 :][valid] - values[left]) / elapsed[valid]).tolist())
    return float(np.median(slopes)) if slopes else 0.0


def _value(rule: Any, preferred: str, fallback: str) -> Any:
    return getattr(rule, preferred) if hasattr(rule, preferred) else getattr(rule, fallback)
