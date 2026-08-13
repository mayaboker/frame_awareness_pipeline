from __future__ import annotations

import time
from typing import Any

import numpy as np

from .awareness import TemporalAwareness, VehicleMotionClassifier, apply_motion
from .detector import YoloDetector
from .tracker import OCSortTracker
from .types import AwarenessResult, Latency


class FrameAwarenessPipeline:
    """Reusable single-frame engine. It deliberately does not own a camera."""

    def __init__(self, config: Any, processing_fps: float | None = None) -> None:
        self.config = config
        self.processing_fps = float(processing_fps or config.source.target_fps)
        self.detector = YoloDetector(config.detector, _device(config.runtime.device))
        self.tracker = OCSortTracker(config.tracker, self.processing_fps)
        self.motion = VehicleMotionClassifier(config.motion, self.processing_fps)
        thresholds = {
            group: float(config.detector.confidence[group])
            for group in ("person", "animal", "vehicle")
        }
        self.awareness = TemporalAwareness(config.awareness, thresholds)
        self.frame_index = 0

    def warmup(self) -> None:
        self.detector.warmup()

    def process(self, frame: np.ndarray, timestamp_seconds: float) -> AwarenessResult:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a non-empty HxWx3 BGR numpy array")
        if not frame.size:
            raise ValueError("frame must not be empty")
        total_started = time.perf_counter()
        detector_started = time.perf_counter()
        detections = self.detector.detect(frame)
        detector_ms = _elapsed_ms(detector_started)

        tracker_started = time.perf_counter()
        tracks = self.tracker.update(self.frame_index, frame.shape, detections)
        tracks = apply_motion(tracks, self.motion, self.frame_index, frame.shape)
        tracker_ms = _elapsed_ms(tracker_started)

        awareness_started = time.perf_counter()
        provisional = Latency(detector_ms, tracker_ms, 0.0, 0.0)
        result = self.awareness.decide(
            self.frame_index, timestamp_seconds, tracks, detections, provisional
        )
        awareness_ms = _elapsed_ms(awareness_started)
        total_ms = _elapsed_ms(total_started)
        result = AwarenessResult(
            **{
                **result.__dict__,
                "latency": Latency(detector_ms, tracker_ms, awareness_ms, total_ms),
            }
        )
        self.frame_index += 1
        return result

    def reset(self, preserve_frame_index: bool = False) -> None:
        self.tracker.reset()
        self.motion.reset()
        self.awareness.reset()
        if not preserve_frame_index:
            self.frame_index = 0

    def close(self) -> None:
        """Release pipeline-owned resources. Current backends require no explicit close."""


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _device(value: Any) -> str | int:
    text = str(value)
    return int(text) if text.isdigit() else text
