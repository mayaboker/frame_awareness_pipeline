from __future__ import annotations

import json
import logging
import signal
import statistics
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .config import validate_config
from .pipeline import FrameAwarenessPipeline
from .types import AwarenessResult, MotionState


LOGGER = logging.getLogger(__name__)


class LatestFrameSource:
    """Bounded live reader: new frames replace old frames instead of building latency."""

    def __init__(self, uri: str | int, config: Any) -> None:
        self.uri = uri
        self.config = config
        self.condition = threading.Condition()
        self.latest: tuple[int, float, np.ndarray] | None = None
        self.sequence = 0
        self.last_delivered = -1
        self.stopped = False
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name="camera-reader")

    def start(self) -> None:
        self.thread.start()

    def read(self, timeout: float) -> tuple[float, np.ndarray] | None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.stopped and (
                self.latest is None or self.latest[0] == self.last_delivered
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            if self.error:
                raise self.error
            if self.latest is None:
                return None
            self.last_delivered = self.latest[0]
            return self.latest[1], self.latest[2].copy()

    def _capture_loop(self) -> None:
        attempts = 0
        while not self.stopped:
            capture = cv2.VideoCapture(self.uri)
            if not capture.isOpened():
                attempts += 1
                maximum = int(self.config.maximum_reconnect_attempts)
                if maximum and attempts >= maximum:
                    self._fail(RuntimeError(f"Could not open live source: {self.uri}"))
                    return
                time.sleep(float(self.config.reconnect_delay_seconds))
                continue
            attempts = 0
            failures = 0
            while not self.stopped:
                ok, frame = capture.read()
                if not ok:
                    failures += 1
                    if failures >= int(self.config.maximum_consecutive_read_failures):
                        break
                    continue
                failures = 0
                with self.condition:
                    self.sequence += 1
                    self.latest = (self.sequence, time.monotonic(), frame)
                    self.condition.notify_all()
            capture.release()
            if not self.stopped:
                time.sleep(float(self.config.reconnect_delay_seconds))
        with self.condition:
            self.condition.notify_all()

    def _fail(self, error: Exception) -> None:
        with self.condition:
            self.error = error
            self.stopped = True
            self.condition.notify_all()

    def close(self) -> None:
        self.stopped = True
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2.0)


class ApplicationRunner:
    def __init__(self, config: DictConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.stop_event = threading.Event()
        self.latencies: dict[str, deque[float]] = {
            name: deque(maxlen=100_000)
            for name in ("detector_ms", "tracker_ms", "awareness_ms", "total_ms")
        }

    def run(self) -> None:
        source_kind = str(self.config.source.kind)
        uri = _source_uri(self.config.source.uri)
        if source_kind == "live":
            self._run_live(uri)
        elif source_kind == "file":
            self._run_file(uri)
        else:
            raise ValueError("source.kind must be 'live' or 'file'")

    def _run_file(self, uri: str | int) -> None:
        capture = cv2.VideoCapture(uri)
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video source: {uri}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(self.config.source.target_fps)
        fps = min(source_fps, float(self.config.source.target_fps))
        pipeline = FrameAwarenessPipeline(self.config, fps)
        pipeline.warmup()
        next_sample_time = 0.0
        source_index = -1
        processed = 0
        resources = _OutputResources(self.config.output, self.output_dir, fps)
        try:
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                source_index += 1
                timestamp = source_index / source_fps
                if timestamp + 1e-9 < next_sample_time:
                    continue
                next_sample_time += 1 / fps
                result = pipeline.process(frame, timestamp)
                resources.emit(frame, result)
                self._record_latency(result)
                processed += 1
                maximum = self.config.runtime.maximum_frames
                if maximum is not None and processed >= int(maximum):
                    break
        finally:
            capture.release()
            resources.close()
            pipeline.close()
            self._write_summary(processed, fps)

    def _run_live(self, uri: str | int) -> None:
        fps = float(self.config.source.target_fps)
        pipeline = FrameAwarenessPipeline(self.config, fps)
        pipeline.warmup()
        source = LatestFrameSource(uri, self.config.source.live)
        resources = _OutputResources(self.config.output, self.output_dir, fps)
        processed = 0
        previous_timestamp: float | None = None
        next_process_time = time.monotonic()
        source.start()
        try:
            while not self.stop_event.is_set():
                wait = next_process_time - time.monotonic()
                if wait > 0 and self.stop_event.wait(wait):
                    break
                item = source.read(float(self.config.source.live.stall_timeout_seconds))
                if item is None:
                    LOGGER.warning("Live source stalled; waiting for reconnect")
                    continue
                timestamp, frame = item
                if (
                    previous_timestamp is not None
                    and timestamp - previous_timestamp
                    > float(self.config.source.live.reset_pipeline_after_seconds)
                ):
                    LOGGER.warning("Long stream interruption; resetting tracking state")
                    pipeline.reset(preserve_frame_index=True)
                previous_timestamp = timestamp
                result = pipeline.process(frame, timestamp)
                resources.emit(frame, result)
                self._record_latency(result)
                processed += 1
                next_process_time = max(next_process_time + 1 / fps, time.monotonic())
                maximum = self.config.runtime.maximum_frames
                if maximum is not None and processed >= int(maximum):
                    break
        finally:
            source.close()
            resources.close()
            pipeline.close()
            self._write_summary(processed, fps)

    def stop(self, *_: Any) -> None:
        self.stop_event.set()

    def _record_latency(self, result: AwarenessResult) -> None:
        for name, values in self.latencies.items():
            values.append(float(getattr(result.latency, name)))

    def _write_summary(self, frames: int, fps: float) -> None:
        summary = {
            "processed_frames": frames,
            "target_fps": fps,
            "effective_processing_fps": (
                1000 / statistics.fmean(self.latencies["total_ms"])
                if self.latencies["total_ms"]
                else None
            ),
            "latency_ms": {
                name: _percentiles(list(values)) for name, values in self.latencies.items()
            },
            "resolved_config": OmegaConf.to_container(self.config, resolve=True),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


class _OutputResources:
    def __init__(self, config: Any, output_dir: Path, fps: float) -> None:
        self.config = config
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = (
            (output_dir / str(config.jsonl.path)).open("w") if bool(config.jsonl.enabled) else None
        )
        self.writer: cv2.VideoWriter | None = None
        self.fps = fps

    def emit(self, frame: np.ndarray, result: AwarenessResult) -> None:
        if bool(self.config.console):
            LOGGER.info(
                "frame=%d person=%s animal=%s moving_vehicle=%s latency=%.1fms",
                result.frame_index,
                result.person_present,
                result.animal_present,
                result.moving_vehicle_present,
                result.latency.total_ms,
            )
        if self.jsonl:
            self.jsonl.write(json.dumps(result.to_dict(bool(self.config.include_tracks))) + "\n")
            self.jsonl.flush()
        if bool(self.config.annotated_video.enabled):
            rendered = _annotate(frame, result)
            maximum = int(self.config.annotated_video.maximum_size)
            if maximum > 0 and max(rendered.shape[:2]) > maximum:
                scale = maximum / max(rendered.shape[:2])
                rendered = cv2.resize(
                    rendered,
                    (round(rendered.shape[1] * scale), round(rendered.shape[0] * scale)),
                )
            if self.writer is None:
                height, width = rendered.shape[:2]
                self.writer = cv2.VideoWriter(
                    str(self.output_dir / str(self.config.annotated_video.path)),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (width, height),
                )
                if not self.writer.isOpened():
                    raise RuntimeError("Could not initialize annotated-video writer")
            self.writer.write(rendered)

    def close(self) -> None:
        if self.jsonl:
            self.jsonl.close()
        if self.writer:
            self.writer.release()


def _annotate(frame: np.ndarray, result: AwarenessResult) -> np.ndarray:
    output = frame.copy()
    colors = {"person": (0, 220, 0), "animal": (255, 170, 0), "vehicle": (0, 180, 255)}
    for track in result.tracks:
        x1, y1, x2, y2 = map(int, track.xyxy)
        color = colors[track.group]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        motion = f" {track.motion_state.value}" if track.group == "vehicle" else ""
        cv2.putText(
            output,
            f"{track.group} #{track.track_id}{motion} {track.score:.2f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def _source_uri(value: Any) -> str | int:
    text = str(value)
    return int(text) if text.isdigit() else text


def _percentiles(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": float(np.percentile(values, 50)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "p99": float(np.percentile(values, 99)) if values else None,
    }


CONFIG_DIRECTORY = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(version_base="1.3", config_path=CONFIG_DIRECTORY, config_name="config")
def main(config: DictConfig) -> None:
    validate_config(config)
    logging.getLogger().setLevel(str(config.runtime.log_level).upper())
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    runner = ApplicationRunner(config, output_dir)
    signal.signal(signal.SIGINT, runner.stop)
    signal.signal(signal.SIGTERM, runner.stop)
    runner.run()


def entrypoint() -> None:
    main()
