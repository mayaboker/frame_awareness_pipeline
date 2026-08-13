from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from ultralytics import YOLO

from .config import resolve_path
from .types import Detection


GROUPS = ("person", "animal", "vehicle")


class YoloDetector:
    """YOLO26s inference, awareness-class pooling, and vehicle-group NMS."""

    def __init__(self, config: Any, device: str | int) -> None:
        self.config = config
        self.device = device
        backend = str(config.backend).lower()
        self.model_path = resolve_path(config.model[backend])
        self.model = YOLO(str(self.model_path), task="detect")
        self.class_map = _class_map(config.classes)
        self.class_ids = sorted(self.class_map)

    def warmup(self) -> None:
        iterations = int(self.config.warmup_iterations)
        if iterations <= 0:
            return
        image = np.zeros(
            (int(self.config.image_size), int(self.config.image_size), 3), dtype=np.uint8
        )
        for _ in range(iterations):
            self.detect(image)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            source=frame,
            classes=self.class_ids,
            imgsz=int(self.config.image_size),
            conf=float(self.config.confidence.collection),
            iou=float(self.config.nms.yolo_iou_threshold),
            max_det=int(self.config.nms.maximum_detections),
            device=self.device,
            verbose=False,
        )[0]
        rows = result.boxes.data.detach().cpu().numpy() if result.boxes is not None else []
        detections = [
            Detection(tuple(map(float, row[:4])), float(row[4]), self.class_map[int(row[5])])
            for row in rows
            if int(row[5]) in self.class_map
        ]
        grouped_nms = self.config.nms.vehicle_group
        if bool(grouped_nms.enabled):
            detections = group_nms(detections, float(grouped_nms.iou_threshold), ("vehicle",))
        return detections


def _class_map(classes: Any) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for group in GROUPS:
        for class_id in classes[group]:
            class_id = int(class_id)
            if class_id in mapping:
                raise ValueError(f"COCO class {class_id} appears in multiple awareness groups")
            mapping[class_id] = group
    return mapping


def group_nms(
    detections: Iterable[Detection], iou_threshold: float, enabled_groups: Iterable[str]
) -> list[Detection]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("NMS IoU threshold must be between 0 and 1")
    enabled = set(enabled_groups)
    unknown = enabled.difference(GROUPS)
    if unknown:
        raise ValueError(f"Unknown awareness groups: {sorted(unknown)}")
    grouped: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.group].append(detection)
    output = []
    for group in GROUPS:
        candidates = sorted(grouped[group], key=lambda item: item.score, reverse=True)
        if group not in enabled:
            output.extend(candidates)
            continue
        while candidates:
            best = candidates.pop(0)
            output.append(best)
            candidates = [
                candidate
                for candidate in candidates
                if box_iou(best.xyxy, candidate.xyxy) <= iou_threshold
            ]
    return output


def box_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0
