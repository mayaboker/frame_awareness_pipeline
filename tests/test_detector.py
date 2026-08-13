from types import SimpleNamespace

import pytest

from frame_awareness.detector import _class_map, group_nms
from frame_awareness.types import Detection


def test_class_mapping_pools_motorcycle_as_vehicle() -> None:
    config = {"person": [0], "animal": [14], "vehicle": [2, 3, 5, 7]}
    assert _class_map(config)[3] == "vehicle"


def test_vehicle_group_nms_does_not_merge_people() -> None:
    values = [
        Detection((0, 0, 20, 20), 0.9, "person"),
        Detection((1, 1, 21, 21), 0.8, "person"),
        Detection((0, 0, 20, 20), 0.7, "vehicle"),
        Detection((1, 1, 21, 21), 0.6, "vehicle"),
    ]
    assert group_nms(values, 0.5, ("vehicle",)) == [values[0], values[1], values[2]]

