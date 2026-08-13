from pathlib import Path

import pytest
from omegaconf import OmegaConf

from frame_awareness.config import ConfigurationError, validate_config


def config():
    result = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/config.yaml")
    result.detector.model.pytorch = __file__
    result.detector.model.onnx = __file__
    result.runtime.device = "cpu"
    return result


def test_valid_configuration() -> None:
    validate_config(config())


def test_image_size_must_match_stride() -> None:
    value = config()
    value.detector.image_size = 1200
    with pytest.raises(ConfigurationError, match="divisible by 32"):
        validate_config(value)


def test_temporal_requirement_cannot_exceed_window() -> None:
    value = config()
    value.awareness.person.presence_detection_frames = 16
    with pytest.raises(ConfigurationError, match="Temporal evidence"):
        validate_config(value)


def test_confidence_ordering_is_validated() -> None:
    value = config()
    value.tracker.confidence.low = 0.2
    with pytest.raises(ConfigurationError, match="low confidence"):
        validate_config(value)


def test_motion_hysteresis_is_validated() -> None:
    value = config()
    value.motion.translation.stationary_box_diagonals_per_second = 0.3
    with pytest.raises(ConfigurationError, match="Stationary translation"):
        validate_config(value)
