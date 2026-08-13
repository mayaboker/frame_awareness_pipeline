from __future__ import annotations

import sysconfig
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTALLED_ROOT = Path(sysconfig.get_path("data")) / "share" / "frame_awareness"


class ConfigurationError(ValueError):
    """Raised when runtime configuration is internally inconsistent."""


def load_config(path: str | Path = "configs/config.yaml", overrides: list[str] | None = None) -> DictConfig:
    path = resolve_config_path(path)
    with initialize_config_dir(version_base="1.3", config_dir=str(path.parent)):
        config = compose(config_name=path.stem, overrides=overrides or [])
    validate_config(config)
    return config


def validate_config(config: Any) -> None:
    image_size = int(config.detector.image_size)
    if image_size <= 0 or image_size % 32:
        raise ConfigurationError("detector.image_size must be positive and divisible by 32")
    backend = str(config.detector.backend).lower()
    if backend not in {"pytorch", "onnx"}:
        raise ConfigurationError("detector.backend must be 'pytorch' or 'onnx'")
    model_path = resolve_path(config.detector.model[backend])
    if not model_path.is_file():
        raise ConfigurationError(f"Detector model is missing: {model_path}")

    values = {
        "collection": config.detector.confidence.collection,
        "person": config.detector.confidence.person,
        "animal": config.detector.confidence.animal,
        "vehicle": config.detector.confidence.vehicle,
        "low": config.tracker.confidence.low,
        "high": config.tracker.confidence.high,
        "new_track": config.tracker.confidence.new_track,
    }
    for name, value in values.items():
        if not 0 <= float(value) <= 1:
            raise ConfigurationError(f"Confidence {name} must be between 0 and 1")
    if float(values["collection"]) > float(values["low"]):
        raise ConfigurationError("Detector collection confidence must not exceed tracker low confidence")
    if float(values["low"]) > float(values["high"]):
        raise ConfigurationError("Tracker low confidence must not exceed high confidence")
    if float(values["new_track"]) < float(values["high"]):
        raise ConfigurationError("New-track confidence must be at least tracker high confidence")

    motion = config.motion
    if float(motion.translation.stationary_box_diagonals_per_second) >= float(
        motion.translation.moving_box_diagonals_per_second
    ):
        raise ConfigurationError("Stationary translation threshold must be below moving threshold")
    if float(motion.scale.stationary_log_height_per_second) >= float(
        motion.scale.moving_log_height_per_second
    ):
        raise ConfigurationError("Stationary scale threshold must be below moving threshold")

    window = int(config.awareness.window_frames)
    requirements = (
        config.awareness.person.presence_detection_frames,
        config.awareness.person.counting_track_frames,
        config.awareness.animal.presence_detection_frames,
        config.awareness.animal.counting_track_frames,
        config.awareness.moving_vehicle.track_frames,
        config.awareness.stationary_vehicle.track_frames,
    )
    if window <= 0 or any(int(value) <= 0 or int(value) > window for value in requirements):
        raise ConfigurationError("Temporal evidence requirements must be in [1, window_frames]")
    if float(config.tracker.maximum_age_seconds) <= 0:
        raise ConfigurationError("tracker.maximum_age_seconds must be positive")
    resolve_device(config.runtime.device, backend)


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    checkout_path = PROJECT_ROOT / path
    if checkout_path.exists():
        return checkout_path
    if path.parts[:2] == ("src", "models"):
        return _installed_root() / "models" / Path(*path.parts[2:])
    return checkout_path


def resolve_config_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    checkout_path = PROJECT_ROOT / path
    if checkout_path.exists():
        return checkout_path
    if path.parts and path.parts[0] == "configs":
        return _installed_root() / Path(*path.parts)
    return checkout_path


def _installed_root() -> Path:
    relative = Path("share/frame_awareness")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative
        if candidate.is_dir():
            return candidate
    return INSTALLED_ROOT


def resolve_device(device: Any, backend: str) -> str | int:
    text = str(device).lower()
    if text == "auto":
        if backend == "pytorch":
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ConfigurationError(
                "ONNX backend requires the optional dependency: pip install '.[onnx]'"
            ) from error
        return 0 if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    if text == "cpu":
        return "cpu"
    if backend == "pytorch":
        import torch

        if not torch.cuda.is_available():
            raise ConfigurationError(
                "A CUDA device was requested but PyTorch cannot access CUDA; "
                "use runtime.device=cpu or install a compatible CUDA build"
            )
        try:
            index = int(text)
        except ValueError as error:
            raise ConfigurationError("PyTorch runtime.device must be a CUDA index or 'cpu'") from error
        if index < 0 or index >= torch.cuda.device_count():
            raise ConfigurationError(
                f"CUDA device {index} is unavailable; detected {torch.cuda.device_count()} device(s)"
            )
        return index
    elif backend == "onnx":
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ConfigurationError(
                "ONNX backend requires the optional dependency: pip install -e '.[onnx]'"
            ) from error
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise ConfigurationError(
                "ONNX CUDA was requested but CUDAExecutionProvider is unavailable; "
                "use runtime.device=cpu or install compatible onnxruntime-gpu"
            )
        return int(text)
    raise ConfigurationError("runtime.device must be 'auto', 'cpu', or a CUDA index")
