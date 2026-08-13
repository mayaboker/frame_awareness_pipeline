from types import SimpleNamespace

from frame_awareness.awareness import TemporalAwareness, VehicleMotionClassifier
from frame_awareness.types import Detection, Latency, MotionState, Track


def temporal_config():
    return SimpleNamespace(
        window_frames=15,
        person=SimpleNamespace(
            presence_detection_frames=8,
            counting_track_frames=8,
            counting_detector_confirmations=4,
        ),
        animal=SimpleNamespace(
            presence_detection_frames=8,
            counting_track_frames=8,
            counting_detector_confirmations=4,
        ),
        moving_vehicle=SimpleNamespace(track_frames=6, detector_confirmations=3),
    )


def motion_config():
    return SimpleNamespace(
        history=SimpleNamespace(
            maximum_observations=10, minimum_observations=5, moving_grace_frames=5
        ),
        translation=SimpleNamespace(
            moving_box_diagonals_per_second=0.18,
            stationary_box_diagonals_per_second=0.072,
            minimum_pixels_per_second=8.0,
            minimum_image_diagonals_per_second=0.0015,
        ),
        scale=SimpleNamespace(
            moving_log_height_per_second=0.15,
            stationary_log_height_per_second=0.06,
            minimum_direction_consistency=0.7,
        ),
    )


def test_person_presence_survives_identity_changes() -> None:
    engine = TemporalAwareness(temporal_config(), {"person": 0.25, "animal": 0.1, "vehicle": 0.25})
    latency = Latency(0, 0, 0, 0)
    for frame in range(8):
        track = Track(frame + 1, "person", (0, 0, 20, 20), 0.9, True)
        result = engine.decide(
            frame,
            frame / 30,
            [track],
            [Detection(track.xyxy, 0.9, "person")],
            latency,
        )
    assert result.person_present
    assert result.person_count == 0


def test_motion_is_fps_normalized() -> None:
    classifier = VehicleMotionClassifier(motion_config(), 30)
    for frame in range(6):
        state = classifier.classify(
            1, frame, (frame * 2, 0, 20 + frame * 2, 20), True, (720, 1280, 3)
        )
    assert state == MotionState.MOVING

