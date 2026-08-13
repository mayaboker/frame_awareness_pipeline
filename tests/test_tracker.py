from types import SimpleNamespace

from frame_awareness.tracker import OCSortTracker


def test_maximum_age_seconds_is_converted_to_frames() -> None:
    config = SimpleNamespace(
        confidence=SimpleNamespace(high=0.1, low=0.02, new_track=0.25),
        association=SimpleNamespace(
            match_threshold=0.7,
            inertia=0.1,
            delta_t=3,
            use_byte_association=True,
            fuse_detection_score=False,
        ),
        maximum_age_seconds=0.5,
        emitted_prediction_frames=1,
    )
    assert OCSortTracker(config, 30).maximum_age_frames == 15
    assert OCSortTracker(config, 15).maximum_age_frames == 8

