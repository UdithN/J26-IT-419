"""Tests for src/preprocess.py. Run from the project root with:  pytest"""

import numpy as np

from preprocess import (
    LEFT_HAND,
    POSE,
    augment_and_finalize_clip,
    finalize_clip,
    normalize_by_shoulders,
    preprocess_clip,
)

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

PREPROCESS_SETTINGS = {
    "num_frames": 64,
    "max_gap_frames": 5,
    "left_shoulder_index": LEFT_SHOULDER,
    "right_shoulder_index": RIGHT_SHOULDER,
}

AUGMENT_SETTINGS = {
    "time_factor_min": 0.8,
    "time_factor_max": 1.2,
    "rotation_degrees": 10,
    "scale_min": 0.9,
    "scale_max": 1.1,
    "noise_std": 0.01,
    "hand_dropout_probability": 1.0,
    "hand_dropout_max_frames": 3,
}


def make_fake_clip(num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """A clip where everything is detected, with random positions in [0.2, 0.8]."""
    rng = np.random.default_rng(0)
    keypoints = rng.uniform(0.2, 0.8, size=(num_frames, 130, 3)).astype(np.float32)
    mask = np.ones((num_frames, 3), dtype=bool)
    return keypoints, mask


def test_output_shape_without_augmentation():
    keypoints, mask = make_fake_clip(num_frames=100)
    prepared_keypoints, prepared_mask = preprocess_clip(keypoints, mask, 16 / 9, PREPROCESS_SETTINGS)
    features = finalize_clip(prepared_keypoints, prepared_mask, num_frames=64)
    assert features.shape == (64, 393)
    assert features.dtype == np.float32


def test_output_shape_with_augmentation_and_short_clip():
    # 20 frames is shorter than 64, so frames must be repeated, not dropped.
    keypoints, mask = make_fake_clip(num_frames=20)
    prepared_keypoints, prepared_mask = preprocess_clip(keypoints, mask, 16 / 9, PREPROCESS_SETTINGS)
    rng = np.random.default_rng(42)
    features = augment_and_finalize_clip(
        prepared_keypoints, prepared_mask, rng, AUGMENT_SETTINGS, num_frames=64
    )
    assert features.shape == (64, 393)


def test_centering_puts_shoulder_midpoint_at_origin_with_unit_width():
    keypoints, mask = make_fake_clip(num_frames=10)
    # Shoulders 0.2 apart, centred at (0.5, 0.4), in every frame.
    keypoints[:, LEFT_SHOULDER, :2] = [0.6, 0.4]
    keypoints[:, RIGHT_SHOULDER, :2] = [0.4, 0.4]

    normalized = normalize_by_shoulders(keypoints, mask, LEFT_SHOULDER, RIGHT_SHOULDER)

    midpoint = (normalized[:, LEFT_SHOULDER, :2] + normalized[:, RIGHT_SHOULDER, :2]) / 2
    width = np.linalg.norm(
        normalized[:, LEFT_SHOULDER, :2] - normalized[:, RIGHT_SHOULDER, :2], axis=1
    )
    np.testing.assert_allclose(midpoint, 0.0, atol=1e-6)
    np.testing.assert_allclose(width, 1.0, atol=1e-6)


def test_centering_keeps_missing_hand_at_zero():
    keypoints, mask = make_fake_clip(num_frames=10)
    keypoints[3, LEFT_HAND] = 0.0  # left hand not detected in frame 3
    mask[3, 1] = False

    normalized = normalize_by_shoulders(keypoints, mask, LEFT_SHOULDER, RIGHT_SHOULDER)

    assert (normalized[3, LEFT_HAND] == 0).all()
    assert (normalized[3, POSE] != 0).any()
