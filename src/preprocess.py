"""Turn raw keypoints (T, 130, 3) into a fixed-size model input (64, 393).

Steps for every clip:
    1. centre each frame on the midpoint of the shoulders
    2. divide by shoulder width (so body size / camera distance do not matter)
    3. fill short hand gaps (<= 5 frames) by linear interpolation
    4. trim frames at the start and end where no hand is visible
    5. resample to a fixed 64 frames
    6. flatten to (64, 390) and append the 3 mask columns -> (64, 393)

Augmentation functions (training only) are at the bottom of this file.

Mask columns: [pose, left hand, right hand]. The mask always means "detected
by MediaPipe"; interpolated frames keep mask = False, so the model can still
tell real detections from filled-in ones.
"""

import numpy as np

# Landmark layout written by extract_keypoints.py (fixed output format).
NUM_LANDMARKS = 130
POSE = slice(0, 25)
LEFT_HAND = slice(25, 46)
RIGHT_HAND = slice(46, 67)
FACE = slice(67, 130)

# Mask column for each hand: (mask column, landmark slice).
HANDS = [(1, LEFT_HAND), (2, RIGHT_HAND)]


# ---------------------------------------------------------------------------
# Steps 1-2: normalisation
# ---------------------------------------------------------------------------


def find_empty_points(keypoints: np.ndarray) -> np.ndarray:
    """(T, 130) bool: True where a landmark is all zeros, i.e. was not detected."""
    return (keypoints == 0).all(axis=-1)


def to_square_units(keypoints: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """Make x, y and z use the same unit (fraction of the image height).

    MediaPipe gives x as a fraction of image WIDTH and y as a fraction of
    image HEIGHT. On a 1920x1080 video one unit of x is 1.78x longer than one
    unit of y, which would distort distances and rotations. MediaPipe's z
    uses roughly the same scale as x, so it is converted the same way.
    """
    converted = keypoints.copy()
    converted[..., 0] *= aspect_ratio
    converted[..., 2] *= aspect_ratio
    return converted


def normalize_by_shoulders(
    keypoints: np.ndarray, mask: np.ndarray, left_shoulder: int, right_shoulder: int
) -> np.ndarray:
    """Centre every frame on the shoulder midpoint and divide by shoulder width.

    Only x and y are centred: hand and face z values are measured from their
    own reference points in MediaPipe, so subtracting the shoulder z would
    not make them more comparable. All three are divided by the width so
    they stay in the same unit.
    """
    pose_found = mask[:, 0]
    if not pose_found.any():
        raise ValueError("No pose detected in any frame; cannot find the shoulders.")

    left = keypoints[:, left_shoulder, :2]
    right = keypoints[:, right_shoulder, :2]
    centres = (left + right) / 2  # (T, 2)

    # Frames without a pose have no shoulders, so use the clip's average centre.
    centres[~pose_found] = centres[pose_found].mean(axis=0)

    # One shoulder width per clip (the median over frames). A per-frame width
    # would wobble with detection noise; the median also ignores a few bad frames.
    widths = np.linalg.norm(left[pose_found] - right[pose_found], axis=1)
    shoulder_width = float(np.median(widths))

    empty = find_empty_points(keypoints)
    normalized = keypoints.copy()
    normalized[..., :2] -= centres[:, None, :]
    normalized /= shoulder_width
    # Centring moved the "missing" zeros away from zero; put them back so
    # missing parts are still exactly zero.
    normalized[empty] = 0.0
    return normalized


# ---------------------------------------------------------------------------
# Steps 3-4: hand gaps and trimming
# ---------------------------------------------------------------------------


def find_gaps(missing: np.ndarray) -> list[tuple[int, int]]:
    """Return (first, last) frame of every run of True values in a 1-D bool array."""
    gaps = []
    start = None
    for frame, is_missing in enumerate(missing):
        if is_missing and start is None:
            start = frame
        elif not is_missing and start is not None:
            gaps.append((start, frame - 1))
            start = None
    if start is not None:
        gaps.append((start, len(missing) - 1))
    return gaps


def interpolate_hand_gaps(keypoints: np.ndarray, mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill short gaps where a hand was missing, by drawing a straight line
    between the last frame before and the first frame after the gap.

    Only gaps with a detected frame on BOTH sides are filled: at the start or
    end of a clip we do not know where the hand went. The mask is not changed.
    """
    filled = keypoints.copy()
    num_frames = len(keypoints)
    for mask_column, hand in HANDS:
        missing = ~mask[:, mask_column]
        for first, last in find_gaps(missing):
            gap_length = last - first + 1
            has_both_sides = first > 0 and last < num_frames - 1
            if gap_length > max_gap or not has_both_sides:
                continue
            before = keypoints[first - 1, hand]
            after = keypoints[last + 1, hand]
            for step, frame in enumerate(range(first, last + 1), start=1):
                weight = step / (gap_length + 1)  # 0 = before, 1 = after
                filled[frame, hand] = (1 - weight) * before + weight * after
    return filled


def trim_to_hands(keypoints: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove frames at the start and end where neither hand was detected.

    These frames are usually "hands down, not signing yet". If no hand is
    found at all, the clip is returned unchanged rather than emptied.
    """
    has_hand = mask[:, 1] | mask[:, 2]
    if not has_hand.any():
        return keypoints, mask
    hand_frames = np.flatnonzero(has_hand)
    first, last = hand_frames[0], hand_frames[-1]
    return keypoints[first : last + 1], mask[first : last + 1]


def preprocess_clip(
    keypoints: np.ndarray, mask: np.ndarray, aspect_ratio: float, settings: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Steps 1-4. Returns a clip of variable length; see finalize_clip for 5-6."""
    if keypoints.shape[1:] != (NUM_LANDMARKS, 3):
        raise ValueError(f"Expected keypoints of shape (T, 130, 3), got {keypoints.shape}")
    keypoints = to_square_units(keypoints, aspect_ratio)
    keypoints = normalize_by_shoulders(
        keypoints, mask, settings["left_shoulder_index"], settings["right_shoulder_index"]
    )
    keypoints = interpolate_hand_gaps(keypoints, mask, settings["max_gap_frames"])
    return trim_to_hands(keypoints, mask)


# ---------------------------------------------------------------------------
# Steps 5-6: fixed length and flattening
# ---------------------------------------------------------------------------


def resample_frames(
    keypoints: np.ndarray, mask: np.ndarray, num_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pick num_frames evenly spaced frames (repeating frames if the clip is short).

    We pick the nearest real frame instead of blending two frames: blending a
    detected hand with a missing (zero) hand would create a fake hand
    halfway between the real position and the origin.
    """
    positions = np.linspace(0, len(keypoints) - 1, num_frames).round().astype(int)
    return keypoints[positions], mask[positions]


def flatten_with_mask(keypoints: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(T, 130, 3) + (T, 3) -> (T, 393): 390 coordinates followed by 3 mask values."""
    flat = keypoints.reshape(len(keypoints), -1)
    return np.concatenate([flat, mask.astype(np.float32)], axis=1).astype(np.float32)


def finalize_clip(keypoints: np.ndarray, mask: np.ndarray, num_frames: int) -> np.ndarray:
    """Steps 5-6 without augmentation (used for test clips)."""
    keypoints, mask = resample_frames(keypoints, mask, num_frames)
    return flatten_with_mask(keypoints, mask)


# ---------------------------------------------------------------------------
# Augmentation (training only). There is deliberately NO left/right mirror:
# in sign language the dominant hand and its side carry meaning.
# ---------------------------------------------------------------------------


def random_time_crop_stretch(
    keypoints: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    factor_min: float,
    factor_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate signing 0.8x-1.2x as fast. Use BEFORE resampling to 64 frames.

    Because every clip is resampled to 64 frames afterwards, simply making
    the clip longer or shorter would change nothing. Instead:
      factor > 1: keep a random window of T / factor frames (a crop), so the
                  signing fills more of the 64 frames and looks slower.
      factor < 1: pad to T / factor frames by repeating the first/last frame
                  (random split between start and end), so the signing looks faster.
    """
    factor = rng.uniform(factor_min, factor_max)
    num_frames = len(keypoints)
    new_length = max(2, round(num_frames / factor))

    if new_length <= num_frames:
        start = rng.integers(0, num_frames - new_length + 1)
        return keypoints[start : start + new_length], mask[start : start + new_length]

    padding = new_length - num_frames
    pad_before = int(rng.integers(0, padding + 1))
    pad_after = padding - pad_before
    keypoints = np.pad(keypoints, ((pad_before, pad_after), (0, 0), (0, 0)), mode="edge")
    mask = np.pad(mask, ((pad_before, pad_after), (0, 0)), mode="edge")
    return keypoints, mask


def random_rotate(keypoints: np.ndarray, rng: np.random.Generator, max_degrees: float) -> np.ndarray:
    """Rotate x, y around the shoulder midpoint (the origin after normalisation).

    Missing points are (0, 0, 0) and a rotation around the origin keeps them there.
    """
    angle = np.deg2rad(rng.uniform(-max_degrees, max_degrees))
    cos, sin = np.cos(angle), np.sin(angle)
    rotated = keypoints.copy()
    x, y = keypoints[..., 0], keypoints[..., 1]
    rotated[..., 0] = cos * x - sin * y
    rotated[..., 1] = sin * x + cos * y
    return rotated


def random_scale(
    keypoints: np.ndarray, rng: np.random.Generator, scale_min: float, scale_max: float
) -> np.ndarray:
    return keypoints * rng.uniform(scale_min, scale_max)


def add_noise(keypoints: np.ndarray, rng: np.random.Generator, noise_std: float) -> np.ndarray:
    """Add small random jitter to detected points only; missing points stay zero."""
    noise = rng.normal(0.0, noise_std, size=keypoints.shape).astype(np.float32)
    detected = ~find_empty_points(keypoints)
    return keypoints + noise * detected[..., None]


def random_hand_dropout(
    keypoints: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    probability: float,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sometimes erase one hand in a few random frames, like a missed detection."""
    if rng.random() >= probability:
        return keypoints, mask
    keypoints, mask = keypoints.copy(), mask.copy()
    num_dropped = int(rng.integers(1, max_frames + 1))
    frames = rng.choice(len(keypoints), size=num_dropped, replace=False)
    mask_column, hand = HANDS[int(rng.integers(0, 2))]
    for frame in frames:
        keypoints[frame, hand] = 0.0
        mask[frame, mask_column] = False
    return keypoints, mask


def augment_and_finalize_clip(
    keypoints: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    settings: dict,
    num_frames: int,
) -> np.ndarray:
    """Steps 5-6 with random augmentation (used for training clips)."""
    keypoints, mask = random_time_crop_stretch(
        keypoints, mask, rng, settings["time_factor_min"], settings["time_factor_max"]
    )
    keypoints, mask = resample_frames(keypoints, mask, num_frames)
    keypoints = random_rotate(keypoints, rng, settings["rotation_degrees"])
    keypoints = random_scale(keypoints, rng, settings["scale_min"], settings["scale_max"])
    keypoints = add_noise(keypoints, rng, settings["noise_std"])
    keypoints, mask = random_hand_dropout(
        keypoints,
        mask,
        rng,
        settings["hand_dropout_probability"],
        settings["hand_dropout_max_frames"],
    )
    return flatten_with_mask(keypoints, mask)
