"""Extract body, hand and face keypoints from every video with MediaPipe Holistic.

This is a stand-in for Component 1: it writes the same output format, so the
rest of the NLP engine can be built before Component 1 is finished.

Output per video: data/keypoints/{video_stem}.npz with
    keypoints: float32 array (T, 130, 3)  -> x, y, z for each landmark
    mask:      bool array    (T, 3)       -> pose / left hand / right hand detected
Landmark order: 25 pose, 21 left hand, 21 right hand, 63 face.
Parts that were not detected are zeros. Every frame is kept, even empty ones,
so frame numbers always line up with the video.

Usage (from the project root):
    python src/extract_keypoints.py                    # all videos
    python src/extract_keypoints.py --preview          # also preview the first video
    python src/extract_keypoints.py --preview 008_t1   # also preview this video
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from data import load_config, resolve_path

NUM_HAND_LANDMARKS = 21

# Colours (BGR) for the preview video, one per body part.
PREVIEW_COLOURS = {
    "pose": (0, 255, 0),
    "left_hand": (255, 0, 0),
    "right_hand": (0, 0, 255),
    "face": (0, 255, 255),
}


def landmarks_to_array(landmark_list, indices: list[int]) -> np.ndarray:
    """Pick the given landmark indices and return them as an (len(indices), 3) array."""
    points = landmark_list.landmark
    return np.array([[points[i].x, points[i].y, points[i].z] for i in indices], dtype=np.float32)


def results_to_keypoints(
    results, num_pose: int, face_indices: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Turn one frame's MediaPipe results into (130, 3) keypoints and a (3,) mask.

    Each part starts as zeros and is only filled in if MediaPipe found it,
    so a missing hand gives zeros instead of an error.
    """
    pose = np.zeros((num_pose, 3), dtype=np.float32)
    left_hand = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
    right_hand = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
    face = np.zeros((len(face_indices), 3), dtype=np.float32)

    if results.pose_landmarks is not None:
        pose = landmarks_to_array(results.pose_landmarks, list(range(num_pose)))
    if results.left_hand_landmarks is not None:
        left_hand = landmarks_to_array(results.left_hand_landmarks, list(range(NUM_HAND_LANDMARKS)))
    if results.right_hand_landmarks is not None:
        right_hand = landmarks_to_array(results.right_hand_landmarks, list(range(NUM_HAND_LANDMARKS)))
    if results.face_landmarks is not None:
        face = landmarks_to_array(results.face_landmarks, face_indices)

    keypoints = np.concatenate([pose, left_hand, right_hand, face], axis=0)
    mask = np.array(
        [
            results.pose_landmarks is not None,
            results.left_hand_landmarks is not None,
            results.right_hand_landmarks is not None,
        ],
        dtype=bool,
    )
    return keypoints, mask


def enhance_low_light(frame_bgr: np.ndarray, clahe: cv2.CLAHE) -> np.ndarray:
    """Brighten dark frames with CLAHE on the lightness channel only.

    Working on L (from the LAB colour space) changes brightness and contrast
    but leaves the colours alone, so skin colour stays natural for MediaPipe.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = clahe.apply(lightness)
    lab = cv2.merge([lightness, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def draw_keypoints(frame_bgr: np.ndarray, keypoints: np.ndarray, num_pose: int) -> None:
    """Draw the 130 saved keypoints as dots, coloured by body part.

    We draw from the saved array (not MediaPipe's full output), so the
    preview shows exactly what goes into the .npz file.
    """
    height, width = frame_bgr.shape[:2]
    part_ranges = {
        "pose": (0, num_pose),
        "left_hand": (num_pose, num_pose + NUM_HAND_LANDMARKS),
        "right_hand": (num_pose + NUM_HAND_LANDMARKS, num_pose + 2 * NUM_HAND_LANDMARKS),
        "face": (num_pose + 2 * NUM_HAND_LANDMARKS, len(keypoints)),
    }
    for part, (start, end) in part_ranges.items():
        points = keypoints[start:end]
        if not points.any():  # all zeros = part not detected
            continue
        for x, y, _ in points:
            # x and y are fractions of the image size, so scale them to pixels.
            centre = (int(x * width), int(y * height))
            cv2.circle(frame_bgr, centre, 3, PREVIEW_COLOURS[part], -1)


def create_holistic(settings: dict):
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,  # video mode: track landmarks between frames
        model_complexity=settings["model_complexity"],
        min_detection_confidence=settings["min_detection_confidence"],
        min_tracking_confidence=settings["min_tracking_confidence"],
    )


def extract_video(
    video_path: Path, settings: dict, preview_path: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Run Holistic on every frame of one video. Returns keypoints (T,130,3), mask (T,3)."""
    num_pose = settings["num_pose_landmarks"]
    face_indices = settings["face_indices"]

    clahe = None
    if settings["low_light_enhancement"]:
        tile = settings["clahe_tile_grid_size"]
        clahe = cv2.createCLAHE(clipLimit=settings["clahe_clip_limit"], tileGridSize=(tile, tile))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f"OpenCV could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    all_keypoints = []
    all_masks = []

    # A new Holistic object per video, so tracking from the previous video
    # cannot leak into the first frames of this one.
    with create_holistic(settings) as holistic:
        for _ in tqdm(range(total_frames), desc=video_path.stem, leave=False):
            success, frame_bgr = capture.read()
            if not success:
                break
            if clahe is not None:
                frame_bgr = enhance_low_light(frame_bgr, clahe)

            # MediaPipe expects RGB; OpenCV reads BGR.
            results = holistic.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            keypoints, mask = results_to_keypoints(results, num_pose, face_indices)
            all_keypoints.append(keypoints)
            all_masks.append(mask)

            if preview_path is not None:
                if writer is None:
                    height, width = frame_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(preview_path), fourcc, fps, (width, height))
                draw_keypoints(frame_bgr, keypoints, num_pose)
                writer.write(frame_bgr)

    capture.release()
    if writer is not None:
        writer.release()
    return np.stack(all_keypoints), np.stack(all_masks)


def save_keypoints(output_path: Path, keypoints: np.ndarray, mask: np.ndarray) -> None:
    np.savez_compressed(output_path, keypoints=keypoints, mask=mask)


def detection_summary(video_stem: str, mask: np.ndarray) -> dict:
    """Percentage of frames in which each part was detected."""
    any_hand = mask[:, 1] | mask[:, 2]
    return {
        "video": video_stem,
        "frames": len(mask),
        "pose_%": 100 * mask[:, 0].mean(),
        "left_hand_%": 100 * mask[:, 1].mean(),
        "right_hand_%": 100 * mask[:, 2].mean(),
        "any_hand_%": 100 * any_hand.mean(),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic keypoints.")
    parser.add_argument(
        "--preview",
        nargs="?",
        const="FIRST",  # value used when --preview is given without a video name
        default=None,
        metavar="VIDEO_STEM",
        help="write a preview video with landmarks drawn (e.g. 008_t1; default: first video)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_config()
    settings = config["keypoints"]
    manifest = pd.read_csv(resolve_path(config["paths"]["manifest_csv"]), encoding="utf-8")
    keypoints_dir = resolve_path(config["paths"]["keypoints_dir"])
    figures_dir = resolve_path(config["paths"]["figures_dir"])

    video_paths = [resolve_path(p) for p in manifest["video_path"]]
    preview_stem = args.preview
    if preview_stem == "FIRST":
        preview_stem = video_paths[0].stem
    if preview_stem is not None and preview_stem not in [p.stem for p in video_paths]:
        raise SystemExit(f"--preview: no video named {preview_stem} in the manifest")

    print(f"Low-light enhancement: {'ON' if settings['low_light_enhancement'] else 'OFF'}")
    summaries = []
    for video_path in video_paths:
        output_path = keypoints_dir / f"{video_path.stem}.npz"
        wants_preview = video_path.stem == preview_stem
        preview_path = figures_dir / f"{video_path.stem}_preview.mp4" if wants_preview else None

        if output_path.exists() and not wants_preview:
            print(f"{video_path.name}: already processed, skipping")
            mask = np.load(output_path)["mask"]
        else:
            keypoints, mask = extract_video(video_path, settings, preview_path)
            if not output_path.exists():
                save_keypoints(output_path, keypoints, mask)
            print(f"{video_path.name}: {keypoints.shape} -> {output_path.name}")
            if preview_path is not None:
                print(f"  preview written to {preview_path}")
        summaries.append(detection_summary(video_path.stem, mask))

    print("\nDetection rate per video (% of frames):")
    table = pd.DataFrame(summaries)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.1f}"))


if __name__ == "__main__":
    main()
