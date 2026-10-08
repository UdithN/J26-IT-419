"""Load the sentence table and the list of videos.

Other scripts import these functions so every part of the project reads the
data in exactly the same way.
"""

import re
import unicodedata
from pathlib import Path

import cv2
import pandas as pd
import yaml

# The project root is the folder above src/. Paths in config.yaml are relative
# to it, so scripts work no matter which folder you run them from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Original CSV header -> short names used in the code.
COLUMN_NAMES = {
    "ID": "id",
    "Domain": "domain",
    "Continuous SSL Gesture Sequence": "gloss",
    "Sinhala Target Translation": "sinhala",
    "English Target Translation": "english",
}

# Matches "[HELLO]", "[THANK-YOU]", "[?]" and captures the text inside the brackets.
GLOSS_TOKEN_PATTERN = re.compile(r"\[([^\]]+)\]")

# Matches "008_t1.mp4" or "008_t1_s2.mp4": sentence id, take, optional signer.
VIDEO_NAME_PATTERN = re.compile(r"^(\d{3})_t(\d+)(?:_(s\d+))?\.mp4$")


def load_config(config_path: Path = PROJECT_ROOT / "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(relative_path: str) -> Path:
    """Turn a path from config.yaml into an absolute path under the project root."""
    return PROJECT_ROOT / relative_path


def normalize_text(text: str) -> str:
    """NFC-normalize text and trim surrounding spaces.

    NFC makes visually identical Sinhala strings have identical code points,
    so comparisons and tokenization are consistent. str.strip() only removes
    whitespace, so the zero-width joiner (U+200D) that Sinhala conjuncts
    need is kept.
    """
    return unicodedata.normalize("NFC", text).strip()


def parse_gloss(gloss_text: str) -> list[str]:
    """"[I] [HOME] [NOW] [GO]" -> ["I", "HOME", "NOW", "GO"].

    "[?]" becomes the token "?", which we keep because it marks a question.
    """
    return [token.strip() for token in GLOSS_TOKEN_PATTERN.findall(gloss_text)]


def load_sentences(csv_path: Path) -> pd.DataFrame:
    """Read sentences.csv with short column names, normalized text and gloss lists."""
    # dtype=str + keep_default_na=False: empty cells become "" instead of NaN,
    # so text functions never crash and empty cells are easy to count.
    # utf-8-sig also handles a BOM, which Excel adds when it saves a CSV.
    table = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    table = table.rename(columns=COLUMN_NAMES)

    for column in ["domain", "gloss", "sinhala", "english"]:
        table[column] = table[column].map(normalize_text)

    # "Int64" (capital I) allows a missing value, so one bad ID is reported
    # by the validator instead of crashing the loader.
    table["id"] = pd.to_numeric(table["id"].str.strip(), errors="coerce").astype("Int64")
    table["gloss_tokens"] = table["gloss"].map(parse_gloss)
    return table


def parse_video_filename(file_name: str, default_signer: str) -> dict | None:
    """Read sentence id, take and signer from a video file name.

    Returns None if the name does not follow the {id:03d}_t{take}.mp4 pattern.
    """
    match = VIDEO_NAME_PATTERN.match(file_name)
    if match is None:
        return None
    sentence_id, take, signer = match.groups()
    return {
        "sentence_id": int(sentence_id),
        "take": int(take),
        "signer": signer if signer is not None else default_signer,
    }


def build_manifest(videos_dir: Path, default_signer: str) -> pd.DataFrame:
    """List every correctly named video in videos_dir, one row per file."""
    rows = []
    for video_path in sorted(videos_dir.glob("*.mp4")):
        info = parse_video_filename(video_path.name, default_signer)
        if info is None:
            # Skipped files would otherwise disappear silently from the dataset.
            print(f"WARNING: skipping badly named video: {video_path.name}")
            continue
        # Store paths relative to the project root, with "/" separators,
        # so the manifest still works if the project folder is moved.
        relative_path = video_path.relative_to(PROJECT_ROOT).as_posix()
        rows.append({"video_path": relative_path, **info})

    columns = ["video_path", "sentence_id", "take", "signer"]
    return pd.DataFrame(rows, columns=columns)


def save_manifest(manifest: pd.DataFrame, manifest_path: Path) -> None:
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")


def get_video_info(video_path: Path) -> dict:
    """Return fps, frame count, duration (seconds), width and height of a video.

    The frame count comes from the file header. It is usually right, but it
    can be slightly off for some encoders.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f"OpenCV could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height,
    }
