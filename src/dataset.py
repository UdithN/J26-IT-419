"""PyTorch Dataset that gives (keypoints, gloss, Sinhala, English, sentence_id).

Usage (from the project root), to see the split and one example:
    python src/dataset.py
"""

import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data import get_video_info, load_config, load_sentences, resolve_path
from preprocess import augment_and_finalize_clip, finalize_clip, preprocess_clip


def load_clip_table(config: dict) -> pd.DataFrame:
    """One row per video: manifest columns plus that sentence's gloss and translations."""
    manifest = pd.read_csv(resolve_path(config["paths"]["manifest_csv"]), encoding="utf-8")
    sentences = load_sentences(resolve_path(config["paths"]["sentences_csv"]))
    sentences = sentences.rename(columns={"id": "sentence_id"})
    sentences["sentence_id"] = sentences["sentence_id"].astype(int)
    # "inner" keeps only videos whose sentence exists in the CSV.
    return manifest.merge(sentences, on="sentence_id", how="inner")


def print_loud_warning(lines: list[str]) -> None:
    border = "!" * 78
    print(f"\n{border}")
    for line in lines:
        print(f"!!  {line}")
    print(f"{border}\n")


def split_clips(clips: pd.DataFrame, train_takes: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split clips into train and test sets, per sentence.

    A sentence that has a take outside train_takes (e.g. t2) trains on its
    train_takes and is tested on the other takes: a real held-out test.
    A sentence with only train_takes (e.g. only t1) has nothing held out, so
    its t1 clip is used for BOTH training and testing.
    """
    is_train_take = clips["take"].isin(train_takes)
    sentences_with_test_take = set(clips.loc[~is_train_take, "sentence_id"])
    only_train_take = ~clips["sentence_id"].isin(sentences_with_test_take)

    train = clips[is_train_take]
    test = clips[~is_train_take | only_train_take]

    reused_ids = sorted(clips.loc[only_train_take, "sentence_id"].unique())
    if reused_ids:
        print_loud_warning(
            [
                f"WARNING: {len(reused_ids)} of {clips['sentence_id'].nunique()} sentences "
                "have only a training take.",
                "Their TEST clips are the SAME clips used for TRAINING.",
                "Scores on these sentences are NOT a generalization result.",
                f"Sentence IDs: {reused_ids}",
            ]
        )
    return train.reset_index(drop=True), test.reset_index(drop=True)


class SignLanguageDataset(Dataset):
    """Keypoint clips with their gloss, Sinhala and English targets.

    The deterministic preprocessing (normalise, fill gaps, trim) runs once in
    __init__. Augmentation is random, so it runs every time a clip is fetched,
    giving a slightly different version of each clip in every epoch.
    """

    def __init__(self, clips: pd.DataFrame, config: dict, augment: bool) -> None:
        self.clips = clips.reset_index(drop=True)
        self.augment = augment
        self.num_frames = config["preprocess"]["num_frames"]
        self.augment_settings = config["augment"]
        # One generator for this dataset, seeded from config, so the random
        # augmentations are the same on every run (works because num_workers=0).
        self.rng = np.random.default_rng(config["seed"])

        keypoints_dir = resolve_path(config["paths"]["keypoints_dir"])
        self.prepared = []  # list of (keypoints, mask) after steps 1-4
        for video_path in self.clips["video_path"]:
            video_path = resolve_path(video_path)
            saved = np.load(keypoints_dir / f"{video_path.stem}.npz")
            video_info = get_video_info(video_path)
            aspect_ratio = video_info["width"] / video_info["height"]
            self.prepared.append(
                preprocess_clip(saved["keypoints"], saved["mask"], aspect_ratio, config["preprocess"])
            )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict:
        keypoints, mask = self.prepared[index]
        if self.augment:
            features = augment_and_finalize_clip(
                keypoints, mask, self.rng, self.augment_settings, self.num_frames
            )
        else:
            features = finalize_clip(keypoints, mask, self.num_frames)

        row = self.clips.iloc[index]
        return {
            "keypoints": torch.from_numpy(features),  # (64, 393) float32
            "gloss": " ".join(row["gloss_tokens"]),  # e.g. "I HOME NOW GO"
            "sinhala": row["sinhala"],
            "english": row["english"],
            "sentence_id": int(row["sentence_id"]),
        }


def make_datasets(config: dict) -> tuple[SignLanguageDataset, SignLanguageDataset]:
    clips = load_clip_table(config)
    train_clips, test_clips = split_clips(clips, config["split"]["train_takes"])
    train_dataset = SignLanguageDataset(train_clips, config, augment=True)
    test_dataset = SignLanguageDataset(test_clips, config, augment=False)
    return train_dataset, test_dataset


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Sinhala on the Windows console
    config = load_config()
    train_dataset, test_dataset = make_datasets(config)
    print(f"Train clips: {len(train_dataset)}   Test clips: {len(test_dataset)}")
    print("Train videos:", [p.split("/")[-1] for p in train_dataset.clips["video_path"]])
    print("Test videos: ", [p.split("/")[-1] for p in test_dataset.clips["video_path"]])

    example = train_dataset[0]
    print("\nOne training example:")
    for key, value in example.items():
        shown = tuple(value.shape) if isinstance(value, torch.Tensor) else value
        print(f"  {key:<12}: {shown}")

    # num_workers=0: required on Windows here, and keeps augmentation reproducible.
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print("\nOne batch of 4:")
    print(f"  keypoints   : {tuple(batch['keypoints'].shape)}")
    print(f"  sentence_id : {batch['sentence_id'].tolist()}")
    print(f"  english     : {batch['english']}")


if __name__ == "__main__":
    main()
