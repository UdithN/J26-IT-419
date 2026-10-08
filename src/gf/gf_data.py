"""Data for the gloss-free model: clip list, train/test split, augmentation, Dataset.
Put in src/gf/. Gloss is NEVER used here."""
import json
import re

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:          # lets the numpy parts be tested without torch
    torch = None
    Dataset = object

from common import PROC_DIR, ROOT

N_FRAMES = 64
N_LAND = 130


def list_clips() -> pd.DataFrame:
    rows = []
    for f in sorted(PROC_DIR.glob("*.npz")):
        m = re.match(r"(\d+)_t(\d+)", f.stem)
        if m:
            rows.append({"stem": f.stem, "id": int(m.group(1)), "take": int(m.group(2))})
    return pd.DataFrame(rows, columns=["stem", "id", "take"])


def make_split(clips: pd.DataFrame, valid_ids, min_takes_for_test: int = 2, save: bool = True):
    """Sentence with >= min_takes_for_test takes: the LAST take is held out for testing.
    Sentences with fewer takes are train-only (no honest test possible for them)."""
    train, test, skipped = [], [], []
    for sid, g in clips.groupby("id"):
        g = g.sort_values("take")
        if sid not in valid_ids:
            skipped.append(int(sid))
            continue
        if len(g) >= min_takes_for_test:
            train += g.iloc[:-1]["stem"].tolist()
            test.append(g.iloc[-1]["stem"])
        else:
            train += g["stem"].tolist()
    if skipped:
        print("WARNING: clips with no sentence in sentences.csv (ignored): ids", skipped)
    if not test:
        print("WARNING: no held-out takes. Any test score is on TRAINING clips and proves nothing about generalisation.")
    if save:
        p = ROOT / "outputs" / "video_split.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"train": train, "test": test}, indent=1))
    return train, test


def augment_clip(feats, mask, rng):
    """Light augmentation on preprocessed features (64,390)+mask (64,3). Never mirrors left/right."""
    T = feats.shape[0]
    keep = rng.uniform(0.85, 1.0)                              # temporal crop, then resample to T
    L = max(8, int(T * keep))
    start = int(rng.integers(0, T - L + 1))
    idx = np.linspace(start, start + L - 1, T).round().astype(int)
    f = feats[idx].reshape(T, N_LAND, 3).copy()
    m = mask[idx].copy()
    valid = np.any(f != 0, axis=-1)
    a = np.deg2rad(rng.uniform(-8, 8))                         # small rotation in the image plane
    c, s = np.cos(a), np.sin(a)
    x, y = f[..., 0].copy(), f[..., 1].copy()
    f[..., 0], f[..., 1] = c * x - s * y, s * x + c * y
    f *= rng.uniform(0.95, 1.05)
    f += rng.normal(0, 0.005, f.shape).astype(np.float32)
    f[~valid] = 0.0                                            # missing landmarks stay exactly zero
    return f.reshape(T, -1).astype(np.float32), m


class SignDataset(Dataset):
    def __init__(self, stems, sentences: pd.DataFrame, lang: str, augment: bool = False, seed: int = 42):
        col = {"si": "sinhala", "en": "english"}[lang]
        text = dict(zip(sentences["id"], sentences[col]))
        self.items = []
        for s in stems:
            d = np.load(PROC_DIR / f"{s}.npz")
            sid = int(re.match(r"(\d+)_t", s).group(1))
            self.items.append((s, sid, d["feats"].astype(np.float32), d["mask"].astype(bool), text[sid]))
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem, sid, feats, mask, text = self.items[i]
        if self.augment:
            feats, mask = augment_clip(feats, mask, self.rng)
        x = np.concatenate([feats, mask.astype(np.float32)], axis=1)      # (64, 393)
        return torch.from_numpy(x), stem, sid, text


def collate(batch):
    x = torch.stack([b[0] for b in batch])
    return x, [b[1] for b in batch], [b[2] for b in batch], [b[3] for b in batch]
