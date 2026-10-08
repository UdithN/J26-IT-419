"""STEP 4 - turn raw keypoints (T,130,3) into fixed-size model inputs (64,390) + mask (64,3).

Run from project root:   python src/step4_preprocess.py
Output: data/processed/<video>.npz  and  data/processed/index.csv
"""
import re

import numpy as np
import pandas as pd

from common import KP_DIR, PROC_DIR

N_FRAMES = 64
POSE, LH, RH = slice(0, 25), slice(25, 46), slice(46, 67)   # face = 67:130
L_SH, R_SH = 11, 12                                          # shoulder indices inside the pose block
MAX_GAP = 5                                                  # fill hand gaps up to 5 frames


def fill_gaps(block, detected, max_gap=MAX_GAP):
    """Linear interpolation across short runs of frames where a hand was not detected."""
    out, filled = block.copy(), detected.copy()
    idx = np.where(detected)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k, t in enumerate(range(a + 1, b), start=1):
                w = k / (gap + 1)
                out[t] = (1 - w) * block[a] + w * block[b]
                filled[t] = True
    return out, filled


def preprocess_clip(kp, mask):
    kp = kp.astype(np.float32).copy()
    mask = mask.astype(bool).copy()
    valid = np.any(kp != 0, axis=-1)                          # (T,130) which landmarks exist

    # 1) interpolate short hand gaps
    for sl, col in ((LH, 1), (RH, 2)):
        kp[:, sl], filled = fill_gaps(kp[:, sl], mask[:, col])
        mask[:, col] = filled
        valid[:, sl] = filled[:, None]

    # 2) trim leading/trailing frames with no hand at all
    hands = mask[:, 1] | mask[:, 2]
    if hands.any():
        first, last = np.where(hands)[0][[0, -1]]
        kp, mask, valid = kp[first:last + 1], mask[first:last + 1], valid[first:last + 1]
    t_trimmed = len(kp)

    # 3) centre on shoulder midpoint, scale by shoulder width
    sh = kp[:, [L_SH, R_SH], :]                               # (T,2,3)
    centre = sh.mean(axis=1)                                  # (T,3)
    good = np.any(sh != 0, axis=(1, 2))
    if good.any():
        centre[~good] = np.median(centre[good], axis=0)
        width = np.linalg.norm(sh[good, 0, :2] - sh[good, 1, :2], axis=1)
        scale = float(np.median(width))
    else:
        scale = 1.0
    scale = scale if scale > 1e-6 else 1.0
    kp = (kp - centre[:, None, :]) / scale
    kp[~valid] = 0.0                                          # missing stays exactly zero

    # 4) resample to a fixed number of frames (nearest frame)
    idx = np.linspace(0, len(kp) - 1, N_FRAMES).round().astype(int)
    feats = kp[idx].reshape(N_FRAMES, -1).astype(np.float32)  # (64, 390)
    return feats, mask[idx], t_trimmed


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(KP_DIR.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files in {KP_DIR}. Run keypoint extraction first.")
    rows = []
    for f in files:
        m = re.match(r"(\d+)_t(\d+)", f.stem)
        if not m:
            print("skip (bad name):", f.name)
            continue
        d = np.load(f)
        feats, mask, t_trim = preprocess_clip(d["keypoints"], d["mask"])
        assert feats.shape == (N_FRAMES, 390) and not np.isnan(feats).any()
        np.savez_compressed(PROC_DIR / f"{f.stem}.npz", feats=feats, mask=mask)
        rows.append({
            "stem": f.stem, "sentence_id": int(m.group(1)), "take": int(m.group(2)),
            "frames_raw": len(d["keypoints"]), "frames_after_trim": t_trim,
            "lh_rate": round(float(mask[:, 1].mean()), 2), "rh_rate": round(float(mask[:, 2].mean()), 2),
            "max_abs": round(float(np.abs(feats).max()), 2),
        })
    idx = pd.DataFrame(rows)
    idx.to_csv(PROC_DIR / "index.csv", index=False)
    print(idx.to_string(index=False))
    print(f"\nProcessed {len(idx)} clips -> {PROC_DIR}")
    print("max_abs should be small (roughly < 15). Huge values mean a bad clip.")
    print("Clips per sentence:", idx.groupby("sentence_id").size().value_counts().to_dict(), "(takes: count of sentences)")


if __name__ == "__main__":
    main()
