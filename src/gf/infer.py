"""Predict with a trained checkpoint and write predictions.csv (same columns as the baselines).
  python src/gf/infer.py --lang si --ckpt CKPT_SI --tag sft            (held-out clips)
  python src/gf/infer.py --lang si --ckpt CKPT_SI --tag sft --split all
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import RUNS, load_sentences, score_predictions
from gf_data import SignDataset, collate, list_clips, make_split
from gf_model import SignLLM, load_ckpt, load_llm
from train import hf_login


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["si", "en"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="sft")
    ap.add_argument("--split", choices=["test", "all"], default="test")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--beams", type=int, default=4)
    a = ap.parse_args()
    dev = "cuda"
    hf_login()

    sentences = load_sentences()
    clips = list_clips()
    train_stems, test_stems = make_split(clips, set(sentences["id"]), save=False)
    stems = clips["stem"].tolist() if a.split == "all" else test_stems
    note = ""
    if not stems:
        stems, note = train_stems, "TRAIN CLIPS (no held-out takes) - not a generalisation result"
        print("WARNING:", note)

    meta = torch.load(Path(a.ckpt) / "ckpt.pt", map_location="cpu", weights_only=False)["meta"]
    tok, llm, _ = load_llm(a.lang)
    sl = SignLLM(llm, tok, a.lang, n_tokens=meta["n_tokens"], enc_dim=meta["enc_dim"], enc_layers=meta["enc_layers"])
    sl.encoder.to(dev); sl.projector.to(dev)
    load_ckpt(Path(a.ckpt) / "ckpt.pt", sl)
    sl.eval()

    ds = SignDataset(stems, sentences, a.lang)
    rows = []
    for x, st, ids, refs in DataLoader(ds, batch_size=a.bs, collate_fn=collate, num_workers=0):
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            gen, mask, texts = sl.generate(x.to(dev), num_beams=a.beams)
            conf = sl.confidence(x.to(dev), gen, mask).cpu().tolist()
        ms = (time.time() - t0) * 1000 / len(ids)
        for s, i, r, p, c in zip(st, ids, refs, texts, conf):
            rows.append(dict(model=f"gf_{a.tag}", fold=0, id=i, lang=a.lang, ref=r, pred=p,
                             confidence=round(c, 4), stem=s, ms=round(ms), note=note))
    df = pd.DataFrame(rows)
    out = RUNS / f"gf_{a.lang}_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    res = score_predictions(df)
    res.to_csv(out / "metrics.csv", index=False)
    print(res.to_string(index=False))
    print(df[["stem", "ref", "pred", "confidence"]].head(8).to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()
