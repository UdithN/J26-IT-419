"""Train the gloss-free model.
  Phase 1: LLM frozen, train encoder + projector only.
  Phase 2: also unfreeze LoRA (small learning rate).
Examples (Colab):
  python src/gf/train.py --lang si --phase 1 --overfit 5 --epochs 40 --out OUT_TEST
  python src/gf/train.py --lang si --phase 1 --epochs 30 --out CKPT_SI
  python src/gf/train.py --lang si --phase 2 --epochs 20 --resume CKPT_SI --out CKPT_SI
"""
import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import SEED, load_sentences
from gf_data import SignDataset, collate, list_clips, make_split
from gf_model import SignLLM, encode_targets, load_ckpt, load_llm, save_ckpt, set_trainable_lora


def hf_login():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        from huggingface_hub import login
        login(tok.strip())


@torch.no_grad()
def eval_loss(sl, loader, dev):
    sl.eval()
    tot, n = 0.0, 0
    for x, _, _, texts in loader:
        t, m = encode_targets(sl.tok, texts, sl.eos, dev)
        with torch.autocast("cuda", dtype=torch.float16):
            tot += sl(x.to(dev), t, m).item() * len(texts)
        n += len(texts)
    sl.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["si", "en"], required=True)
    ap.add_argument("--phase", type=int, choices=[1, 2], default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="encoder/projector lr (default 1e-3 phase1, 3e-4 phase2)")
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--resume", type=str, default=None, help="checkpoint folder to start from")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--overfit", type=int, default=0, help="sanity test: train+test on N clips, no augmentation")
    ap.add_argument("--n_tokens", type=int, default=16)
    ap.add_argument("--min_takes_test", type=int, default=2)
    a = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda"
    hf_login()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences()
    clips = list_clips()
    train_stems, test_stems = make_split(clips, set(sentences["id"]), a.min_takes_test, save=not a.overfit)
    if a.overfit:
        seen, keep = set(), []
        for _, r in clips.sort_values(["id", "take"]).iterrows():
            if r["id"] not in seen and len(keep) < a.overfit:
                seen.add(r["id"]); keep.append(r["stem"])
        train_stems, test_stems = keep, []
        print("OVERFIT TEST on:", keep)
    print(f"train clips: {len(train_stems)} | held-out test clips: {len(test_stems)}")

    tok, llm, base = load_llm(a.lang)
    sl = SignLLM(llm, tok, a.lang, n_tokens=a.n_tokens)
    sl.encoder.to(dev); sl.projector.to(dev)
    if a.resume:
        load_ckpt(Path(a.resume) / "ckpt.pt", sl)
        print("resumed from", a.resume)
    if a.phase == 2:
        print("LoRA params unfrozen:", set_trainable_lora(llm, a.lang))
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.config.use_cache = False

    lr_main = a.lr or (1e-3 if a.phase == 1 else 3e-4)
    main_params = list(sl.encoder.parameters()) + list(sl.projector.parameters())
    lora_params = [p for n, p in llm.named_parameters() if p.requires_grad]
    groups = [{"params": main_params, "lr": lr_main}]
    if lora_params:
        groups.append({"params": lora_params, "lr": a.lr_lora})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    print(f"trainable: encoder+projector {sum(p.numel() for p in main_params)/1e6:.1f}M | LoRA {sum(p.numel() for p in lora_params)/1e6:.1f}M")

    tr = DataLoader(SignDataset(train_stems, sentences, a.lang, augment=not a.overfit),
                    batch_size=a.bs, shuffle=True, collate_fn=collate, num_workers=0)
    te = DataLoader(SignDataset(test_stems, sentences, a.lang), batch_size=a.bs, collate_fn=collate,
                    num_workers=0) if test_stems else None

    steps_per_epoch = math.ceil(len(tr) / a.accum)
    total, warm = a.epochs * steps_per_epoch, max(1, int(0.1 * a.epochs * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    scaler = torch.amp.GradScaler("cuda")
    sl.train()
    meta = {"lang": a.lang, "n_tokens": a.n_tokens, "enc_dim": sl.enc_dim, "enc_layers": sl.enc_layers, "phase": a.phase}

    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        tot, n = 0.0, 0
        for i, (x, _, _, texts) in enumerate(tr):
            t, m = encode_targets(tok, texts, sl.eos, dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = sl(x.to(dev), t, m)
            scaler.scale(loss / a.accum).backward()
            if (i + 1) % a.accum == 0 or i == len(tr) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(main_params + lora_params, 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item() * len(texts); n += len(texts)
        msg = f"epoch {ep:>3}/{a.epochs} train loss {tot/n:.4f} | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB | {time.time()-t0:.0f}s"
        if te is not None and (ep % 5 == 0 or ep == a.epochs):
            msg += f" | held-out loss {eval_loss(sl, te, dev):.4f}"
        print(msg, flush=True)
        if ep % 5 == 0 or ep == a.epochs:
            save_ckpt(out / "ckpt.pt", sl, meta)          # always the LAST epoch: no model selection on test data
    print("saved", out / "ckpt.pt")


if __name__ == "__main__":
    main()
