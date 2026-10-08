"""STEP 5 - text-only baselines (no video), 5-fold cross-validation by sentence.

Run from project root:
  python src/step5_baselines.py retrieval            # fast, no GPU
  python src/step5_baselines.py nllb --folds 0       # test ONE fold first
  python src/step5_baselines.py nllb                 # all 5 folds
"""
import argparse
import difflib
import random
import time

import pandas as pd

from common import RUNS, SEED, load_sentences, make_folds, score_predictions

MODEL_ID = "facebook/nllb-200-distilled-600M"
CODES = {"si": "sin_Sinh", "en": "eng_Latn"}


# ---------------------------------------------------------------- simple baselines
def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def retrieve(tokens, train_df):
    """Return the training row whose gloss is most similar (Jaccard, then order similarity)."""
    best, best_key = None, (-1, -1)
    for _, r in train_df.iterrows():
        key = (jaccard(tokens, r["gloss_tokens"]), difflib.SequenceMatcher(None, tokens, r["gloss_tokens"]).ratio())
        if key > best_key:
            best, best_key = r, key
    return best


def gloss_concat_en(tokens):
    """'Label list' baseline: just the English gloss words in sign order."""
    words = [t.lower().replace("-", " ") for t in tokens if t != "?"]
    return " ".join(words) + ("?" if "?" in tokens else "")


def run_retrieval(df, folds):
    rows = []
    for k, f in enumerate(folds):
        train, test = df[df.id.isin(f["train"])], df[df.id.isin(f["test"])]
        for _, r in test.iterrows():
            hit = retrieve(r["gloss_tokens"], train)
            rows.append(dict(model="retrieval", fold=k, id=r["id"], lang="si", ref=r["sinhala"], pred=hit["sinhala"]))
            rows.append(dict(model="retrieval", fold=k, id=r["id"], lang="en", ref=r["english"], pred=hit["english"]))
            rows.append(dict(model="gloss_concat", fold=k, id=r["id"], lang="en", ref=r["english"], pred=gloss_concat_en(r["gloss_tokens"])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- NLLB + LoRA (gloss -> text)
def gloss_to_source(tokens):
    return " ".join(t.lower().replace("-", " ") for t in tokens)


def train_and_predict(train_df, test_df, epochs, bs, lr):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    random.seed(SEED)
    torch.manual_seed(SEED)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="SEQ_2_SEQ_LM")
    model = get_peft_model(base, cfg).to(dev)
    eos, pad = tok.eos_token_id, tok.pad_token_id
    en_id = tok.convert_tokens_to_ids(CODES["en"])

    def src_ids(text):                       # [eng_Latn] words </s>
        return [en_id] + tok(text, add_special_tokens=False)["input_ids"][:60] + [eos]

    def tgt_ids(text, lang):                 # [lang] words </s>
        return [tok.convert_tokens_to_ids(CODES[lang])] + tok(text, add_special_tokens=False)["input_ids"][:60] + [eos]

    examples = []
    for _, r in train_df.iterrows():
        s = src_ids(gloss_to_source(r["gloss_tokens"]))
        examples.append((s, tgt_ids(r["sinhala"], "si")))
        examples.append((s, tgt_ids(r["english"], "en")))

    def collate(batch):
        ms, mt = max(len(s) for s, _ in batch), max(len(t) for _, t in batch)
        x = torch.tensor([s + [pad] * (ms - len(s)) for s, _ in batch])
        y = torch.tensor([t + [-100] * (mt - len(t)) for _, t in batch])
        return x.to(dev), (x != pad).long().to(dev), y.to(dev)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for ep in range(1, epochs + 1):
        random.shuffle(examples)
        total, n = 0.0, 0
        for i in range(0, len(examples), bs):
            x, am, y = collate(examples[i:i + bs])
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=x, attention_mask=am, labels=y).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            total, n = total + loss.item(), n + 1
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"  epoch {ep:>3}/{epochs}  loss {total / n:.4f}  peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    model.eval()
    out_rows = []
    for _, r in test_df.iterrows():
        x = torch.tensor([src_ids(gloss_to_source(r["gloss_tokens"]))]).to(dev)
        for lang in ("si", "en"):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                gen = model.generate(input_ids=x, attention_mask=(x != pad).long(), num_beams=4, max_new_tokens=64,
                                     forced_bos_token_id=tok.convert_tokens_to_ids(CODES[lang]))
            out_rows.append((r["id"], lang, tok.batch_decode(gen, skip_special_tokens=True)[0]))
    del model, base, opt
    torch.cuda.empty_cache()
    return out_rows


def run_nllb(df, folds, which, epochs, bs, lr):
    rows = []
    for k in which:
        f = folds[k]
        train, test = df[df.id.isin(f["train"])], df[df.id.isin(f["test"])]
        print(f"\n=== fold {k}: train {len(train)} / test {len(test)} sentences ===")
        t0 = time.time()
        preds = train_and_predict(train, test, epochs, bs, lr)
        print(f"  fold time {time.time() - t0:.0f}s")
        ref = {(r["id"], "si"): r["sinhala"] for _, r in test.iterrows()} | {(r["id"], "en"): r["english"] for _, r in test.iterrows()}
        for i, lang, text in preds:
            rows.append(dict(model="gloss_nllb", fold=k, id=i, lang=lang, ref=ref[(i, lang)], pred=text))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["retrieval", "nllb"])
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()

    df = load_sentences()
    folds = make_folds(df)
    print(f"{len(df)} sentences, {len(folds)} folds")
    preds = run_retrieval(df, folds) if a.which == "retrieval" else run_nllb(df, folds, a.folds, a.epochs, a.bs, a.lr)

    out = RUNS / ("baseline_" + a.which)
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    res = []
    for m, g in preds.groupby("model"):
        s = score_predictions(g)
        s.insert(0, "model", m)
        res.append(s)
    res = pd.concat(res)
    res.to_csv(out / "metrics.csv", index=False)
    print("\n", res.to_string(index=False))
    print("\nExample predictions:")
    print(preds.sample(min(6, len(preds)), random_state=1)[["model", "id", "lang", "ref", "pred"]].to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
