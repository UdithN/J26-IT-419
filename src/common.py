"""Shared helpers for Component 2. Put this file in src/ and run scripts from the project root."""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KP_DIR = DATA / "keypoints"
PROC_DIR = DATA / "processed"
RUNS = ROOT / "outputs" / "runs"
SEED = 42

# A drop of punctuation is applied before metrics so "ඕනේ." and "ඕනේ" count as the same word.
_PUNCT = re.compile(r"[.,!?;:\"'()\[\]]")


def norm(text) -> str:
    """NFC-normalise (keeps Sinhala ZWJ intact) and strip spaces."""
    return unicodedata.normalize("NFC", str(text)).strip()


def load_sentences(csv_path: Path = DATA / "sentences.csv") -> pd.DataFrame:
    """Read sentences.csv -> columns: id, domain, gloss, sinhala, english, gloss_tokens."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.iloc[:, :5].copy()
    df.columns = ["id", "domain", "gloss", "sinhala", "english"]
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)
    df["gloss_tokens"] = df["gloss"].apply(lambda g: re.findall(r"\[(.*?)\]", str(g)))
    df["sinhala"] = df["sinhala"].map(norm)
    df["english"] = df["english"].map(norm)
    return df.sort_values("id").reset_index(drop=True)


def make_folds(df: pd.DataFrame, k: int = 5):
    """5-fold split by SENTENCE, stratified by domain. Saved so every experiment uses the same folds."""
    from sklearn.model_selection import StratifiedKFold

    path = ROOT / "outputs" / "folds.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["ids"] == df["id"].tolist():
            return saved["folds"]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    folds = []
    for tr, te in skf.split(df["id"], df["domain"]):
        folds.append({"train": df.iloc[tr]["id"].tolist(), "test": df.iloc[te]["id"].tolist()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ids": df["id"].tolist(), "folds": folds}))
    return folds


# ---------------- metrics ----------------
def toks(text: str):
    return _PUNCT.sub("", norm(text).lower()).split()


def _lcs(a, b) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(hyp: str, ref: str) -> float:
    """ROUGE-L F1 on whitespace tokens (own version: the stock library deletes Sinhala letters)."""
    h, r = toks(hyp), toks(ref)
    if not h or not r:
        return 0.0
    lcs = _lcs(h, r)
    if lcs == 0:
        return 0.0
    p, rc = lcs / len(h), lcs / len(r)
    return 2 * p * rc / (p + rc)


def score(hyps, refs) -> dict:
    """BLEU, chrF++, ROUGE-L, exact match for one language."""
    out = {"n": len(hyps)}
    try:
        import sacrebleu

        out["BLEU"] = round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2)
        out["chrF++"] = round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2)
    except ImportError:
        out["BLEU"] = out["chrF++"] = float("nan")
    out["ROUGE-L"] = round(100 * float(np.mean([rouge_l(h, r) for h, r in zip(hyps, refs)])), 2)
    out["ExactMatch"] = round(100 * float(np.mean([toks(h) == toks(r) for h, r in zip(hyps, refs)])), 2)
    return out


def score_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    """pred_df needs columns: lang ('si'/'en'), ref, pred."""
    rows = []
    for lang, g in pred_df.groupby("lang"):
        rows.append({"lang": lang, **score(g["pred"].fillna("").tolist(), g["ref"].tolist())})
    return pd.DataFrame(rows)
