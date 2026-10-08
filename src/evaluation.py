"""Shared helpers for the text baselines: folds, scores and saving a run.

Both baselines use exactly the same folds and metrics, so their numbers can
be compared directly.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
from sklearn.model_selection import StratifiedKFold

from data import PROJECT_ROOT

LANGUAGES = ["sinhala", "english"]


def make_folds(sentences: pd.DataFrame, num_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split sentences into folds; each domain is spread evenly across folds.

    Splitting by sentence means a test sentence never appears in training.
    Stratifying by domain stops a fold from, e.g., testing only on
    Healthcare while training on none of it.
    """
    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    return list(splitter.split(sentences, sentences["domain"]))


def score_predictions(predictions: list[str], references: list[str]) -> dict:
    """Corpus BLEU and chrF (sacrebleu). chrF works on characters, so it is
    more informative than BLEU for Sinhala and for very small test sets."""
    return {
        "bleu": sacrebleu.corpus_bleu(predictions, [references]).score,
        "chrf": sacrebleu.corpus_chrf(predictions, [references]).score,
    }


def summarize_folds(fold_scores: list[dict]) -> dict:
    """Mean and standard deviation over folds for every language and metric.

    Uses the sample standard deviation (ddof=1), the usual choice when the
    folds are a sample of possible splits. With one fold, std is reported as 0.
    """
    summary = {}
    for language in LANGUAGES:
        for metric in ["bleu", "chrf"]:
            values = np.array([fold[language][metric] for fold in fold_scores])
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary[f"{language}_{metric}"] = {"mean": float(values.mean()), "std": std}
    return summary


def print_summary(summary: dict, num_folds: int) -> None:
    print(f"\nResults over {num_folds} fold(s) (mean +- std):")
    for name, values in summary.items():
        print(f"  {name:<14}: {values['mean']:6.2f} +- {values['std']:5.2f}")


def create_run_dir(runs_dir: Path) -> Path:
    run_dir = runs_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, data: dict) -> None:
    # ensure_ascii=False writes Sinhala as readable text instead of \u escapes.
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_run(
    run_dir: Path,
    run_info: dict,
    fold_scores: list[dict],
    summary: dict,
    predictions: pd.DataFrame,
) -> None:
    """Write everything needed to reproduce and inspect a run."""
    shutil.copy(PROJECT_ROOT / "config.yaml", run_dir / "config.yaml")
    save_json(run_dir / "run_info.json", run_info)
    save_json(run_dir / "fold_scores.json", {"folds": fold_scores, "summary": summary})
    predictions.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8")
    print(f"\nSaved run to {run_dir}")
