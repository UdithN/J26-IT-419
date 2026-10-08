"""Baseline 1: translation by retrieval (no learning, no video).

For each test gloss, find the training sentence whose gloss tokens overlap
most (Jaccard), and output that sentence's Sinhala and English text.
Ties are broken by how similar the token ORDER is.

Any learned model should beat this; if it does not, it is only memorising.

Usage (from the project root):
    python src/baseline_retrieval.py
"""

import sys
from difflib import SequenceMatcher

import pandas as pd

from data import load_config, load_sentences, resolve_path
from evaluation import (
    LANGUAGES,
    create_run_dir,
    make_folds,
    print_summary,
    save_run,
    score_predictions,
    summarize_folds,
)


def jaccard(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Shared unique tokens divided by all unique tokens (0 = none, 1 = same set)."""
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def order_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """How similar the token sequences are, order included (0..1).

    SequenceMatcher counts tokens that appear in the same relative order,
    so [I, HOME, GO] vs [I, GO, HOME] scores lower than an exact match.
    """
    return SequenceMatcher(None, tokens_a, tokens_b).ratio()


def retrieve_best_match(query_tokens: list[str], train_sentences: pd.DataFrame) -> pd.Series:
    """Return the training row with the highest (Jaccard, order similarity).

    If both scores tie, the earliest training sentence wins, so results are
    deterministic.
    """
    best_row = None
    best_score = (-1.0, -1.0)
    for _, row in train_sentences.iterrows():
        score = (
            jaccard(query_tokens, row["gloss_tokens"]),
            order_similarity(query_tokens, row["gloss_tokens"]),
        )
        if score > best_score:  # tuples compare Jaccard first, then order
            best_score = score
            best_row = row
    return best_row


def run_fold(fold: int, train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, list[dict]]:
    prediction_rows = []
    for _, test_row in test.iterrows():
        match = retrieve_best_match(test_row["gloss_tokens"], train)
        for language in LANGUAGES:
            prediction_rows.append(
                {
                    "fold": fold,
                    "sentence_id": test_row["id"],
                    "domain": test_row["domain"],
                    "gloss": test_row["gloss"],
                    "language": language,
                    "reference": test_row[language],
                    "prediction": match[language],
                    "retrieved_id": match["id"],
                    "retrieved_gloss": match["gloss"],
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    scores = {}
    for language in LANGUAGES:
        rows = predictions[predictions["language"] == language]
        scores[language] = score_predictions(rows["prediction"].tolist(), rows["reference"].tolist())
    return scores, prediction_rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Sinhala on the Windows console
    config = load_config()
    sentences = load_sentences(resolve_path(config["paths"]["sentences_csv"]))
    folds = make_folds(sentences, config["text_baselines"]["num_folds"], config["seed"])

    fold_scores = []
    all_predictions = []
    for fold, (train_index, test_index) in enumerate(folds):
        scores, prediction_rows = run_fold(fold, sentences.iloc[train_index], sentences.iloc[test_index])
        fold_scores.append(scores)
        all_predictions.extend(prediction_rows)
        print(
            f"Fold {fold}: train {len(train_index)}, test {len(test_index)} | "
            f"si BLEU {scores['sinhala']['bleu']:5.1f} chrF {scores['sinhala']['chrf']:5.1f} | "
            f"en BLEU {scores['english']['bleu']:5.1f} chrF {scores['english']['chrf']:5.1f}"
        )

    summary = summarize_folds(fold_scores)
    print_summary(summary, len(folds))

    predictions = pd.DataFrame(all_predictions)
    print("\nExamples (fold 0):")
    for _, row in predictions[predictions["fold"] == 0].head(6).iterrows():
        print(f"  [{row['language']}] gloss {row['gloss']}  ->  retrieved {row['retrieved_gloss']}")
        print(f"      reference : {row['reference']}")
        print(f"      prediction: {row['prediction']}")

    run_dir = create_run_dir(resolve_path(config["paths"]["runs_dir"]))
    run_info = {"baseline": "retrieval", "seed": config["seed"], "folds_run": list(range(len(folds)))}
    save_run(run_dir, run_info, fold_scores, summary, predictions)


if __name__ == "__main__":
    main()
