"""Baseline 2: gloss text -> Sinhala / English with a fine-tuned NLLB model (no video).

One model learns both directions. The output language is chosen by the
language code the decoder starts with (sin_Sinh or eng_Latn), which is how
NLLB was trained.

Memory (6 GB GPU): only small LoRA adapter weights are trained; the 600M
base weights, including the large embedding table, stay frozen. Training
runs in fp16 autocast.

Usage (from the project root):
    python src/baseline_gloss_nllb.py              # all 5 folds
    python src/baseline_gloss_nllb.py --folds 0    # only fold 0
"""

import argparse
import random
import sys
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peft
import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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

ZWJ = "‍"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Text <-> token ids
# ---------------------------------------------------------------------------


def protect_zwj(text: str, placeholder: str) -> str:
    """NLLB's tokenizer turns ZWJ into a space (breaking Sinhala conjuncts), so
    we swap it for a placeholder character the tokenizer keeps."""
    return text.replace(ZWJ, placeholder)


def restore_zwj(text: str, placeholder: str) -> str:
    return unicodedata.normalize("NFC", text.replace(placeholder, ZWJ))


def build_examples(sentences: pd.DataFrame, nllb_settings: dict) -> list[dict]:
    """Two training examples per sentence: gloss -> Sinhala and gloss -> English."""
    placeholder = nllb_settings["zwj_placeholder"]
    examples = []
    for _, row in sentences.iterrows():
        source = " ".join(row["gloss_tokens"])
        for language in LANGUAGES:
            examples.append(
                {
                    "source": source,
                    "target_code": nllb_settings["target_langs"][language],
                    "target": protect_zwj(row[language], placeholder),
                }
            )
    return examples


def encode_example(tokenizer, example: dict, max_length: int) -> dict:
    """Token ids for one example.

    Source: [eng_Latn] tokens </s>   (the tokenizer adds these because src_lang is set)
    Labels: [target_code] tokens </s> (built by hand so each example can have its
    own target language; checked to match the tokenizer's own text_target output)
    """
    input_ids = tokenizer(example["source"], truncation=True, max_length=max_length)["input_ids"]
    target_ids = tokenizer(example["target"], add_special_tokens=False)["input_ids"]
    language_id = tokenizer.convert_tokens_to_ids(example["target_code"])
    labels = [language_id] + target_ids[: max_length - 2] + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "labels": labels}


def make_collate_fn(pad_token_id: int):
    """Pad a list of encoded examples into tensors of equal length."""

    def collate(batch: list[dict]) -> dict:
        max_input = max(len(item["input_ids"]) for item in batch)
        max_label = max(len(item["labels"]) for item in batch)
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            input_padding = max_input - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * input_padding)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * input_padding)
            # -100 tells the loss function to ignore padding positions.
            labels.append(item["labels"] + [-100] * (max_label - len(item["labels"])))
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    return collate


# ---------------------------------------------------------------------------
# Model, training, generation
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(nllb_settings: dict, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        nllb_settings["model_id"], src_lang=nllb_settings["source_lang"]
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(nllb_settings["model_id"])
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=nllb_settings["lora_r"],
        lora_alpha=nllb_settings["lora_alpha"],
        lora_dropout=nllb_settings["lora_dropout"],
        target_modules=nllb_settings["lora_target_modules"],
    )
    # get_peft_model freezes every original weight (embeddings included)
    # and adds small trainable LoRA matrices to the listed layers.
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model.to(device), tokenizer


def train_model(model, loader: DataLoader, nllb_settings: dict, device: torch.device) -> list[float]:
    """Train with fp16 autocast and gradient accumulation. Returns mean loss per epoch."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=nllb_settings["learning_rate"])
    use_fp16 = nllb_settings["fp16"] and device.type == "cuda"
    # GradScaler stops tiny fp16 gradients from rounding to zero.
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    accumulation = nllb_settings["grad_accumulation_steps"]

    epoch_losses = []
    model.train()
    for epoch in range(nllb_settings["epochs"]):
        batch_losses = []
        optimizer.zero_grad()
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                loss = model(**batch).loss
            # Divide so that the accumulated gradient is an average, not a sum.
            scaler.scale(loss / accumulation).backward()
            batch_losses.append(loss.item())

            is_last_batch = step == len(loader)
            if step % accumulation == 0 or is_last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        epoch_losses.append(float(np.mean(batch_losses)))
        print(f"  epoch {epoch + 1:2d}/{nllb_settings['epochs']}  loss {epoch_losses[-1]:.4f}")
    return epoch_losses


@torch.no_grad()
def translate(
    model, tokenizer, sources: list[str], target_code: str, nllb_settings: dict, device: torch.device
) -> list[str]:
    model.eval()
    batch = tokenizer(
        sources, return_tensors="pt", padding=True, truncation=True, max_length=nllb_settings["max_length"]
    ).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=nllb_settings["fp16"]):
        output_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # Force the first generated token to be the language code: this is
            # what selects Sinhala or English output.
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
            num_beams=nllb_settings["num_beams"],
            max_new_tokens=nllb_settings["max_length"],
        )
    texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [restore_zwj(text, nllb_settings["zwj_placeholder"]) for text in texts]


def save_loss_plot(epoch_losses: list[float], path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", markersize=3)
    axis.set_xlabel("epoch")
    axis.set_ylabel("training loss")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def run_fold(
    fold: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict,
    device: torch.device,
    run_dir: Path,
) -> tuple[dict, list[dict], dict]:
    nllb_settings = config["nllb"]
    # Same seed at the start of every fold, so each fold is reproducible on its own.
    set_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_model_and_tokenizer(nllb_settings, device)
    encoded = [encode_example(tokenizer, ex, nllb_settings["max_length"]) for ex in build_examples(train, nllb_settings)]
    loader = DataLoader(
        encoded,
        batch_size=nllb_settings["batch_size"],
        shuffle=True,
        num_workers=0,  # Windows: worker processes cause problems
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    print(f"Fold {fold}: {len(train)} train sentences ({len(encoded)} examples), {len(test)} test sentences")
    epoch_losses = train_model(model, loader, nllb_settings, device)
    save_loss_plot(epoch_losses, run_dir / f"loss_fold{fold}.png", f"NLLB + LoRA, fold {fold}")

    sources = [" ".join(tokens) for tokens in test["gloss_tokens"]]
    prediction_rows = []
    scores = {}
    for language in LANGUAGES:
        target_code = nllb_settings["target_langs"][language]
        predictions = translate(model, tokenizer, sources, target_code, nllb_settings, device)
        references = test[language].tolist()
        scores[language] = score_predictions(predictions, references)
        for (_, row), prediction in zip(test.iterrows(), predictions):
            prediction_rows.append(
                {
                    "fold": fold,
                    "sentence_id": row["id"],
                    "domain": row["domain"],
                    "gloss": row["gloss"],
                    "language": language,
                    "reference": row[language],
                    "prediction": prediction,
                }
            )

    fold_info = {"epoch_losses": epoch_losses}
    if device.type == "cuda":
        fold_info["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        fold_info["peak_vram_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        print(
            f"  peak VRAM: {fold_info['peak_vram_gib']:.2f} GiB allocated, "
            f"{fold_info['peak_vram_reserved_gib']:.2f} GiB reserved"
        )
    print(
        f"  si BLEU {scores['sinhala']['bleu']:5.1f} chrF {scores['sinhala']['chrf']:5.1f} | "
        f"en BLEU {scores['english']['bleu']:5.1f} chrF {scores['english']['chrf']:5.1f}"
    )

    # Free GPU memory before the next fold loads a fresh model.
    del model
    torch.cuda.empty_cache()
    return scores, prediction_rows, fold_info


def parse_arguments(num_folds: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gloss -> text baseline with NLLB + LoRA.")
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=list(range(num_folds)),
        help="which folds to run (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Sinhala on the Windows console
    config = load_config()
    args = parse_arguments(config["text_baselines"]["num_folds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sentences = load_sentences(resolve_path(config["paths"]["sentences_csv"]))
    folds = make_folds(sentences, config["text_baselines"]["num_folds"], config["seed"])
    run_dir = create_run_dir(resolve_path(config["paths"]["runs_dir"]))

    fold_scores, all_predictions, fold_infos = [], [], {}
    for fold in args.folds:
        train_index, test_index = folds[fold]
        scores, prediction_rows, fold_info = run_fold(
            fold, sentences.iloc[train_index], sentences.iloc[test_index], config, device, run_dir
        )
        fold_scores.append(scores)
        all_predictions.extend(prediction_rows)
        fold_infos[fold] = fold_info

    summary = summarize_folds(fold_scores)
    print_summary(summary, len(fold_scores))
    run_info = {
        "baseline": "gloss_nllb_lora",
        "seed": config["seed"],
        "folds_run": args.folds,
        "fold_info": fold_infos,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
    }
    save_run(run_dir, run_info, fold_scores, summary, pd.DataFrame(all_predictions))


if __name__ == "__main__":
    main()
