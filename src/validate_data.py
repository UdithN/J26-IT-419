"""Check the dataset, write data/manifest.csv and print a report.

Usage (from the project root):
    python src/validate_data.py
"""

import sys

import pandas as pd

from data import (
    build_manifest,
    get_video_info,
    load_config,
    load_sentences,
    resolve_path,
    save_manifest,
)

ZWJ = "‍"  # zero-width joiner, needed by Sinhala conjuncts such as ප්‍ර


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def report_row_count(sentences: pd.DataFrame) -> None:
    print_header("Rows")
    print(f"Sentences: {len(sentences)}")


def report_missing_cells(sentences: pd.DataFrame) -> None:
    print_header("Missing / empty cells")
    found_any = False
    for column in ["domain", "gloss", "sinhala", "english"]:
        empty_ids = sentences.loc[sentences[column] == "", "id"].tolist()
        if empty_ids:
            found_any = True
            print(f"{column}: empty in IDs {empty_ids}")

    # The loader turns a missing or non-numeric ID into <NA>.
    bad_id_rows = sentences.index[sentences["id"].isna()].tolist()
    if bad_id_rows:
        found_any = True
        print(f"id: missing or not a number in CSV rows {[row + 2 for row in bad_id_rows]}")

    # A gloss cell with text but no [TOKEN] in it means a formatting mistake.
    no_tokens = sentences.loc[
        (sentences["gloss"] != "") & (sentences["gloss_tokens"].map(len) == 0), "id"
    ].tolist()
    if no_tokens:
        found_any = True
        print(f"gloss: no [TOKEN] found in IDs {no_tokens}")

    if not found_any:
        print("None")


def report_duplicates(sentences: pd.DataFrame) -> None:
    print_header("Duplicates")
    found_any = False
    for column in ["id", "gloss", "sinhala", "english"]:
        # keep=False marks every copy, not just the second one.
        duplicated = sentences[sentences[column].duplicated(keep=False)]
        if not duplicated.empty:
            found_any = True
            for value, group in duplicated.groupby(column):
                print(f"{column} repeated in IDs {group['id'].tolist()}: {value}")
    if not found_any:
        print("None")


def report_gloss_vocabulary(sentences: pd.DataFrame) -> None:
    print_header("Gloss vocabulary")
    all_tokens = [token for tokens in sentences["gloss_tokens"] for token in tokens]
    print(f"Total gloss tokens: {len(all_tokens)}")
    print(f"Vocabulary size (unique tokens): {len(set(all_tokens))}")


def report_domains(sentences: pd.DataFrame) -> None:
    print_header("Sentences per domain")
    for domain, count in sentences["domain"].value_counts().items():
        print(f"{count:3d}  {domain}")


def report_video_coverage(sentences: pd.DataFrame, manifest: pd.DataFrame) -> None:
    print_header("Videos found vs expected")
    sentence_ids = set(sentences["id"].dropna().astype(int))
    video_ids = set(manifest["sentence_id"])

    # We expect at least one video (take) per sentence.
    print(f"Video files found: {len(manifest)}")
    print(f"Sentences with at least one video: {len(sentence_ids & video_ids)} / {len(sentence_ids)}")
    print(f"Sentences with no video: {sorted(sentence_ids - video_ids)}")

    takes_per_sentence = manifest.groupby("sentence_id").size()
    multi_take = takes_per_sentence[takes_per_sentence > 1]
    print(f"Sentences with more than one take: {multi_take.to_dict()}")
    print(f"Signers: {manifest['signer'].value_counts().to_dict()}")

    print_header("Videos with no matching sentence")
    orphans = manifest[~manifest["sentence_id"].isin(sentence_ids)]
    if orphans.empty:
        print("None")
    else:
        for path in orphans["video_path"]:
            print(path)


def report_video_details(manifest: pd.DataFrame) -> None:
    print_header("Video details")
    print(f"{'video':<14}{'fps':>8}{'frames':>8}{'seconds':>9}")
    for relative_path in manifest["video_path"]:
        video_path = resolve_path(relative_path)
        try:
            info = get_video_info(video_path)
        except IOError as error:
            print(f"{video_path.name:<14} ERROR: {error}")
            continue
        print(
            f"{video_path.name:<14}{info['fps']:>8.2f}"
            f"{info['frame_count']:>8d}{info['duration']:>9.2f}"
        )


def report_sinhala_samples(sentences: pd.DataFrame, sample_count: int = 3) -> None:
    print_header("Sinhala samples (check they display correctly)")
    for _, row in sentences.head(sample_count).iterrows():
        print(f"{row['id']}: {row['sinhala']}  |  {row['english']}")

    # The console font may not show conjuncts properly even when the data is
    # fine, so this count shows the ZWJ characters survived loading.
    zwj_count = sentences["sinhala"].str.contains(ZWJ).sum()
    print(f"Sinhala sentences containing ZWJ (U+200D): {zwj_count}")


def main() -> None:
    # Windows may print with a non-Unicode code page (e.g. when output is
    # redirected), which crashes on Sinhala. Force UTF-8 output.
    sys.stdout.reconfigure(encoding="utf-8")

    config = load_config()
    sentences = load_sentences(resolve_path(config["paths"]["sentences_csv"]))
    manifest = build_manifest(
        resolve_path(config["paths"]["videos_dir"]),
        default_signer=config["data"]["default_signer"],
    )
    manifest_path = resolve_path(config["paths"]["manifest_csv"])
    save_manifest(manifest, manifest_path)
    print(f"Wrote {len(manifest)} rows to {manifest_path}")

    report_row_count(sentences)
    report_missing_cells(sentences)
    report_duplicates(sentences)
    report_gloss_vocabulary(sentences)
    report_domains(sentences)
    report_video_coverage(sentences, manifest)
    report_video_details(manifest)
    report_sinhala_samples(sentences)


if __name__ == "__main__":
    main()
