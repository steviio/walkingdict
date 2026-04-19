"""Parse Urban Dictionary CSV from Kaggle into the unified schema.

Dataset: https://www.kaggle.com/datasets/therohk/urban-dictionary-words-dataset
File:    urbandict-word-defs.csv  →  data/raw/urban_dict/

Columns: word, up_votes, down_votes, author, definition, example

Quality filter: keep entries where upvote ratio > 0.7 AND up_votes >= 50.

Run:
    python -m ingestion.etl_urban_dict [--limit 5000]
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_URBAN_DICT_DIR, PROCESSED_DIR


MIN_UP_VOTES = 50
MIN_UPVOTE_RATIO = 0.70


def _clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)  # [linked word] → linked word
    return text.strip()


def _difficulty(definition: str) -> str:
    words = definition.split()
    if len(words) < 15:
        return "beginner"
    if len(words) < 40:
        return "intermediate"
    return "advanced"


def parse_csv(path: Path, limit: int | None = None) -> list[dict]:
    df = pd.read_csv(path, on_bad_lines="skip")

    # Normalise column names
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"word", "up_votes", "down_votes", "definition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Got: {list(df.columns)}")

    # Quality filter
    df = df.dropna(subset=["word", "definition"])
    df["up_votes"] = pd.to_numeric(df["up_votes"], errors="coerce").fillna(0)
    df["down_votes"] = pd.to_numeric(df["down_votes"], errors="coerce").fillna(0)
    df["total"] = df["up_votes"] + df["down_votes"]
    df = df[df["total"] > 0]
    df["ratio"] = df["up_votes"] / df["total"]
    df = df[(df["up_votes"] >= MIN_UP_VOTES) & (df["ratio"] >= MIN_UPVOTE_RATIO)]

    # Sort best-first so we keep the highest-quality entry per word
    df = df.sort_values("up_votes", ascending=False)
    df = df.drop_duplicates(subset=["word"], keep="first")

    if limit:
        df = df.head(limit)

    docs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Urban Dict"):
        word = _clean(str(row["word"]))
        definition = _clean(str(row["definition"]))
        example_raw = _clean(str(row.get("example", "") or ""))
        if not word or not definition:
            continue

        docs.append({
            "word": word,
            "source": "urban_dictionary",
            "source_file": path.name,
            "category": "slang",
            "difficulty": _difficulty(definition),
            "etymology": "",
            "related_words": [],
            "last_updated": "2024-01-01",
            "senses": [{
                "part_of_speech": "unknown",
                "definition": definition,
                "examples": [example_raw] if example_raw else [],
            }],
        })

    return docs


def run(limit: int | None = None) -> list[dict]:
    candidates = list(RAW_URBAN_DICT_DIR.glob("*.csv"))
    if not candidates:
        print(f"No .csv files found in {RAW_URBAN_DICT_DIR}")
        return []

    src = candidates[0]
    print(f"Parsing {src.name} …")
    docs = parse_csv(src, limit=limit)
    print(f"  → {len(docs):,} documents after quality filter")
    return docs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    docs = run(limit=args.limit)

    out = PROCESSED_DIR / "urban_dict_entries.jsonl"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Written → {out}")
