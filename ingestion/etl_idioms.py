"""Parse idiom corpora into the unified schema.

Supported formats
─────────────────
1. LIdioms / MAGPIE / EPIE  — any JSON(L) with an "idiom" / "phrase" field
2. Simple TSV / CSV: columns  phrase, definition[, example]

Place files in data/raw/idioms/

Run:
    python -m ingestion.etl_idioms [--limit 2000]
"""

import csv
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_IDIOMS_DIR, PROCESSED_DIR


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _make_doc(phrase: str, definition: str, examples: list[str], source_file: str = "") -> dict:
    return {
        "word": phrase,
        "source": "idioms",
        "source_file": source_file,
        "category": "idiom",
        "difficulty": "advanced",
        "etymology": "",
        "related_words": [],
        "last_updated": "2024-01-01",
        "senses": [{
            "part_of_speech": "phrase",
            "definition": definition,
            "examples": examples[:3],
        }],
    }


# ── Parsers ────────────────────────────────────────────────────────────────────

def _parse_jsonl(path: Path) -> list[dict]:
    docs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            phrase = _clean(
                obj.get("idiom") or obj.get("phrase") or obj.get("expression") or ""
            )
            definition = _clean(
                obj.get("definition") or obj.get("meaning") or obj.get("gloss") or ""
            )
            example_raw = obj.get("example") or obj.get("sentence") or ""
            examples = [_clean(example_raw)] if example_raw else []

            if phrase and definition:
                docs.append(_make_doc(phrase, definition, examples, source_file=path.name))
    return docs


def _parse_json_list(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        data = list(data.values()) if isinstance(data, dict) else []
    docs = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        phrase = _clean(
            obj.get("idiom") or obj.get("phrase") or obj.get("expression") or ""
        )
        definition = _clean(
            obj.get("definition") or obj.get("meaning") or obj.get("gloss") or ""
        )
        example_raw = obj.get("example") or obj.get("sentence") or ""
        examples = [_clean(example_raw)] if example_raw else []
        if phrase and definition:
            docs.append(_make_doc(phrase, definition, examples, source_file=path.name))
    return docs


def _parse_csv(path: Path) -> list[dict]:
    docs = []
    with path.open(encoding="utf-8", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|")
        reader = csv.DictReader(fh, dialect=dialect)
        for row in reader:
            lower_row = {k.lower().strip(): v for k, v in row.items()}
            phrase = _clean(
                lower_row.get("phrase") or lower_row.get("idiom")
                or lower_row.get("expression") or ""
            )
            definition = _clean(
                lower_row.get("definition") or lower_row.get("meaning")
                or lower_row.get("gloss") or ""
            )
            example_raw = (
                lower_row.get("example") or lower_row.get("sentence") or ""
            )
            examples = [_clean(example_raw)] if example_raw else []
            if phrase and definition:
                docs.append(_make_doc(phrase, definition, examples, source_file=path.name))
    return docs


def _dispatch(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _parse_jsonl(path)
    if suffix == ".json":
        return _parse_json_list(path)
    if suffix in (".csv", ".tsv"):
        return _parse_csv(path)
    return []


def run(limit: int | None = None) -> list[dict]:
    files = [
        f for f in RAW_IDIOMS_DIR.iterdir()
        if f.suffix.lower() in (".jsonl", ".json", ".csv", ".tsv")
    ]
    if not files:
        print(f"No supported files found in {RAW_IDIOMS_DIR}")
        return []

    all_docs: list[dict] = []
    for f in files:
        print(f"Parsing {f.name} …")
        docs = _dispatch(f)
        print(f"  → {len(docs):,} documents")
        all_docs.extend(docs)

    if limit:
        all_docs = all_docs[:limit]
    print(f"Total idiom documents: {len(all_docs):,}")
    return all_docs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    docs = run(limit=args.limit)

    out = PROCESSED_DIR / "idiom_entries.jsonl"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Written → {out}")
