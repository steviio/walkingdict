"""Merge ETL outputs into a single unified_corpus.jsonl.

Each source ETL script writes its own intermediate JSONL.  This script
reads them all, deduplicates (same word + source), and writes the final
corpus.

Run:
    python -m ingestion.unify_schema
"""

import hashlib
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, CORPUS_FILE

SOURCE_FILES = [
    "wiktionary_entries.jsonl",
    "urban_dict_entries.jsonl",
    "wordnet_entries.jsonl",
    "idiom_entries.jsonl",
]

REQUIRED_FIELDS = {
    "word", "senses", "source", "category",
    "difficulty", "related_words", "etymology", "last_updated",
}


def _doc_id(doc: dict) -> str:
    key = f"{doc['word'].lower()}|{doc['source']}"
    return hashlib.md5(key.encode()).hexdigest()


def _validate(doc: dict) -> dict | None:
    """Return cleaned doc or None if too broken to keep."""
    missing = REQUIRED_FIELDS - set(doc.keys())
    if missing:
        return None

    word = doc["word"].strip()
    senses = [
        s for s in (doc.get("senses") or [])
        if isinstance(s, dict) and s.get("definition", "").strip()
    ]
    if not word or not senses:
        return None

    doc["word"] = word
    doc["senses"] = senses
    doc["related_words"] = [str(r).strip() for r in (doc.get("related_words") or []) if str(r).strip()]
    return doc


def run() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    total_written = 0

    with CORPUS_FILE.open("w", encoding="utf-8") as out:
        for fname in SOURCE_FILES:
            src = PROCESSED_DIR / fname
            if not src.exists():
                print(f"  [skip] {fname} not found")
                continue

            count = written = 0
            with src.open(encoding="utf-8") as fh:
                for line in tqdm(fh, desc=fname):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    count += 1
                    doc = _validate(doc)
                    if doc is None:
                        continue

                    doc_id = _doc_id(doc)
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)

                    out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    written += 1

            print(f"  {fname}: {count:,} in → {written:,} written (dedup/filtered)")
            total_written += written

    print(f"\nCorpus: {total_written:,} documents → {CORPUS_FILE}")
    return total_written


if __name__ == "__main__":
    run()
