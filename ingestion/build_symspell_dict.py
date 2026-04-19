"""Build SymSpell frequency dictionary from the unified corpus.

SymSpell uses a word-frequency list to rank spelling correction candidates.
Only single-word terms are included — multi-word phrases break SymSpell's
single-token assumption.

Output
──────
  data/processed/symspell_dict.txt  — "word frequency\\n" plain text

Run:
    python -m ingestion.build_symspell_dict
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CORPUS_FILE, SYMSPELL_DICT_FILE, PROCESSED_DIR


def build_symspell_dict(corpus_path: Path) -> dict[str, int]:
    """Return single-word → frequency map from corpus (multi-word phrases excluded)."""
    freq: dict[str, int] = defaultdict(int)

    with corpus_path.open(encoding="utf-8") as fh:
        for line in tqdm(fh, desc="Building SymSpell dict"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            word = doc["word"].lower().strip()
            if word and " " not in word:
                freq[word] += 1

    return dict(freq)


def run() -> None:
    if not CORPUS_FILE.exists():
        print(f"Corpus not found: {CORPUS_FILE}")
        print("Run ingestion/unify_schema.py first.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Building SymSpell frequency dictionary …")
    freq = build_symspell_dict(CORPUS_FILE)
    print(f"  {len(freq):,} single-word entries")

    with SYMSPELL_DICT_FILE.open("w", encoding="utf-8") as fh:
        for word, count in freq.items():
            fh.write(f"{word} {count}\n")
    print(f"  SymSpell dict saved → {SYMSPELL_DICT_FILE}")


if __name__ == "__main__":
    run()
