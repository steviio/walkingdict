"""Parse Wiktionary JSONL from kaikki.org into the unified schema.

Download:
    wget https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl.gz
    gunzip kaikki.org-dictionary-English.jsonl.gz
    mv kaikki.org-dictionary-English.jsonl data/raw/wiktionary/

Run:
    python -m ingestion.etl_wiktionary [--limit 5000]
"""

import json
import re
import sys
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_WIKTIONARY_DIR, PROCESSED_DIR

# ── Unified schema ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Strip wikitext markup and normalise whitespace."""
    text = re.sub(r"\{\{[^}]*\}\}", "", text)   # {{templates}}
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)  # [[link|text]]
    text = re.sub(r"'''?", "", text)              # bold/italic
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _difficulty(pos: str, category: str) -> str:
    if category in ("idiom", "phrase", "proverb"):
        return "advanced"
    if pos in ("verb", "adjective"):
        return "intermediate"
    return "beginner"


_SKIP_IPA_TAGS = {"archaic", "dialectal", "obsolete"}
_KEEP_FORM_TAGS = {
    "comparative", "superlative", "plural",
    "past", "past-participle", "present-participle", "third-person-singular",
}


def _extract_ipa(sounds: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for sound in sounds:
        ipa = sound.get("ipa", "")
        if not ipa:
            continue
        tags = sound.get("tags", [])
        if any(t in _SKIP_IPA_TAGS for t in tags):
            continue
        if ipa not in seen:
            seen.add(ipa)
            result.append({"tags": tags, "ipa": ipa})
    return result[:4]


def _extract_forms(forms: list[dict]) -> list[str]:
    seen: set[str] = set()
    result = []
    for form in forms:
        form_str = form.get("form", "").strip()
        tags = set(form.get("tags", []))
        if form_str and form_str not in seen and tags & _KEEP_FORM_TAGS:
            seen.add(form_str)
            result.append(form_str)
    return result[:6]


def _extract_relwords(items: list) -> list[str]:
    """Extract clean word strings from synonyms/antonyms lists, skipping noise entries."""
    seen: set[str] = set()
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # Entries with only tags (no sense) are metadata artifacts, not real words
        if "tags" in item and "sense" not in item:
            continue
        word = item.get("word", "").strip()
        # Skip descriptive phrases (contain " of " or are multi-word noise)
        if not word or " of " in word or len(word.split()) > 3:
            continue
        if word not in seen:
            seen.add(word)
            result.append(word)
    return result[:8]


def parse_entry(raw: dict, source_file: str = "") -> dict | None:
    """Return a single word entry with a senses list for one POS block, or None."""
    word = raw.get("word", "").strip()
    if not word:
        return None

    pos = raw.get("pos", "").lower()
    etymology = _clean_text(raw.get("etymology_text", "") or "")
    ipa = _extract_ipa(raw.get("sounds", []))
    forms = _extract_forms(raw.get("forms", []))
    synonyms = _extract_relwords(raw.get("synonyms", []))
    antonyms = _extract_relwords(raw.get("antonyms", []))
    category = "formal"

    all_tags: list[str] = []
    for sense in raw.get("senses", []):
        all_tags.extend(sense.get("tags", []))
    if any(t in all_tags for t in ("slang", "colloquial", "informal", "internet-slang")):
        category = "slang"
    elif pos in ("phrase",):
        category = "idiom"


    senses: list[dict] = []
    related_set: list[str] = []
    for sense in raw.get("senses", []):
        gloss = _clean_text(sense.get("glosses", [""])[0] or "")
        if not gloss:
            continue

        raw_examples = sense.get("examples", []) or []
        examples = [_clean_text(e.get("text", "") or "") for e in raw_examples if e.get("text")]

        related = [
            link["word"]
            for link in (sense.get("links", []) or [])
            if isinstance(link, dict) and link.get("word")
        ]
        related_set.extend(related)

        senses.append({
            "part_of_speech": pos,
            "definition": gloss,
            "examples": examples[:3],
        })

    if not senses:
        return None

    seen_defs: set[str] = set()
    senses = [s for s in senses if s["definition"] not in seen_defs and not seen_defs.add(s["definition"])]

    seen: set[str] = set()
    related_words: list[str] = []
    for w in related_set:
        if w not in seen:
            seen.add(w)
            related_words.append(w)

    return {
        "word": word,
        "source": "wiktionary",
        "source_file": source_file,
        "category": category,
        "difficulty": _difficulty(pos, category),
        "ipa": ipa,
        "forms": forms,
        "senses": senses,
        "etymology": etymology,
        "synonyms": synonyms,
        "antonyms": antonyms,
        "related_words": related_words[:10],
        "last_updated": "2025-01-01",
    }


def iter_raw(jsonl_path: Path) -> Iterator[dict]:
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def run(limit: int | None = None) -> list[dict]:
    candidates = list(RAW_WIKTIONARY_DIR.glob("*.jsonl"))
    if not candidates:
        print(f"No .jsonl files found in {RAW_WIKTIONARY_DIR}")
        return []

    src = candidates[0]
    print(f"Parsing {src.name} …")

    # Parse all raw entries (one per word+POS block)
    raw_entries: list[dict] = []
    for i, raw in enumerate(tqdm(iter_raw(src))):
        if limit and i >= limit:
            break
        entry = parse_entry(raw, source_file=src.name)
        if entry:
            raw_entries.append(entry)

    # Group by word — merge senses and related_words across POS blocks
    grouped: dict[str, dict] = {}
    for entry in raw_entries:
        key = entry["word"].lower()
        if key not in grouped:
            grouped[key] = entry
        else:
            existing = grouped[key]
            existing["senses"].extend(entry["senses"])
            seen_defs: set[str] = set()
            existing["senses"] = [s for s in existing["senses"] if s["definition"] not in seen_defs and not seen_defs.add(s["definition"])]
            for field, dedup_key in (
                ("related_words", None),
                ("synonyms", None),
                ("antonyms", None),
            ):
                seen_set = set(existing[field])
                for w in entry[field]:
                    if w not in seen_set:
                        existing[field].append(w)
                        seen_set.add(w)
            # Merge IPA — keep unique pronunciations
            seen_ipa = {i["ipa"] for i in existing["ipa"]}
            for item in entry["ipa"]:
                if item["ipa"] not in seen_ipa:
                    existing["ipa"].append(item)
                    seen_ipa.add(item["ipa"])
            # Merge forms — keep unique
            seen_forms = set(existing["forms"])
            for f in entry["forms"]:
                if f not in seen_forms:
                    existing["forms"].append(f)
                    seen_forms.add(f)
            if len(entry["etymology"]) > len(existing["etymology"]):
                existing["etymology"] = entry["etymology"]
            if entry["category"] in ("slang", "idiom") and existing["category"] == "formal":
                existing["category"] = entry["category"]

    documents = list(grouped.values())
    print(f"  → {len(documents):,} word entries ({len(raw_entries):,} POS blocks merged)")
    return documents


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    docs = run(limit=args.limit)

    out = PROCESSED_DIR / "wiktionary_entries.jsonl"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Written → {out}")
