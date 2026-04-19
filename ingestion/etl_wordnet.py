"""Parse Open English WordNet (2025 edition) into the unified schema.

Auto-download:
    On first run the gzipped LMF-XML release is fetched from GitHub,
    unpacked into data/raw/wordnet/, and parsed. Override the URL via
    the WORDNET_URL env var; use --skip-download to fail loudly when
    no local file is present (for air-gapped runs).

Supports:
  - english-wordnet-*.jsonld  (JSON-LD format)
  - english-wordnet-*.xml     (LMF-XML format)

Data: Open English WordNet, CC BY 4.0 — https://github.com/globalwordnet/english-wordnet

Run:
    python -m ingestion.etl_wordnet [--limit 5000] [--skip-download]
"""

import gzip
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_WORDNET_DIR, PROCESSED_DIR


WORDNET_URL = os.environ.get(
    "WORDNET_URL",
    "https://github.com/globalwordnet/english-wordnet/releases/download/2025-edition/english-wordnet-2025.xml.gz",
)


def _download_if_missing() -> None:
    """Fetch the pinned WordNet release into RAW_WORDNET_DIR if no raw file exists."""
    RAW_WORDNET_DIR.mkdir(parents=True, exist_ok=True)
    if any(RAW_WORDNET_DIR.glob("*.jsonld")) or any(RAW_WORDNET_DIR.glob("*.xml")):
        return

    gz_name = WORDNET_URL.rsplit("/", 1)[-1]
    gz_path = RAW_WORDNET_DIR / gz_name
    xml_path = RAW_WORDNET_DIR / gz_name.removesuffix(".gz")

    print(f"Downloading {WORDNET_URL} → {gz_path}")
    with requests.get(WORDNET_URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) or None
        with gz_path.open("wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc="WordNet"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))

    print(f"Decompressing → {xml_path}")
    with gzip.open(gz_path, "rb") as src, xml_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    gz_path.unlink()


def _difficulty(pos: str) -> str:
    mapping = {"n": "beginner", "v": "intermediate", "a": "intermediate", "r": "advanced"}
    return mapping.get(pos, "intermediate")


def _pos_name(pos: str) -> str:
    mapping = {"n": "noun", "v": "verb", "a": "adjective", "r": "adverb", "s": "adjective satellite"}
    return mapping.get(pos, pos)


def _group_by_word(raw_entries: list[dict]) -> list[dict]:
    """Merge entries with the same word, combining their senses."""
    grouped: dict[str, dict] = {}
    for entry in raw_entries:
        key = entry["word"].lower()
        if key not in grouped:
            grouped[key] = entry
        else:
            grouped[key]["senses"].extend(entry["senses"])
            seen_defs: set[str] = set()
            grouped[key]["senses"] = [s for s in grouped[key]["senses"] if s["definition"] not in seen_defs and not seen_defs.add(s["definition"])]
    return list(grouped.values())


# ── JSON-LD parser ─────────────────────────────────────────────────────────────

def _iter_jsonld(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    entries = data if isinstance(data, list) else data.get("@graph", [])
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        yield entry


def _parse_jsonld(path: Path) -> list[dict]:
    raw_entries: list[dict] = []

    for entry in tqdm(_iter_jsonld(path), desc="WordNet JSON-LD"):
        lemma = entry.get("lemma", {})
        if isinstance(lemma, dict):
            word = lemma.get("writtenForm", "")
            pos = lemma.get("partOfSpeech", "n")
        else:
            word = str(lemma)
            pos = "n"
        if not word:
            continue

        senses: list[dict] = []
        related: list[str] = []
        for sense in entry.get("sense", []):
            definition = sense.get("definition", "") or ""
            if isinstance(definition, list):
                definition = " ".join(definition)
            definition = definition.strip()
            if not definition:
                continue

            examples = sense.get("example", [])
            if isinstance(examples, str):
                examples = [examples]
            examples = [e.strip() for e in examples if e.strip()][:3]

            for rel in sense.get("SenseRelation", []):
                if isinstance(rel, dict):
                    target = rel.get("target", "")
                    if target:
                        related.append(target.split("-")[0])

            senses.append({
                "part_of_speech": _pos_name(pos),
                "definition": definition,
                "examples": examples,
            })

        if not senses:
            continue

        raw_entries.append({
            "word": word,
            "source": "wordnet",
            "source_file": path.name,
            "category": "formal",
            "difficulty": _difficulty(pos),
            "etymology": "",
            "related_words": list(dict.fromkeys(related))[:10],
            "last_updated": "2025-01-01",
            "senses": senses,
        })

    return _group_by_word(raw_entries)


# ── LMF-XML parser ─────────────────────────────────────────────────────────────

def _parse_xml(path: Path) -> list[dict]:
    raw_entries: list[dict] = []
    tree = ET.parse(path)
    root = tree.getroot()

    # Build synset id → definition/examples map
    synset_defs: dict[str, str] = {}
    synset_examples: dict[str, list[str]] = {}
    for synset in root.iter("Synset"):
        sid = synset.get("id", "")
        defn = ""
        examples = []
        for child in synset:
            if child.tag == "Definition":
                defn = (child.text or "").strip()
            elif child.tag == "Example":
                examples.append((child.text or "").strip())
        if sid and defn:
            synset_defs[sid] = defn
            synset_examples[sid] = examples[:3]

    for le in tqdm(root.iter("LexicalEntry"), desc="WordNet XML"):
        lemma_el = le.find("Lemma")
        if lemma_el is None:
            continue
        word = lemma_el.get("writtenForm", "")
        pos = lemma_el.get("partOfSpeech", "n")
        if not word:
            continue

        senses: list[dict] = []
        for sense in le.iter("Sense"):
            synset_id = sense.get("synset", "")
            definition = synset_defs.get(synset_id, "")
            if not definition:
                continue
            senses.append({
                "part_of_speech": _pos_name(pos),
                "definition": definition,
                "examples": synset_examples.get(synset_id, []),
            })

        if not senses:
            continue

        raw_entries.append({
            "word": word,
            "source": "wordnet",
            "source_file": path.name,
            "category": "formal",
            "difficulty": _difficulty(pos),
            "etymology": "",
            "related_words": [],
            "last_updated": "2025-01-01",
            "senses": senses,
        })

    return _group_by_word(raw_entries)


def run(limit: int | None = None, skip_download: bool = False) -> list[dict]:
    if not skip_download:
        _download_if_missing()

    jsonld_files = list(RAW_WORDNET_DIR.glob("*.jsonld"))
    xml_files = list(RAW_WORDNET_DIR.glob("*.xml"))

    if jsonld_files:
        docs = _parse_jsonld(jsonld_files[0])
    elif xml_files:
        docs = _parse_xml(xml_files[0])
    else:
        print(f"No .jsonld or .xml files found in {RAW_WORDNET_DIR}")
        return []

    if limit:
        docs = docs[:limit]
    print(f"  → {len(docs):,} WordNet word entries extracted")
    return docs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not fetch the WordNet release; fail if no local raw file exists.",
    )
    args = ap.parse_args()

    docs = run(limit=args.limit, skip_download=args.skip_download)

    out = PROCESSED_DIR / "wordnet_entries.jsonl"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Written → {out}")
