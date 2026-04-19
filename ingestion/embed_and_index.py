"""Embed the unified corpus and store in ChromaDB.

Uses Ollama's nomic-embed-text to generate embeddings locally.
Documents are embedded in batches to avoid memory issues.

Run:
    python -m ingestion.embed_and_index [--batch-size 64] [--reset]
"""

import argparse
import json
import logging
import urllib.request
import urllib.error
import sys
from pathlib import Path

import chromadb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CORPUS_FILE,
    CHROMA_DIR,
    CHROMA_COLLECTION,
    OLLAMA_BASE_URL,
    EMBED_MODEL,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


def _get_embedding(texts: list[str], base_url: str, model: str) -> list[list[float]]:
    """Call Ollama embed endpoint for a batch of texts."""

    payload = json.dumps({"model": model, "input": texts}).encode()
    payload_size_mb = len(payload) / (1024 * 1024)
    
    req = urllib.request.Request(
        f"{base_url}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["embeddings"]
    except Exception as e:
        logger.error(f"Exception calling Ollama: {type(e).__name__}: {e}", exc_info=True)
        logger.error(f"Payload size: {payload_size_mb:.2f} MB")
        logger.error(f"Number of texts: {len(texts)}")
        logger.error(f"Text lengths: min={min(len(t) for t in texts) if texts else 0}, max={max(len(t) for t in texts) if texts else 0}")
        import sys
        sys.stderr.flush()
        raise

# reduced limit for JSON payload safety; 
# nomic-embed-text has 8192 token context
_EMBED_MAX_CHARS = 8000


def _sense_text(sense: dict) -> str:
    pos = sense.get("part_of_speech", "")
    defn = sense.get("definition", "")
    entry = f"{pos}: {defn}" if pos else defn
    examples = sense.get("examples", [])
    sense_str = f"{entry} | {examples[0]}" if examples else entry
    
    # Truncate individual sense if it exceeds half the max to leave room for word + other senses
    max_sense_len = _EMBED_MAX_CHARS // 2
    if len(sense_str) > max_sense_len:
        sense_str = sense_str[:max_sense_len] + "…"
    
    return sense_str


def _split_doc(doc: dict) -> list[dict]:
    """Split a word entry so each part's embedding text stays within _EMBED_MAX_CHARS.

    The word index always stores the full entry; only ChromaDB gets the split parts.
    Since individual senses are now truncated in _sense_text(), we just need to split
    when accumulated senses exceed the limit.
    """
    senses = doc.get("senses", [])
    header = doc["word"] + " | "

    parts: list[dict] = []
    current: list[dict] = []
    current_len = len(header)

    for sense in senses:
        st = _sense_text(sense)
        sep_len = 3  # " | " separator
        
        # If adding this sense exceeds limit and we have senses, split
        if current and current_len + sep_len + len(st) > _EMBED_MAX_CHARS:
            parts.append({**doc, "senses": current})
            current = [sense]
            current_len = len(header) + len(st)
        else:
            current.append(sense)
            current_len += sep_len + len(st)

    if current:
        parts.append({**doc, "senses": current})

    return parts


def _doc_to_text(doc: dict) -> str:
    """Rich text representation used for embedding — covers all senses in the chunk."""
    parts = [doc["word"]]
    for sense in doc.get("senses", []):
        parts.append(_sense_text(sense))
    return " | ".join(p for p in parts if p)


def _doc_to_document(doc: dict) -> str:
    """Subset of doc fields stored as the ChromaDB document (returned at query time)."""
    payload = {
        "word":         doc["word"],
        "source":       doc["source"],
        "category":     doc["category"],
        "ipa":          doc.get("ipa", []),
        "forms":        doc.get("forms", []),
        "senses":       doc.get("senses", []),
        "etymology":    doc.get("etymology", ""),
        "synonyms":     doc.get("synonyms", []),
        "antonyms":     doc.get("antonyms", []),
        "related_words": doc.get("related_words", [])[:10],
    }
    return json.dumps(payload, ensure_ascii=False)


def _doc_to_metadata(doc: dict):
    """Flat metadata for ChromaDB filtering (no nested structures)."""
    return {
        "word": doc["word"],
        "source": doc["source"],
        "category": doc["category"],
        "difficulty": doc.get("difficulty", ""),
        "related_words": ", ".join(doc.get("related_words", [])[:10]),
    }


def run(batch_size: int = 64, reset: bool = False, resume: bool = False, update_metadata: bool = False) -> None:
    if not CORPUS_FILE.exists():
        print(f"Corpus not found: {CORPUS_FILE}")
        print("Run ingestion/unify_schema.py first.")
        return

    # Load corpus
    print("Loading corpus …")
    docs: list[dict] = []
    with CORPUS_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"  {len(docs):,} documents loaded")

    # Set up ChromaDB
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
            print(f"  Deleted existing collection '{CHROMA_COLLECTION}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()

    # ── Metadata-only update (no re-embedding) ─────────────────────────────────
    if update_metadata:
        print(f"Updating metadata for {existing_count:,} existing documents …")
        for i in tqdm(range(0, len(docs), batch_size)):
            batch = docs[i : i + batch_size]
            ids = [f"doc_{i + j}" for j in range(len(batch))]
            metadatas = [_doc_to_metadata(d) for d in batch]
            documents = [_doc_to_document(d) for d in batch]
            collection.update(ids=ids, metadatas=metadatas, documents=documents)
        print(f"\nMetadata updated for {collection.count():,} documents → {CHROMA_DIR}")
        return

    # ── Resume: skip already-indexed docs ─────────────────────────────────────
    start = 0
    if resume and existing_count > 0:
        start = existing_count
        print(f"  Resuming from document {start:,} ({existing_count:,} already indexed)")
    elif existing_count > 0 and not reset:
        print(f"  Collection already has {existing_count:,} documents. Use --reset to re-index or --resume to continue.")
        return

    # ── Expand docs into split parts before indexing ──────────────────────────
    remaining = docs[start:]
    split_docs: list[dict] = []
    for doc in remaining:
        split_docs.extend(_split_doc(doc))
    print(f"Embedding {len(split_docs):,} chunks ({len(remaining):,} corpus docs) in batches of {batch_size} …")

    # ── Embed and insert ───────────────────────────────────────────────────────
    for i in tqdm(range(0, len(split_docs), batch_size)):
        batch = split_docs[i : i + batch_size]
        ids = [f"doc_{start + i + j}" for j in range(len(batch))]

        texts = [_doc_to_text(d) for d in batch]
        batch_num = i // batch_size + 1
        total_chars = sum(len(t) for t in texts)
        
        try:
            embeddings = _get_embedding(texts, OLLAMA_BASE_URL, EMBED_MODEL)
        except Exception as e:
            logger.error(f"Failed to embed batch {batch_num}")
            raise
        metadatas = [_doc_to_metadata(d) for d in batch]
        documents = [_doc_to_document(d) for d in batch]

        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    print(f"\nIndexed {collection.count():,} chunks → {CHROMA_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--reset", action="store_true", help="Drop and rebuild the collection from scratch")
    ap.add_argument("--resume", action="store_true", help="Continue indexing from where a previous run stopped")
    ap.add_argument("--update-metadata", action="store_true", help="Patch metadata on existing vectors without re-embedding")
    args = ap.parse_args()

    run(batch_size=args.batch_size, reset=args.reset, resume=args.resume, update_metadata=args.update_metadata)
