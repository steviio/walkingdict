"""Build exact-lookup word index from the ChromaDB collection.

Reads all document IDs and word metadata from ChromaDB and writes a
word → [chroma_id, ...] mapping so the query pipeline can fetch exact
hits via collection.get(ids=[...]) instead of a full metadata scan.

Output
──────
  data/processed/word_index.pkl  — dict[str, list[str]]  (word → chroma IDs)

Run:
    python -m ingestion.build_word_index
"""

import pickle
import sys
from pathlib import Path
import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_DIR, CHROMA_COLLECTION, WORD_INDEX_FILE, PROCESSED_DIR


def run() -> None:
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        print(f"ChromaDB not found at {CHROMA_DIR}")
        print("Run ingestion/embed_and_index.py first.")
        return

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=CHROMA_COLLECTION)

    total_docs = collection.count()
    print(f"Reading {total_docs:,} documents from ChromaDB …")

    word_to_ids: dict[str, list[str]] = {}
    batch_size = 5000
    
    # Fetch documents in batches to avoid SQLite variable limit
    for offset in range(0, total_docs, batch_size):
        batch = collection.get(include=["metadatas"], offset=offset, limit=batch_size)
        for doc_id, meta in zip(batch["ids"], batch["metadatas"]):
            word = meta["word"].lower()
            word_to_ids.setdefault(word, []).append(doc_id)
        print(f"  Processed {min(offset + batch_size, total_docs):,} / {total_docs:,} documents")

    print(f"  {len(word_to_ids):,} unique words")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with WORD_INDEX_FILE.open("wb") as fh:
        pickle.dump(word_to_ids, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Word index saved → {WORD_INDEX_FILE}")


if __name__ == "__main__":
    run()
