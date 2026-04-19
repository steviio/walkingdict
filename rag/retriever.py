"""ChromaDB retrieval with metadata filtering.

Usage
─────
    from rag.retriever import Retriever

    retriever = Retriever()
    retriever.load()

    results = retriever.retrieve("ghosting", top_k=5)
    results = retriever.retrieve("ghosting", category_filter="slang")
    results = retriever.get_by_exact_word("ghosting")
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    OLLAMA_BASE_URL,
    EMBED_MODEL,
    TOP_K_RESULTS,
    MIN_VECTOR_SCORE,
)


@dataclass
class RetrievalResult:
    id: str
    document: str
    metadata: dict[str, Any]
    score: float  # lower = more similar for cosine distance


class Retriever:
    """Wraps ChromaDB for semantic + filtered retrieval."""

    def __init__(self) -> None:
        self._collection = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Connect to the persisted ChromaDB collection."""
        if self._collection is not None:
            return

        import chromadb

        if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
            raise RuntimeError(
                f"ChromaDB not found at {CHROMA_DIR}. "
                "Run ingestion/embed_and_index.py first."
            )

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # Always embed queries manually via Ollama to guarantee dimension consistency
        # (avoids chromadb falling back to its default all-MiniLM-L6-v2 / 384-dim model)
        self._collection = client.get_collection(name=CHROMA_COLLECTION)

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        category_filter: str | None = None,
        source_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """Semantic vector search with optional metadata filters."""
        self._ensure_loaded()

        where: dict[str, Any] = {}
        conditions: list[dict] = []

        if category_filter:
            conditions.append({"category": {"$eq": category_filter}})
        if source_filter:
            conditions.append({"source": {"$eq": source_filter}})
        if difficulty_filter:
            conditions.append({"difficulty": {"$eq": difficulty_filter}})

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        query_embedding = self._embed(query)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = self._collection.query(**kwargs)
        except Exception as exc:
            print(f"[Retriever] Query error: {exc}")
            return []

        output: list[RetrievalResult] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for id_, doc, meta, dist in zip(ids, docs, metas, distances):
            if dist <= MIN_VECTOR_SCORE:
                output.append(RetrievalResult(id=id_, document=doc, metadata=meta, score=dist))

        return output

    def get_by_ids(self, ids: list[str]) -> list[RetrievalResult]:
        """Fetch documents by ChromaDB IDs — O(log n) vs O(n) metadata scan."""
        self._ensure_loaded()

        try:
            results = self._collection.get(
                ids=ids,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            print(f"[Retriever] ID lookup error: {exc}")
            return []

        return [
            RetrievalResult(id=id_, document=doc, metadata=meta, score=0.0)
            for id_, doc, meta in zip(results["ids"], results["documents"], results["metadatas"])
        ]

    def count(self) -> int:
        """Return total number of indexed documents."""
        self._ensure_loaded()
        return self._collection.count()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Embed a single string via Ollama /api/embed (same path as embed_and_index.py)."""
        import json
        import urllib.request

        payload = json.dumps({"model": EMBED_MODEL, "input": [text]}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data["embeddings"][0]

    def _ensure_loaded(self) -> None:
        if self._collection is None:
            self.load()
