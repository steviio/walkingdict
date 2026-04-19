"""3-layer query pipeline: exact lookup → fuzzy correction → vector search.

Retrieval strategy:
  1. Exact/fuzzy lookup  — always, returns all senses for the matched word
  2. Vector search       — always, returns top_k semantically similar *other* words
  Both results are returned together; exact hits are the primary entry,
  vector results are related concepts.

Usage
─────
    from rag.query_pipeline import QueryPipeline

    pipeline = QueryPipeline()
    pipeline.load()

    result = pipeline.query("gosting")
    # result.chunks          → all entries for the matched word
    # result.similar_chunks  → top_k semantically related words
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WORD_INDEX_FILE, TOP_K_RESULTS
from rag.spell_correct import SpellCorrector, CorrectionResult
from rag.retriever import Retriever, RetrievalResult


@dataclass
class QueryResult:
    query: str                              # original user input (normalised)
    correction: CorrectionResult | None     # populated when fuzzy correction fires
    chunks: list[dict[str, Any]]            # all entries for the exact/corrected word
    similar_chunks: list[dict[str, Any]]    # semantically similar words (different word)
    lookup_method: str                      # "exact" | "fuzzy" | "vector" | "llm_only"
    low_confidence: bool = False            # True when nothing useful was found


class QueryPipeline:
    """Orchestrates exact → fuzzy lookup + parallel vector search."""

    def __init__(self) -> None:
        self._word_index: dict[str, list[str]] = {}  # word → list of ChromaDB IDs
        self._spell: SpellCorrector = SpellCorrector()
        self._retriever: Retriever = Retriever()
        self._loaded = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        if self._loaded:
            return

        if WORD_INDEX_FILE.exists():
            with WORD_INDEX_FILE.open("rb") as fh:
                self._word_index = pickle.load(fh)
            print(f"[QueryPipeline] Word index: {len(self._word_index):,} entries")
        else:
            print(f"[QueryPipeline] No word index at {WORD_INDEX_FILE} — exact lookup disabled")

        self._spell.load()

        try:
            self._retriever.load()
            print(f"[QueryPipeline] ChromaDB: {self._retriever.count():,} vectors loaded")
        except RuntimeError as exc:
            print(f"[QueryPipeline] {exc} — vector search disabled")

        self._loaded = True

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_correction_candidates(self, raw_input: str, top_n: int = 4) -> list[tuple[str, int]]:
        """Return correction candidates if input looks like a typo, else empty list."""
        if not self._loaded:
            self.load()

        normalised = self._normalize(raw_input)
        if not normalised:
            return []

        if normalised in self._word_index:
            return []

        return self._spell.get_suggestions(normalised, top_n=top_n)

    def query(
        self,
        raw_input: str,
        top_k: int = TOP_K_RESULTS,
        category_hint: str | None = None,
    ) -> QueryResult:
        """Run exact/fuzzy lookup + vector search and return combined results."""
        if not self._loaded:
            self.load()

        normalised = self._normalize(raw_input)
        if not normalised:
            return QueryResult(normalised, None, [], [], "exact", low_confidence=True)

        # ── Exact lookup ───────────────────────────────────────────────────────
        correction = None
        exact_word = normalised
        exact_hits = self._fetch_exact(normalised)

        if not exact_hits:
            correction = self._spell.correct(normalised)
            if correction.was_corrected:
                exact_word = correction.corrected
                exact_hits = self._fetch_exact(exact_word)

        # ── Vector search for similar words (always runs) ──────────────────────
        search_term = exact_word if exact_hits else normalised
        try:
            # Fetch extra results to account for filtering out the exact word
            vector_results = self._retriever.retrieve(
                search_term,
                top_k=top_k + 5,
                category_filter=category_hint,
            )
            similar = [
                self._result_to_chunk(r) for r in vector_results
                if r.metadata.get("word", "").lower() != exact_word.lower()
            ][:top_k]
        except RuntimeError:
            similar = []

        # ── Return combined result ─────────────────────────────────────────────
        if exact_hits:
            method = "fuzzy" if (correction and correction.was_corrected) else "exact"
            return QueryResult(
                query=normalised,
                correction=correction,
                chunks=exact_hits,
                similar_chunks=similar,
                lookup_method=method,
            )

        if similar:
            return QueryResult(
                query=normalised,
                correction=correction,
                chunks=[],
                similar_chunks=similar,
                lookup_method="vector",
                low_confidence=True,
            )

        return QueryResult(
            query=normalised,
            correction=correction,
            chunks=[],
            similar_chunks=[],
            lookup_method="llm_only",
            low_confidence=True,
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _fetch_exact(self, word: str) -> list[dict]:
        ids = self._word_index.get(word, [])
        if not ids:
            return []
        return [self._result_to_chunk(r) for r in self._retriever.get_by_ids(ids)]

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s'\-]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _result_to_chunk(r: RetrievalResult) -> dict:
        chunk = json.loads(r.document)
        chunk["_chroma_id"] = r.id
        chunk["_vector_score"] = r.score
        return chunk
