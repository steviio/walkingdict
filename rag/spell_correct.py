"""Spell correction layer — SymSpell for single words, RapidFuzz for phrases.

SymSpell operates on a pre-built frequency dictionary derived from the corpus
(data/processed/symspell_dict.txt).  If that file doesn't exist yet (e.g. during
testing) it falls back to a built-in English frequency list.

Usage
─────
    from rag.spell_correct import SpellCorrector

    sc = SpellCorrector()
    sc.load()                          # call once at startup
    result = sc.correct("gosting")    # → ("ghosting", 1)
    result = sc.correct("ghosting")   # → ("ghosting", 0)  (exact match)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SYMSPELL_DICT_FILE, MAX_EDIT_DISTANCE, FUZZY_SCORE_CUTOFF


@dataclass
class CorrectionResult:
    original: str
    corrected: str
    edit_distance: int
    was_corrected: bool


class SpellCorrector:
    """Thin wrapper around SymSpell + RapidFuzz for typo correction."""

    def __init__(
        self,
        max_edit_distance: int = MAX_EDIT_DISTANCE,
        fuzzy_cutoff: int = FUZZY_SCORE_CUTOFF,
    ) -> None:
        self.max_edit_distance = max_edit_distance
        self.fuzzy_cutoff = fuzzy_cutoff
        self._sym: object | None = None
        self._loaded = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load SymSpell dictionary.  Safe to call multiple times."""
        if self._loaded:
            return

        try:
            from symspellpy import SymSpell
        except ImportError:
            print("[SpellCorrector] symspellpy not installed — spell correction disabled")
            self._loaded = True
            return

        sym = SymSpell(max_dictionary_edit_distance=self.max_edit_distance, prefix_length=7)

        # Always load the built-in 82K-word English frequency dictionary first.
        # This ensures common English words (e.g. "defy", "ephemeral") are never
        # misidentified as typos just because they aren't in the corpus.
        import importlib.resources as pkg_resources
        try:
            freq_file = "frequency_dictionary_en_82_765.txt"
            with pkg_resources.files("symspellpy").joinpath(freq_file).open("rb") as fh:
                tmp = Path("/tmp/symspell_builtin.txt")
                tmp.write_bytes(fh.read())
            sym.load_dictionary(str(tmp), term_index=0, count_index=1)
        except Exception as e:
            print(f"[SpellCorrector] Could not load built-in dictionary: {e}")

        # Layer corpus-specific words on top (higher frequency = preferred suggestions)
        if SYMSPELL_DICT_FILE.exists():
            sym.load_dictionary(str(SYMSPELL_DICT_FILE), term_index=0, count_index=1)

        self._sym = sym
        self._loaded = True

    def correct(self, text: str) -> CorrectionResult:
        """Correct a single word or short phrase.

        Single-word input → SymSpell (fast, edit-distance based).
        Multi-word input  → RapidFuzz phrase matching against known words.
        """
        original = text.strip()
        if not original:
            return CorrectionResult(original, original, 0, False)

        if self._sym is None:
            return CorrectionResult(original, original, 0, False)

        words = original.split()
        if len(words) == 1:
            return self._correct_word(original)
        else:
            return self._correct_phrase(original)

    def get_suggestions(self, word: str, top_n: int = 5) -> list[tuple[str, int]]:
        """Return up to top_n (candidate, edit_distance) pairs, excluding exact matches."""
        if self._sym is None:
            return []

        try:
            from symspellpy import Verbosity
            suggestions = self._sym.lookup(
                word.lower(), Verbosity.ALL, max_edit_distance=self.max_edit_distance
            )
            # Only return suggestions that are actual corrections (edit_distance > 0)
            return [(s.term, s.distance) for s in suggestions if s.distance > 0][:top_n]
        except Exception:
            return []

    # ── Internal ───────────────────────────────────────────────────────────────

    def _correct_word(self, word: str) -> CorrectionResult:
        try:
            from symspellpy import Verbosity

            suggestions = self._sym.lookup(
                word.lower(),
                Verbosity.CLOSEST,
                max_edit_distance=self.max_edit_distance,
            )
            if not suggestions:
                return CorrectionResult(word, word, 0, False)

            best = suggestions[0]
            if best.distance == 0:
                return CorrectionResult(word, word, 0, False)
            return CorrectionResult(word, best.term, best.distance, True)

        except Exception as exc:
            print(f"[SpellCorrector] SymSpell error: {exc}")
            return CorrectionResult(word, word, 0, False)

    def _correct_phrase(self, phrase: str) -> CorrectionResult:
        """Use RapidFuzz to find the closest known multi-word entry."""
        try:
            from rapidfuzz import process, fuzz

            # Build candidate list from SymSpell's internal word list
            if not hasattr(self._sym, "words"):
                return CorrectionResult(phrase, phrase, 0, False)

            candidates = list(self._sym.words.keys())
            if not candidates:
                return CorrectionResult(phrase, phrase, 0, False)

            match = process.extractOne(
                phrase.lower(),
                candidates,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self.fuzzy_cutoff,
            )
            if match is None:
                return CorrectionResult(phrase, phrase, 0, False)

            corrected, score, _ = match
            if corrected.lower() == phrase.lower():
                return CorrectionResult(phrase, phrase, 0, False)

            # Approximate edit distance as inverse of score
            est_distance = max(1, round((100 - score) / 20))
            return CorrectionResult(phrase, corrected, est_distance, True)

        except ImportError:
            return CorrectionResult(phrase, phrase, 0, False)
        except Exception as exc:
            print(f"[SpellCorrector] RapidFuzz error: {exc}")
            return CorrectionResult(phrase, phrase, 0, False)
