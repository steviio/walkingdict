"""Persistent JSON storage for user history and bookmarks.

Files live in data/user/ and are read once at session startup (to seed
session state) then written on every mutation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HISTORY_FILE, BOOKMARKS_FILE, USER_DIR, MAX_HISTORY


# ── CorrectionResult serialization ────────────────────────────────────────────
# CorrectionResult is a dataclass — not JSON-serializable by default.

def _serialize_correction(c: Any) -> dict | None:
    if c is None:
        return None
    return {
        "original": c.original,
        "corrected": c.corrected,
        "edit_distance": c.edit_distance,
        "was_corrected": c.was_corrected,
    }


def _deserialize_correction(d: dict | None) -> Any:
    if d is None:
        return None
    from rag.spell_correct import CorrectionResult
    return CorrectionResult(**d)


# ── History ────────────────────────────────────────────────────────────────────

def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open(encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data:
            entry["correction"] = _deserialize_correction(entry.get("correction"))
        return data
    except Exception:
        return []


def save_history(history: list[dict]) -> None:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    trimmed = history[-MAX_HISTORY:]
    serializable = []
    for entry in trimmed:
        e = dict(entry)
        e["correction"] = _serialize_correction(e.get("correction"))
        serializable.append(e)
    with HISTORY_FILE.open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, ensure_ascii=False, indent=2)


# ── Bookmarks ──────────────────────────────────────────────────────────────────

def load_bookmarks() -> list[str]:
    if not BOOKMARKS_FILE.exists():
        return []
    try:
        with BOOKMARKS_FILE.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []


def save_bookmarks(bookmarks: list[str]) -> None:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    with BOOKMARKS_FILE.open("w", encoding="utf-8") as fh:
        json.dump(bookmarks, fh, ensure_ascii=False, indent=2)
