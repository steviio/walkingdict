"""Right column — Word of the Day, recent searches, and bookmarks."""

from __future__ import annotations

import json
import random
import sys
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CORPUS_FILE, WOTD_CACHE_FILE
from ui.storage import save_history, save_bookmarks


# ── Word of the Day ────────────────────────────────────────────────────────────

def _sample_word_of_the_day() -> dict | None:
    """Pick a deterministic daily word from the corpus."""
    if not CORPUS_FILE.exists():
        return None

    today_seed = int(date.today().strftime("%Y%m%d"))
    rng = random.Random(today_seed)

    chosen = None
    with CORPUS_FILE.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            if rng.random() < 1 / i:
                chosen = line

    if chosen is None:
        return None

    try:
        doc = json.loads(chosen)
    except json.JSONDecodeError:
        return None

    senses = doc.get("senses") or []
    first_def = senses[0].get("definition", "") if senses else ""
    if len(first_def) < 20:
        return None
    return doc


def get_word_of_the_day() -> dict | None:
    """Return today's Word of the Day (cached within the day)."""
    today = str(date.today())

    if st.session_state.get("wotd_date") == today and st.session_state.get("wotd"):
        return st.session_state["wotd"]

    if WOTD_CACHE_FILE.exists():
        try:
            with WOTD_CACHE_FILE.open(encoding="utf-8") as fh:
                cache = json.load(fh)
            if cache.get("date") == today:
                st.session_state["wotd_date"] = today
                st.session_state["wotd"] = cache["word"]
                return cache["word"]
        except Exception:
            pass

    wotd = _sample_word_of_the_day()
    if wotd:
        st.session_state["wotd_date"] = today
        st.session_state["wotd"] = wotd
        try:
            WOTD_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with WOTD_CACHE_FILE.open("w", encoding="utf-8") as fh:
                json.dump({"date": today, "word": wotd}, fh)
        except Exception:
            pass

    return wotd


def render_word_of_the_day() -> None:
    """Display Word of the Day card and a button to look it up."""
    wotd = get_word_of_the_day()
    if wotd is None:
        return

    word = wotd.get("word", "")
    senses = wotd.get("senses") or []
    first_sense = senses[0] if senses else {}
    raw_def = first_sense.get("definition", "")
    definition = raw_def[:200] + ("…" if len(raw_def) > 200 else "")

    with st.container(border=True):
        st.markdown("**Word of the Day**")
        st.markdown(f"### {word}")
        if first_sense.get("part_of_speech"):
            st.caption(first_sense["part_of_speech"])
        st.write(definition)
        if st.button("Look it up →", key="wotd_btn"):
            st.session_state["pending_query"] = word
            st.rerun()


# ── Recent searches ────────────────────────────────────────────────────────────

def render_recent_searches() -> None:
    """Render the recent search history list and clear button."""
    history = st.session_state.get("history", [])
    current_idx = st.session_state.get("current_idx")

    if not history:
        return

    st.subheader("Recent")
    for i, entry in enumerate(reversed(history[-30:])):
        actual_idx = len(history) - 1 - i
        word = entry.get("word", entry.get("query", ""))
        is_current = actual_idx == current_idx

        label = word
        if entry.get("method") == "llm_only":
            label += " ·"

        if st.button(
            label,
            key=f"hist_{actual_idx}",
            use_container_width=True,
            type="primary" if is_current else "secondary",
        ):
            st.session_state["pending_history_idx"] = actual_idx
            st.rerun()

    if st.button("Clear all history", use_container_width=True):
        st.session_state["history"] = []
        st.session_state["current_idx"] = None
        save_history([])
        st.rerun()


# ── Bookmarks ──────────────────────────────────────────────────────────────────

def render_bookmarks() -> None:
    """Render the bookmarks list."""
    bookmarks = st.session_state.get("bookmarks", [])
    if not bookmarks:
        return

    st.subheader(f"Bookmarks ({len(bookmarks)})")
    for bm in reversed(bookmarks[-20:]):
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(bm, key=f"bm_{bm}", use_container_width=True):
                st.session_state["pending_query"] = bm
                st.rerun()
        with col2:
            if st.button("✕", key=f"rm_{bm}"):
                bookmarks.remove(bm)
                st.session_state["bookmarks"] = bookmarks
                save_bookmarks(bookmarks)
                st.rerun()


# ── Composite entry point ──────────────────────────────────────────────────────

def render_right_column() -> None:
    """Render all right-column blocks in order."""
    render_word_of_the_day()
    st.divider()
    render_recent_searches()
    st.divider()
    render_bookmarks()
