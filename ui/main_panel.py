"""Main panel — search bar, lookup flow, and result display."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.query_pipeline import QueryPipeline
from rag.prompt_builder import PromptBuilder, UserProfile
from rag.generator import Generator
from ui.extras import render_related_words, render_source_attribution
from ui.storage import load_history, load_bookmarks, save_history, save_bookmarks


# ── Session state ──────────────────────────────────────────────────────────────

def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state["history"] = load_history()
    if "bookmarks" not in st.session_state:
        st.session_state["bookmarks"] = load_bookmarks()

    defaults = {
        "current_idx": None,
        "last_chunks": [],
        "pending_query": None,
        "pending_history_idx": None,
        "correction_candidates": [],
        "correction_original": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ── Pipeline singletons ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading WalkingDict…")
def get_pipeline() -> QueryPipeline:
    pipeline = QueryPipeline()
    pipeline.load()
    return pipeline


@st.cache_resource(show_spinner=False)
def get_generator() -> Generator:
    return Generator()


# ── Display a single stored result ─────────────────────────────────────────────

def _display_entry(entry: dict[str, Any], idx: int) -> None:
    word = entry.get("word", entry.get("query", ""))
    correction = entry.get("correction")
    method = entry.get("method", "")
    chunks = entry.get("chunks", [])
    similar_chunks = entry.get("similar_chunks", [])

    st.subheader(word)

    if correction and correction.was_corrected:
        st.info(
            f"Showing results for **{correction.corrected}** "
            f"(you typed: *{correction.original}*)"
        )

    if method == "llm_only":
        st.warning(
            "This word isn't in the corpus — answering from general knowledge.",
            icon="⚠️",
        )

    st.markdown(entry.get("response", ""))

    all_chunks = chunks + similar_chunks
    if all_chunks:
        render_related_words(all_chunks)
        render_source_attribution(all_chunks)

    col1, _ = st.columns([1, 5])
    with col1:
        bookmarks = st.session_state.get("bookmarks", [])
        label = "★ Saved" if word in bookmarks else "☆ Bookmark"
        if st.button(label, key=f"bm_btn_{idx}"):
            if word not in bookmarks:
                bookmarks.append(word)
                st.session_state["bookmarks"] = bookmarks
                save_bookmarks(bookmarks)
                st.toast(f"Bookmarked: {word}")
            st.rerun()


# ── Run a new lookup and stream the response ───────────────────────────────────

def _run_lookup(
    query: str,
    profile: UserProfile,
    pipeline: QueryPipeline,
    generator: Generator,
    skip_correction: bool = False,
) -> None:
    with st.spinner(f"Looking up \"{query}\"…"):
        result = pipeline.query(query, category_hint=_infer_category_hint(query), skip_correction=skip_correction)

    word = result.correction.corrected if (result.correction and result.correction.was_corrected) else result.query

    st.subheader(word)

    if result.correction and result.correction.was_corrected:
        st.info(
            f"Showing results for **{result.correction.corrected}** "
            f"(you typed: *{result.correction.original}*)"
        )

    if result.lookup_method == "llm_only" or result.lookup_method == "vector":
        st.warning(
            "This exact word isn't in the corpus — answering from general knowledge.",
            icon="⚠️",
        )

    effective_query = word

    history = st.session_state.get("history", [])
    recent_words = [
        e.get("word") or e.get("query", "")
        for e in history[-10:]
    ]
    recent_words = [w for w in recent_words if w and w.lower() != effective_query.lower()][-5:]

    builder = PromptBuilder()
    messages = builder.build(
        query=effective_query,
        chunks=result.chunks,
        similar_chunks=result.similar_chunks,
        profile=profile,
        low_confidence=result.low_confidence,
        recent_words=recent_words,
    )

    full_response = ""
    placeholder = st.empty()

    if generator.is_available():
        for token in generator.stream(messages):
            full_response += token
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)
    else:
        full_response = (
            "Ollama is not running. "
            "Start it with `ollama serve` and refresh the page."
        )
        placeholder.error(full_response)

    all_chunks = result.chunks + result.similar_chunks
    if all_chunks:
        render_related_words(all_chunks)
        render_source_attribution(all_chunks)

    entry = {
        "word": word,
        "query": query,
        "response": full_response,
        "method": result.lookup_method,
        "correction": result.correction,
        "chunks": result.chunks,
        "similar_chunks": result.similar_chunks,
        "low_confidence": result.low_confidence,
    }
    history = st.session_state.get("history", [])
    history.append(entry)
    new_idx = len(history) - 1
    st.session_state["history"] = history
    st.session_state["current_idx"] = new_idx
    st.session_state["last_chunks"] = result.chunks
    save_history(history)

    col1, _ = st.columns([1, 5])
    with col1:
        bookmarks = st.session_state.get("bookmarks", [])
        label = "★ Saved" if word in bookmarks else "☆ Bookmark"
        if st.button(label, key=f"bm_btn_{new_idx}"):
            if word not in bookmarks:
                bookmarks.append(word)
                st.session_state["bookmarks"] = bookmarks
                save_bookmarks(bookmarks)
                st.toast(f"Bookmarked: {word}")
            st.rerun()


def _infer_category_hint(query: str) -> str | None:
    q = query.lower()
    if any(kw in q for kw in ["slang", "meaning", "urban"]):
        return "slang"
    if any(kw in q for kw in ["idiom", "phrase", "expression"]):
        return "idiom"
    return None


# ── Public entry point ─────────────────────────────────────────────────────────

def render_main_panel(profile: UserProfile) -> None:
    init_session_state()

    pipeline = get_pipeline()
    generator = get_generator()

    # ── Search bar — always at the top ────────────────────────────────────────
    with st.form(key="search_form", clear_on_submit=True, border=False):
        col_in, col_btn = st.columns([6, 1])
        with col_in:
            user_input = st.text_input(
                "search",
                placeholder="Look up a word, phrase, or slang…",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Search →", use_container_width=True)

    st.divider()

    # ── Resolve navigation / pending queries ───────────────────────────────────
    history = st.session_state.get("history", [])

    pending_idx = st.session_state.pop("pending_history_idx", None)
    if pending_idx is not None and 0 <= pending_idx < len(history):
        st.session_state["current_idx"] = pending_idx
        st.session_state["correction_candidates"] = []
        st.session_state["correction_original"] = None

    pending = st.session_state.pop("pending_query", None)
    pending_skip_correction = st.session_state.pop("pending_skip_correction", False)
    if pending:
        st.session_state["correction_candidates"] = []
        st.session_state["correction_original"] = None
        _run_lookup(pending, profile, pipeline, generator, skip_correction=pending_skip_correction)
        st.rerun()

    if submitted and user_input.strip():
        raw = user_input.strip()
        st.session_state["correction_candidates"] = []
        st.session_state["correction_original"] = None

        candidates = pipeline.get_correction_candidates(raw)
        if candidates:
            st.session_state["correction_candidates"] = candidates
            st.session_state["correction_original"] = raw
            st.rerun()
        else:
            st.session_state["pending_query"] = raw
            st.rerun()

    # ── Spell-correction candidate picker ─────────────────────────────────────
    candidates = st.session_state.get("correction_candidates", [])
    original_input = st.session_state.get("correction_original")

    if candidates and original_input:
        st.markdown(f"**Did you mean…?** (you typed: *{original_input}*)")
        cols = st.columns(len(candidates) + 1)
        for i, (word, _) in enumerate(candidates):
            with cols[i]:
                if st.button(word, key=f"cand_{i}_{word}"):
                    st.session_state["correction_candidates"] = []
                    st.session_state["correction_original"] = None
                    st.session_state["pending_query"] = word
                    st.rerun()
        with cols[len(candidates)]:
            if st.button(f'Search "{original_input}" anyway', key="cand_original"):
                st.session_state["correction_candidates"] = []
                st.session_state["correction_original"] = None
                st.session_state["pending_query"] = original_input
                st.session_state["pending_skip_correction"] = True
                st.rerun()
        return

    # ── Display current result ─────────────────────────────────────────────────
    current_idx = st.session_state.get("current_idx")
    history = st.session_state.get("history", [])

    if current_idx is not None and history:
        _display_entry(history[current_idx], current_idx)
    else:
        st.markdown(
            "<div style='text-align:center; padding: 4rem 0; color: #888;'>"
            "<h3>Look up any word, phrase, or slang</h3>"
            "<p>Type above and get an explanation tailored to your level.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
