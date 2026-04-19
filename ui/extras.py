"""Main-panel supplementary components — related words and source attribution."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Related words panel ────────────────────────────────────────────────────────

def render_related_words(chunks: list[dict[str, Any]]) -> None:
    """Display related words extracted from the last query's chunks."""
    if not chunks:
        return

    related: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        raw = chunk.get("related_words", "")
        if isinstance(raw, str):
            words = [w.strip() for w in raw.split(",") if w.strip()]
        elif isinstance(raw, list):
            words = [str(w).strip() for w in raw if str(w).strip()]
        else:
            words = []
        for w in words:
            if w.lower() not in seen:
                related.append(w)
                seen.add(w.lower())

    if not related:
        return

    st.markdown("**Related words**")
    cols = st.columns(min(len(related), 4))
    for i, word in enumerate(related[:8]):
        with cols[i % 4]:
            if st.button(word, key=f"related_{word}_{i}", use_container_width=True):
                st.session_state["pending_query"] = word
                st.rerun()


# ── Source attribution ─────────────────────────────────────────────────────────

SOURCE_LABELS = {
    "wiktionary": "Wiktionary (CC BY-SA 3.0)",
    "urban_dictionary": "Urban Dictionary",
    "wordnet": "Open English WordNet (CC BY 4.0)",
    "idioms": "Idiom Corpora",
}


def render_source_attribution(chunks: list[dict[str, Any]]) -> None:
    """Show a clickable 'Sources:' summary that expands to reveal each full entry."""
    if not chunks:
        return

    unique_sources = list(dict.fromkeys(
        SOURCE_LABELS.get(c.get("source", ""), c.get("source", ""))
        for c in chunks
        if c.get("source")
    ))
    summary = "Sources: " + " · ".join(unique_sources)

    with st.expander(summary, expanded=False):
        seen: set[tuple] = set()
        for chunk in chunks:
            source = chunk.get("source", "")
            word = chunk.get("word", "")

            dedup_key = (source, word.lower())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            label = SOURCE_LABELS.get(source, source)
            chroma_id = chunk.get("_chroma_id", "")
            id_str = f" · `{chroma_id}`" if chroma_id else ""
            st.markdown(f"**{label}**{id_str}")

            if word:
                st.caption(f"Word: {word}")

            if chunk.get("ipa"):
                ipa_str = " | ".join(
                    f"/{item['ipa']}/ ({', '.join(item['tags'])})" if item.get("tags")
                    else f"/{item['ipa']}/"
                    for item in chunk["ipa"]
                )
                st.caption(f"IPA: {ipa_str}")

            if chunk.get("forms"):
                st.caption(f"Forms: {', '.join(chunk['forms'])}")

            for sense in chunk.get("senses", []):
                pos = sense.get("part_of_speech", "")
                defn = sense.get("definition", "")
                st.markdown(f"*({pos})* {defn}" if pos else f"> {defn}")
                for ex in sense.get("examples", []):
                    st.caption(f"e.g. {ex}")

            if chunk.get("synonyms"):
                st.caption(f"Synonyms: {', '.join(chunk['synonyms'])}")
            if chunk.get("antonyms"):
                st.caption(f"Antonyms: {', '.join(chunk['antonyms'])}")
            if chunk.get("etymology"):
                st.caption(f"Etymology: {chunk['etymology']}")
            if chunk.get("related_words"):
                st.caption(f"Related: {', '.join(chunk['related_words'])}")

            st.divider()
