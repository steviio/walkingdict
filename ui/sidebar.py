"""Streamlit sidebar — user preference controls.

Call render_sidebar() to display and return the current UserProfile.
"""

from __future__ import annotations

import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROFICIENCY_LEVELS, INTEREST_OPTIONS, DEFAULT_PROFICIENCY, DEFAULT_INTERESTS
from rag.prompt_builder import UserProfile


def render_sidebar() -> UserProfile:
    """Render preference controls and return the current UserProfile."""
    with st.sidebar:
        st.title("WalkingDict")
        st.caption("A dictionary that actually teaches you.")
        st.divider()

        # ── Proficiency ────────────────────────────────────────────────────────
        st.subheader("Your level")
        proficiency = st.select_slider(
            "Proficiency",
            options=PROFICIENCY_LEVELS,
            value=st.session_state.get("proficiency", DEFAULT_PROFICIENCY),
            label_visibility="collapsed",
        )
        st.session_state["proficiency"] = proficiency

        level_descriptions = {
            "beginner": "Simple words, concrete examples, short sentences.",
            "intermediate": "Natural language, some idioms, real-life examples.",
            "advanced": "Rich vocabulary, collocations, cultural nuance.",
        }
        st.caption(level_descriptions[proficiency])

        st.divider()

        # ── Interests ──────────────────────────────────────────────────────────
        st.subheader("Your interests")
        st.caption("Examples will reference these topics when possible.")

        saved_interests = st.session_state.get("interests", DEFAULT_INTERESTS)
        interests = st.multiselect(
            "Interests",
            options=INTEREST_OPTIONS,
            default=saved_interests,
            label_visibility="collapsed",
        )
        if not interests:
            interests = ["everyday life"]
        st.session_state["interests"] = interests

        st.divider()

        # ── Explanation style ──────────────────────────────────────────────────
        st.subheader("Explanation style")
        style = st.radio(
            "Style",
            options=["vivid", "concise", "academic"],
            index=["vivid", "concise", "academic"].index(
                st.session_state.get("explanation_style", "vivid")
            ),
            label_visibility="collapsed",
            captions=[
                "Storytelling with sensory detail",
                "Just the essentials",
                "Precise, formal, etymological",
            ],
        )
        st.session_state["explanation_style"] = style

    return UserProfile(
        proficiency=proficiency,
        interests=interests,
        explanation_style=style,
    )
