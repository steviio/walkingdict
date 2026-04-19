"""Assemble the system + user prompt from retrieval output and user preferences.

Usage
─────
    from rag.prompt_builder import PromptBuilder, UserProfile

    profile = UserProfile(proficiency="beginner", interests=["cooking", "technology"])
    builder = PromptBuilder()

    messages = builder.build(
        query="ephemeral",
        chunks=[...],          # exact word entries from QueryPipeline
        similar_chunks=[...],  # related words from vector search
        profile=profile,
        low_confidence=False,
    )
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEFAULT_PROFICIENCY, DEFAULT_INTERESTS, DEFAULT_LANGUAGE


# ── User profile ───────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    proficiency: str = DEFAULT_PROFICIENCY
    interests: list[str] = field(default_factory=lambda: list(DEFAULT_INTERESTS))
    target_language: str = DEFAULT_LANGUAGE
    explanation_style: str = "vivid"  # vivid | concise | academic


# ── Prompt builder ─────────────────────────────────────────────────────────────

class PromptBuilder:

    PROFICIENCY_GUIDANCE = {
        "beginner": (
            "Use very simple, everyday words. Short sentences. "
            "Avoid jargon entirely. Use relatable, concrete examples. "
            "Think: explaining to someone who just started learning English."
        ),
        "intermediate": (
            "Use clear, natural language. Some idiomatic expressions are fine. "
            "Explain nuances briefly. Examples should connect to real life."
        ),
        "advanced": (
            "Use rich vocabulary freely. Include collocations, "
            "subtle connotation differences, and cultural context. "
            "Examples can include literary or professional contexts."
        ),
    }

    STYLE_GUIDANCE = {
        "vivid": "Use storytelling and sensory details. Paint a picture the user can see.",
        "concise": "Be direct. Cover the essentials in as few words as possible.",
        "academic": "Use precise, formal language. Note linguistic and etymological detail.",
    }

    def build(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        profile: UserProfile,
        low_confidence: bool = False,
        recent_words: list[str] | None = None,
        similar_chunks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        """Return a list of chat messages suitable for the Ollama chat API."""
        system = self._system_prompt(profile)
        user = self._user_message(
            query, chunks, low_confidence,
            recent_words or [], similar_chunks or [],
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ── Internal ───────────────────────────────────────────────────────────────

    def _system_prompt(self, profile: UserProfile) -> str:
        proficiency_note = self.PROFICIENCY_GUIDANCE.get(
            profile.proficiency, self.PROFICIENCY_GUIDANCE["intermediate"]
        )
        style_note = self.STYLE_GUIDANCE.get(profile.explanation_style, self.STYLE_GUIDANCE["vivid"])
        interests_str = ", ".join(profile.interests) if profile.interests else "everyday life"

        return f"""You are WalkingDict — an immersive language tutor and dictionary assistant.

## User Profile
- Proficiency level: {profile.proficiency}
- Interests: {interests_str}
- Target language: {profile.target_language}

## Your Approach
{proficiency_note}

## Style
{style_note}

## Response Format

Structure EVERY response using these sections in order.
Important: section names are instruction labels only. In the final answer, DO NOT print titles for Section 1.

### Section 1: Senses
Instructions:
- Do not print a section title.
- For each sense, start with the part of speech in parentheses (if available), then the definition. 
- If the sense is slang or informal, mark it clearly as "(slang)" or "(informal)" after the part of speech.
- The definition should be in simple language appropriate to the user's proficiency level, even if the original dictionary entry is more complex.
- Include one example sentence if available, ideally tied to the user's interests.
- Number senses as 1., 2., 3. ...

### Section 2: Etymology
Instructions:
- First print the title "**Etymology:**" followed by a concise explanation of the word's origin and history. Focus on what makes the word memorable.
- If the etymology includes a root word that is still recognizable, highlight that connection as a hook for the user to remember the word.
- If unavailable, print "**Etymology:** Not available."

### Section 3: Word connections
Instructions:
- First print the title "**Word connections:**"
- Then include any of the following that are available and relevant, prioritising those most likely to resonate with the user based on their profile and recent lookups:
    - Synonyms: 2–3 synonyms the user likely knows at their level
    - Antonyms: 1–2 opposites, if meaningful
    - Related: linked words for user to remember or looked-like words that are often confused. If any of the user's recent lookups are related, mention them.
    - If a line has no data, write "Not available."

## Exact Output Order (must follow)
1) Senses list, no heading
2) **Etymology:** ...
3) **Word connections:** ...

## Rules
1. Explain ONLY in {profile.target_language}.
2. Calibrate vocabulary to {profile.proficiency} level (i+1 principle — mostly familiar words with one or two slightly richer ones).
3. Tie examples to the user's interests ({interests_str}) where natural.
4. If the word is slang or informal, mark it explicitly in the Senses section.
5. Always include all sections — never skip any.
6. Final check before answering: do NOT output headings for the first sections.
"""

    def _user_message(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        low_confidence: bool,
        recent_words: list[str],
        similar_chunks: list[dict[str, Any]],
    ) -> str:
        parts: list[str] = []

        # ── Primary dictionary entries ─────────────────────────────────────────
        if chunks:
            context_lines = []
            for i, chunk in enumerate(chunks, 1):
                word = chunk.get("word", query)
                source = chunk.get("source", "")
                etymology = chunk.get("etymology", "")
                related_words = chunk.get("related_words", [])
                if isinstance(related_words, str):
                    related_words = [w.strip() for w in related_words.split(",") if w.strip()]
                category = chunk.get("category", "")

                ipa = chunk.get("ipa", [])
                forms = chunk.get("forms", [])
                synonyms = chunk.get("synonyms", [])
                antonyms = chunk.get("antonyms", [])

                entry = f"[{i}] {word}"

                if ipa:
                    ipa_str = " | ".join(
                        f"/{item['ipa']}/ ({', '.join(item['tags'])})" if item.get("tags")
                        else f"/{item['ipa']}/"
                        for item in ipa[:3]
                    )
                    entry += f"\n  IPA: {ipa_str}"
                if forms:
                    entry += f"\n  Forms: {', '.join(forms)}"

                for j, sense in enumerate(chunk.get("senses", []), 1):
                    pos = sense.get("part_of_speech", "")
                    defn = sense.get("definition", "")
                    examples = sense.get("examples", [])
                    entry += f"\n  {j}. ({pos}) {defn}" if pos else f"\n  {j}. {defn}"
                    if examples:
                        entry += f"\n     e.g. {examples[0]}"

                if synonyms:
                    entry += f"\n  Synonyms: {', '.join(synonyms[:6])}"
                if antonyms:
                    entry += f"\n  Antonyms: {', '.join(antonyms[:6])}"
                if etymology:
                    entry += f"\n  Etymology: {etymology}"
                if related_words:
                    entry += f"\n  Related: {', '.join(related_words[:6])}"
                if category and category not in ("formal",):
                    entry += f"\n  Category: {category}"
                if source:
                    entry += f"\n  Source: {source}"

                context_lines.append(entry)

            parts.append("## Dictionary Context\n" + "\n\n".join(context_lines))

        # ── Related concepts from vector search ────────────────────────────────
        if similar_chunks:
            similar_lines = []
            for chunk in similar_chunks:
                word = chunk.get("word", "")
                senses = chunk.get("senses", [])
                if senses:
                    first = senses[0]
                    pos = first.get("part_of_speech", "")
                    defn = first.get("definition", "")
                    line = f"- {word} ({pos}): {defn}" if pos else f"- {word}: {defn}"
                else:
                    line = f"- {word}"
                similar_lines.append(line)
            parts.append("## Related Concepts\n" + "\n".join(similar_lines))

        # ── Recent lookups ─────────────────────────────────────────────────────
        if recent_words:
            parts.append(
                "## User's Recent Lookups\n"
                + ", ".join(recent_words)
                + "\n(Reference these in the Word connections → Related line when a meaningful link exists.)"
            )

        # ── Confidence notes ───────────────────────────────────────────────────
        if low_confidence and not chunks:
            parts.append(
                "⚠️ IMPORTANT INSTRUCTION: This word or phrase is NOT in the dictionary corpus. "
                "You must start your response with exactly this sentence: "
                "'This word isn't in my dictionary yet, so I'll do my best from general knowledge.' "
                "Then give your best explanation. Do NOT skip or rephrase this opening sentence."
            )
        elif low_confidence and chunks:
            parts.append(
                "⚠️ NOTE: The retrieved context may not be an exact match. "
                "Use it as a guide but rely on your broader knowledge where needed. "
                "If the context seems unrelated, say so briefly before answering."
            )

        parts.append(f"Please explain the word or phrase: **{query}**")

        return "\n\n".join(parts)
