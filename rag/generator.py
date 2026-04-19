"""Ollama LLM interface with streaming support.

Usage
─────
    from rag.generator import Generator

    gen = Generator()

    # Streaming (yields str tokens)
    for token in gen.stream(messages):
        print(token, end="", flush=True)

    # Non-streaming (returns full response)
    response = gen.generate(messages)
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Generator as TypingGenerator

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


class Generator:
    """Thin wrapper around the Ollama /api/chat endpoint."""

    def __init__(
        self,
        model: str = LLM_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ── Public API ─────────────────────────────────────────────────────────────

    def stream(
        self,
        messages: list[dict[str, str]],
    ) -> TypingGenerator[str, None, None]:
        """Yield tokens as they arrive from the LLM."""
        payload = self._build_payload(messages, stream=True)
        req = self._make_request(payload)

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = obj.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if obj.get("done"):
                        break
        except urllib.error.URLError as exc:
            yield f"\n\n[Error connecting to Ollama: {exc.reason}]\n"
            yield "Make sure Ollama is running: `ollama serve`"

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Return the full response as a single string (non-streaming)."""
        return "".join(self.stream(messages))

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return names of locally available Ollama models."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_payload(self, messages: list[dict[str, str]], stream: bool) -> bytes:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        return json.dumps(body).encode("utf-8")

    def _make_request(self, payload: bytes) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
