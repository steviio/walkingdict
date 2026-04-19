#!/bin/sh
# entrypoint.sh — Container startup checks before launching Streamlit
set -e

OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
MAX_RETRIES=30

# ── Wait for Ollama to be reachable ──────────────────────────────────────────
echo "[startup] Waiting for Ollama at $OLLAMA_URL ..."
i=0
until curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -ge "$MAX_RETRIES" ]; then
    echo "[startup] ERROR: Ollama did not respond after $((MAX_RETRIES * 2))s. Is the ollama service running?"
    exit 1
  fi
  echo "[startup]   not ready yet (attempt $i/$MAX_RETRIES) — retrying in 2s..."
  sleep 2
done
echo "[startup] Ollama is ready."

# ── Check dictionary data ─────────────────────────────────────────────────────
MISSING=0

if [ ! -f "/app/data/processed/word_index.pkl" ]; then
  echo "[startup] WARN: data/processed/word_index.pkl not found — exact lookup disabled"
  MISSING=1
fi

if [ ! -f "/app/data/processed/symspell_dict.txt" ]; then
  echo "[startup] WARN: data/processed/symspell_dict.txt not found — using built-in dictionary"
  MISSING=1
fi

CHROMA_POPULATED=false
if [ -d "/app/chroma_db" ] && [ "$(ls -A /app/chroma_db 2>/dev/null)" ]; then
  CHROMA_POPULATED=true
fi

if [ "$CHROMA_POPULATED" = "false" ]; then
  echo "[startup] WARN: chroma_db/ is empty — semantic search disabled"
  MISSING=1
fi

if [ "$MISSING" -eq 1 ]; then
  echo ""
  echo "[startup] ┌────────────────────────────────────────────────────────────────────┐"
  echo "[startup] │  Dictionary data is missing. Run the ingestion pipeline:           │"
  echo "[startup] │                                                                    │"
  echo "[startup] │  docker compose run --rm app python ingestion/etl_urban_dict.py    │"
  echo "[startup] │  docker compose run --rm app python ingestion/etl_wordnet.py       │"
  echo "[startup] │  # optional: etl_wiktionary.py, etl_idioms.py                      │"
  echo "[startup] │  docker compose run --rm app python ingestion/unify_schema.py      │"
  echo "[startup] │  docker compose run --rm app python ingestion/embed_and_index.py   │"
  echo "[startup] │  docker compose run --rm app python ingestion/build_word_index.py  │"
  echo "[startup] │  docker compose run --rm app python ingestion/build_symspell_dict.py│"
  echo "[startup] │                                                                    │"
  echo "[startup] │  Starting in LLM-only mode until data is loaded.                   │"
  echo "[startup] └────────────────────────────────────────────────────────────────────┘"
  echo ""
fi

# If a command was passed (e.g. `python ingestion/...`), run it instead of Streamlit
if [ "$#" -gt 0 ]; then
  exec "$@"
fi

# ── Launch Streamlit ──────────────────────────────────────────────────────────
echo "[startup] Starting WalkingDict on port 8501..."
exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true
