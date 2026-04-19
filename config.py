"""Central configuration for WalkingDict — all tunables in one place."""

import os
from pathlib import Path

# ── Directory layout ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
USER_DIR = DATA_DIR / "user"

# Subdirectories for raw sources
RAW_WIKTIONARY_DIR = RAW_DIR / "wiktionary"
RAW_URBAN_DICT_DIR = RAW_DIR / "urban_dict"
RAW_WORDNET_DIR = RAW_DIR / "wordnet"
RAW_IDIOMS_DIR = RAW_DIR / "idioms"

# Processed artefacts
CORPUS_FILE = PROCESSED_DIR / "unified_corpus.jsonl"
WORD_INDEX_FILE = PROCESSED_DIR / "word_index.pkl"
SYMSPELL_DICT_FILE = PROCESSED_DIR / "symspell_dict.txt"  # plain frequency dict

# User data
HISTORY_FILE = USER_DIR / "history.json"
BOOKMARKS_FILE = USER_DIR / "bookmarks.json"
MAX_HISTORY = 100  # max recent searches stored on disk and in session

# Vector store
CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_COLLECTION = "walkingdict"

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Model names — swap LLM_MODEL to a larger variant on GCP
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3.5:9b"

# ── Query pipeline ─────────────────────────────────────────────────────────────
TOP_K_RESULTS = 5           # how many chunks to retrieve
MAX_EDIT_DISTANCE = 2       # SymSpell max edit distance
FUZZY_SCORE_CUTOFF = 80     # RapidFuzz min score (0–100)
MIN_VECTOR_SCORE = 0.6      # Chroma distance threshold (lower = more similar)

# ── LLM generation ─────────────────────────────────────────────────────────────
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1024
# Using larger setting
# LLM_MAX_TOKENS = 2048

# ── User-facing defaults ───────────────────────────────────────────────────────
DEFAULT_PROFICIENCY = "intermediate"   # beginner | intermediate | advanced
DEFAULT_INTERESTS: list[str] = ["everyday life", "technology"]
DEFAULT_LANGUAGE = "English"

PROFICIENCY_LEVELS = ["beginner", "intermediate", "advanced"]

INTEREST_OPTIONS = [
    "everyday life",
    "technology",
    "cooking",
    "sports",
    "music",
    "science",
    "business",
    "travel",
    "literature",
    "internet culture",
]

# ── Word of the Day ────────────────────────────────────────────────────────────
WOTD_CACHE_FILE = PROCESSED_DIR / "wotd_cache.json"
