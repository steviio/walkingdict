# WalkingDict

Look up any word, slang, or idiom and get explanations that actually make sense to *you*. Powered by a local LLM (Ollama) and a multi-source dictionary corpus (Urban Dictionary, WordNet, idioms; Wiktionary optional).

Read the full write-up: [Project Report](https://drive.google.com/drive/folders/1xVVet956cnOGJiRxlmk6EEYR4yCoQDef?usp=sharing)

## How it works

1. You type a word, phrase, or slang term.
2. A 3-layer retrieval pipeline runs: **exact lookup → fuzzy/spell correction → semantic vector search**.
3. The retrieved entries are sent to a local LLM along with your proficiency level, interests, and preferred explanation style.
4. You get a personalized, streamed explanation — no cloud APIs, no data leaving your machine.

Alongside the main lookup panel, the UI shows a daily **Word of the Day**, a **recent-search** history, and **bookmarks** — all persisted to local JSON under `data/user/`.

## Requirements

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose (included with Docker Desktop)
- **NVIDIA GPU + NVIDIA Container Toolkit** — `docker-compose.yml` reserves a GPU for the Ollama container. Remove the `deploy.resources` block if you want to run CPU-only.

Ollama, the LLM / embedding models, and all Python dependencies are installed inside the containers.

## Quickstart

```bash
git clone https://github.com/your-username/walkingdict.git
cd walkingdict
docker compose up --build
```

On first run, the `ollama-init` service pulls `qwen3.5:9b` (LLM) and `nomic-embed-text` (embeddings) — this takes several minutes and several GB of disk. Models are stored in the `ollama_data` named volume and persist across restarts.

Once startup finishes, open **http://localhost:8501**.

## Loading dictionary data (first time only)

The repo ships without pre-built corpus or vector index. With the containers running, open a second terminal and run the ingestion pipeline in order, remember to use `sudo` if you didn't set up Docker permissions for your user:

```bash
# 1. Parse raw sources into per-source JSONL
#    etl_wordnet auto-downloads the Open English WordNet 2025 release on first run
#    (override with WORDNET_URL; use --skip-download for air-gapped runs).
docker compose run --rm app python -m ingestion.etl_urban_dict
docker compose run --rm app python -m ingestion.etl_wordnet
# Optional sources:
#   Wiktionary — large (multi-GB dump). See ingestion/etl_wiktionary.py for the download link.
#   Idioms    — requires a LIdioms/MAGPIE/EPIE JSONL or a phrase/definition TSV in data/raw/idioms/.
# docker compose run --rm app python -m ingestion.etl_wiktionary
# docker compose run --rm app python -m ingestion.etl_idioms

# 2. Merge into the unified corpus
docker compose run --rm app python -m ingestion.unify_schema

# 3. Embed and index into ChromaDB (slowest step — progress bar shown)
docker compose run --rm app python -m ingestion.embed_and_index

# 4. Build the exact-lookup index (reads from ChromaDB) and SymSpell frequency dict
docker compose run --rm app python -m ingestion.build_word_index
docker compose run --rm app python -m ingestion.build_symspell_dict
```

Processed artefacts land in `data/processed/` and `chroma_db/` on your host, so you only need to run this once. The app starts in **LLM-only mode** until these files exist — `entrypoint.sh` prints a warning listing what's missing.

### Why isn't this baked into `docker compose up`?

Ingestion is intentionally **user-initiated** rather than part of container startup:

- **Only WordNet auto-downloads.** `etl_wordnet` pulls the Open English WordNet 2025 release from GitHub on first run. The other sources are user-managed: **Urban Dictionary** must be downloaded manually from Kaggle (licence + API credentials), **Wiktionary** is optional and too large (multi-GB) to fetch on behalf of every user, and **Idioms** is left opt-in so you can pick your own corpus.
- **Heavy, one-time cost.** `embed_and_index` can take tens of minutes — you shouldn't have to wait for it every time the container boots, and a failed embed shouldn't block the app from starting.
- **No silent network fetches on boot.** The WordNet auto-download fires only when you explicitly run `ingestion.etl_wordnet`, keeping `docker compose up` offline-safe.
- **Re-runnability.** Each step is a plain `python -m` invocation — easy to re-run one stage (e.g. after updating the corpus) without rebuilding the image.

If any required artefact is missing, `entrypoint.sh` falls back to LLM-only mode and prints the exact commands to fix it.

### Data sources and attribution

| Source | How it's obtained | Licence |
|---|---|---|
| **Open English WordNet** (2025) | Auto-downloaded by `etl_wordnet.py` from [globalwordnet/english-wordnet](https://github.com/globalwordnet/english-wordnet/releases). Override via `WORDNET_URL`, or use `--skip-download` for air-gapped runs. | CC BY 4.0 |
| **Urban Dictionary** | Manual: download [therohk/urban-dictionary-words-dataset](https://www.kaggle.com/datasets/therohk/urban-dictionary-words-dataset) from Kaggle and place the CSV in `data/raw/urban_dict/`. | Kaggle dataset terms |
| **Wiktionary** (optional, via [kaikki.org](https://kaikki.org/dictionary/English/)) | Manual: third-party pre-processed JSONL dump of English Wiktionary; download link in `ingestion/etl_wiktionary.py`. Skipped by default because the dump is multi-GB. | Wiktionary content: CC BY-SA 4.0 |
| **Idioms** (optional) | Manual: bring your own LIdioms / MAGPIE / EPIE JSON(L) or a `phrase,definition,example` CSV/TSV into `data/raw/idioms/`. | Varies by source |

## Stopping and restarting

```bash
# Stop (models and data are preserved)
docker compose down

# Restart
docker compose up
```

Ollama models live in the `ollama_data` named volume. Dictionary data lives in your local `data/` and `chroma_db/` directories.

## Configuration

All tunables are in [config.py](config.py). Most commonly changed:

| Setting | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `qwen3.5:9b` | Swap to a smaller variant (e.g. `qwen3:0.6b`) for low-memory machines |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `LLM_TEMPERATURE` | `0.7` | Higher = more creative explanations |
| `LLM_MAX_TOKENS` | `1024` | Response length cap |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `MIN_VECTOR_SCORE` | `0.6` | Chroma distance threshold (lower = stricter) |
| `MAX_EDIT_DISTANCE` | `2` | SymSpell max edit distance for spell correction |
| `FUZZY_SCORE_CUTOFF` | `80` | RapidFuzz minimum score (0–100) |

To switch models, update `LLM_MODEL` in `config.py` (and the `ollama-init` pull command in [docker-compose.yml](docker-compose.yml)), then pull the new model:

```bash
docker compose exec ollama ollama pull <model-name>
```

User preference defaults (`DEFAULT_PROFICIENCY`, `DEFAULT_INTERESTS`, `INTEREST_OPTIONS`) are also in `config.py`.

## Local development (without Docker)

If you prefer to run outside Docker, you need [Ollama](https://ollama.com) installed locally.

```bash
# Pull models
ollama pull qwen3.5:9b
ollama pull nomic-embed-text

# Install Python dependencies (Python 3.12 recommended)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run ingestion (same order as above), then:
streamlit run app.py
```

## Project structure

```
walkingdict/
├── app.py              # Streamlit entry point
├── config.py           # All tunables in one place
├── Dockerfile          # App image
├── docker-compose.yml  # App + Ollama + model puller
├── entrypoint.sh       # Waits for Ollama, warns on missing data, launches Streamlit
├── requirements.txt    # Python dependencies
├── ingestion/          # ETL pipeline: etl_{urban_dict,wordnet,wiktionary,idioms}.py
│                       #   → unify_schema → embed_and_index
│                       #   → build_word_index + build_symspell_dict
├── rag/                # 3-layer retrieval: retriever, spell_correct, query_pipeline,
│                       #   prompt_builder, generator
├── ui/                 # Streamlit components: sidebar, main_panel, right_column,
│                       #   extras, storage
├── scripts/            # run_eda.py, run_eval.py — thin CLI wrappers over the notebooks
├── notebooks/          # eda.ipynb, eval.ipynb (corpus + retrieval eval)
├── data/
│   ├── raw/            # Source dumps (gitignored)
│   ├── processed/      # unified_corpus.jsonl, word_index.pkl, symspell_dict.txt, wotd_cache.json
│   ├── eval/           # Evaluation query sets and reports
│   └── user/           # history.json, bookmarks.json
└── chroma_db/          # Vector store (gitignored)
```

## Acknowledgements

WalkingDict stands on the shoulders of a lot of open work:

- **[Open English WordNet](https://github.com/globalwordnet/english-wordnet)** — structured lexical backbone (CC BY 4.0).
- **[Urban Dictionary dataset](https://www.kaggle.com/datasets/therohk/urban-dictionary-words-dataset)** by therohk on Kaggle — slang coverage.
- **[Wiktionary](https://en.wiktionary.org/)** via [kaikki.org](https://kaikki.org/dictionary/English/) pre-processed dumps — etymology and cross-lingual definitions (CC BY-SA 4.0).
- **[LIdioms](https://github.com/dice-group/LIdioms) / [MAGPIE](https://github.com/hslh/magpie-corpus) / [EPIE](https://github.com/prateeksaxena2809/EPIE_Corpus)** — idiom corpora.
- **[Ollama](https://ollama.com/)** for local LLM + embedding serving, with [Qwen](https://github.com/QwenLM/Qwen) as the default generator and [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1) for embeddings.
- **[ChromaDB](https://www.trychroma.com/)** for vector storage, **[SymSpell](https://github.com/wolfgarbe/SymSpell)** + **[RapidFuzz](https://github.com/rapidfuzz/RapidFuzz)** for spell correction and fuzzy matching, and **[Streamlit](https://streamlit.io/)** for the UI.

Thanks to the maintainers of all of the above — this project is essentially a thin RAG glue layer over their work.
