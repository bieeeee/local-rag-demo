# Offline Outdoor RAG

A local, fully offline retrieval-augmented generation (RAG) demo. It answers
outdoor safety questions (hiking, camping, weather, wildlife, water,
emergencies) using a small curated corpus of adapted public documents from the
U.S. National Park Service (NPS) and the Centers for Disease Control and
Prevention (CDC).

Everything runs on your machine:

- **Embeddings** — `intfloat/multilingual-e5-base` via sentence-transformers.
- **Vector store** — Chroma, persisted to `chroma_db/`.
- **LLM** — a local model served by [Ollama](https://ollama.com) (default
  `gemma3:4b`).
- **Web UI** — a single-page chat interface served by FastAPI.

No document text or question ever leaves your computer.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/download)
- 16 GB RAM or more recommended

## Setup

### 1. Install Ollama and pull the model

```bash
# After installing Ollama:
ollama pull gemma3:4b
```

Make sure the Ollama server is running (`ollama serve`, or the desktop app).

### 2. Create the Python environment and install dependencies

```bash
python setup.py        # or: python3 setup.py
```

This creates a `.venv/` virtual environment and installs everything from
`requirements.txt`. Only the standard library is needed to run `setup.py`
itself.

## Run

```bash
python serve.py        # or: python3 serve.py
```

`serve.py` re-executes itself inside `.venv`, starts the FastAPI server on
<http://127.0.0.1:8000>, and opens your browser once the server is ready. The
first run can take a while because the embedding model is downloaded and the
vector database is built.

Stop the server with `Ctrl+C`.

## How it works

1. **Indexing** — Markdown documents under `docs/` are split into chunks,
   embedded, and stored in Chroma (`chroma_db/`). The corpus is only re-indexed
   when a source file changes, so subsequent startups are fast.
2. **Retrieval** — your question is embedded and the nearest chunks are
   fetched. An optional category (camping / water / wildlife / weather /
   emergencies) softly re-ranks results toward that topic.
3. **Generation** — the retrieved chunks and recent question history are built
   into a prompt and sent to the local Ollama model, which answers strictly
   from the provided documents.

## Project layout

```text
setup.py                  # Create .venv and install dependencies
serve.py                  # Launch the FastAPI server and open the browser
requirements.txt          # Runtime Python dependencies
src/
  server.py               # FastAPI app: static UI + /api/chat
  qa.py                   # Retrieval -> generation core (calls Ollama)
  indexing.py             # Build/load the Chroma vector DB, change detection
  embeddings.py           # multilingual-e5 embedding wrapper (query:/passage:)
  loader.py               # Load and chunk docs/ Markdown into Documents
  prompt.py               # Build the LLM prompt from chunks + history
  config.py               # Paths and model name
static/index.html         # Single-page chat UI
docs/                     # Indexed Markdown corpus (grouped by category)
scripts/                  # Corpus-generation scrapers (see scripts/README.md)
chroma_db/                # Persisted vector store (generated)
```

## Notes

- Add or edit `.md` files under `docs/` and the corpus is automatically
  re-indexed on the next startup.
- To change the LLM, pull another model with Ollama and update `MODEL` in
  [src/config.py](src/config.py). The model name must match one you have
  already pulled.
- The documents in `docs/` are adapted from public agency material and are for
  demonstration only. See [NOTICE.md](NOTICE.md) for provenance and reuse
  details, and [scripts/README.md](scripts/README.md) for how the corpus is
  generated.
