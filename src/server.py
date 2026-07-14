"""
Web server entry point

On boot, prepares the embedding model and vector DB once, then serves queries
over HTTP endpoints. The retrieval->generation core lives in qa.ask(); this
file only adds the HTTP presentation layer on top.

  GET  /            -> static/index.html (chat UI)
  POST /api/chat    -> takes { query/search, context, category, history },
                       returns { answer, sources }
"""

from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from embeddings import E5Embeddings
from indexing import build_or_load_db
from qa import ask

# Location of static/index.html. static/ sits above this file (src/server.py).
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# The vector DB is heavy, so build it once at server startup and reuse it on
# every request. Kept in a module global, populated in lifespan.
state = {"db": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    embeddings = E5Embeddings("intfloat/multilingual-e5-base")
    state["db"] = build_or_load_db(embeddings)
    print("Web server ready — http://localhost:8000")
    yield


app = FastAPI(title="offline-outdoor-rag", lifespan=lifespan)


class ChatRequest(BaseModel):
    """Chat request body.

    A single query is enough for the basic case. If the client can build a
    natural-language search string and a situational description separately,
    it can send search/context alongside.

    query: the user's question. Used for both search and prompt if
        search/context are absent.
    search: for embedding search. One natural-language sentence close to the
        document's tone. e.g. "campfire safety before starting a fire in a campground".
    context: for the (generation) prompt. May include the user's situation,
        location cues, equipment, constraints, etc.
    category: optional topic (camping/water/wildlife/weather/emergencies) to
        narrow retrieval toward. Empty string means search across everything.
    history: recent conversation history [{"q", "a"}, ...]. The client holds
        the state and sends it with each request (the server is stateless).
    """
    query: str = ""
    search: str = ""
    context: str = ""
    category: str = ""
    history: list[dict] = Field(default_factory=list)


class Source(BaseModel):
    """Preview of a chunk retrieved as answer evidence."""
    filename: Optional[str]
    title: Optional[str] = None
    category: Optional[str] = None
    publisher: Optional[str] = None
    source_type: Optional[str] = None
    source_url: Optional[str] = None
    source_updated_at: Optional[str] = None
    preview: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Takes a single question and returns the answer plus retrieved source chunks.

    Search uses `search` if present, otherwise the first of query/context.
    The generation prompt uses `context` if present, otherwise query/search.
    Conversation history is not stored by the server; the client-sent history
    is reflected as-is.
    """
    search_text = (req.search or req.query or req.context).strip()
    prompt_query = (req.context or req.query or req.search).strip()

    if not search_text:
        raise HTTPException(
            status_code=400,
            detail="One of query or search must be non-empty.",
        )

    answer, results = ask(
        state["db"],
        req.history,
        search_text,
        prompt_query,
        category=req.category.strip() or None,
    )

    # Give the UI the same info as the console's '[retrieved chunks]' preview.
    sources = [
        Source(
            filename=r.metadata.get("filename"),
            title=r.metadata.get("title"),
            category=r.metadata.get("category"),
            publisher=r.metadata.get("publisher"),
            source_type=r.metadata.get("source_type"),
            source_url=r.metadata.get("source_url"),
            source_updated_at=r.metadata.get("source_updated_at"),
            preview=r.page_content[:200].replace("\n", " "),
        )
        for r in results
    ]
    return {"answer": answer, "sources": sources}


@app.get("/")
def index():
    """Serves the chat UI (static/index.html)."""
    return FileResponse(STATIC_DIR / "index.html")


# Serve any other static assets under /static.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
