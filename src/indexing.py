"""
Preprocessing step that splits source documents into chunks, converts them
to vectors, and stores them in a vector DB.

This module handles the orchestration of "when to re-index (change detection)"
and "how to build/load the vector DB."

Terminology:
  - corpus: the full set of source text to be indexed.
  - chunk: a piece of the corpus split into a retrieval unit. The smallest
    unit of embedding/retrieval.
  - embedding: text converted into a high-dimensional vector that captures meaning.
  - vector store (vector DB): a DB that stores embeddings and provides
    similarity search. Here, Chroma is used, persisted to local disk (DB_DIR).
"""

import os
import glob
import shutil
from langchain_community.vectorstores import Chroma
from config import DOCS_DIR, DB_DIR
from loader import load_and_chunk_docs


def iter_doc_source_files() -> list[str]:
    """Returns the list of indexing input files under docs/.

    The actual embedding targets are Markdown documents, but generated
    manifests like source-manifest.yaml also describe the corpus's
    provenance/review status, so they're included in change detection too.
    """
    patterns = ("**/*.md", "**/*.yaml", "**/*.yml")
    files = {
        path
        for pattern in patterns
        for path in glob.glob(str(DOCS_DIR / pattern), recursive=True)
        if os.path.isfile(path)
    }
    return sorted(files)


def corpus_newer_than_db() -> bool:
    """
    Returns True if the corpus is newer than the existing vector DB (= re-indexing needed).

    Rebuilding embeddings from scratch every time is expensive, so we only
    re-index when source files have changed. The criterion is file
    modification time (mtime): if any source file was modified later than
    the DB's creation time, it's a re-indexing candidate.

    Return rules:
    - The DB directory itself doesn't exist   -> True (initial build)
    - Any docs/**/*.md has mtime > DB mtime   -> True (document body/front matter changed)
    - Any docs/**/*.yaml, **/*.yml likewise   -> True (source manifest changed)
    - Otherwise                               -> False (reuse cached DB)
    """
    # If the DB doesn't exist yet, there's nothing to compare against, so build unconditionally.
    if not os.path.exists(DB_DIR):
        return True

    # Use the DB directory's modification time as the baseline.
    db_mtime = os.path.getmtime(DB_DIR)

    def _newer(path: str) -> bool:
        try:
            return os.path.getmtime(path) > db_mtime
        except OSError:
            # There can be a race condition where a file is deleted after
            # glob lists it but before getmtime is called. Treat it as absent.
            return False

    # Document corpus: recursively scan Markdown and source manifests under docs/
    for f in iter_doc_source_files():
        if _newer(f):
            return True

    return False



def build_or_load_db(embeddings) -> Chroma:
    """Rebuilds the vector DB (if needed) or loads the existing one, and returns it.

    Indexes the outdoor safety document corpus under docs/ into a single
    Chroma collection. Each document's YAML front matter is reflected as
    searchable metadata (source, publisher, category, etc.) at the loader stage.

    embeddings: an embedding object (E5Embeddings) that converts text to vectors.
        - On the rebuild path, it's used to embed chunks before storing them in the DB.
        - On the load path, it's attached to the DB as embedding_function so
          subsequent queries can be converted into the same vector space.
        Indexing and querying must use the same embedding model for search to be meaningful.
    """
    if corpus_newer_than_db():
        # --- Rebuild path: source changed or DB doesn't exist ---
        print("Document corpus change detected — rebuilding vector DB...")

        # Partial updates are tricky, so wipe the old DB entirely and rebuild.
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)

        # Load and chunk documents. Returns a list of embeddable Document chunks.
        chunks = load_and_chunk_docs()

        print(f"Indexing {len(chunks)} document chunks")

        # Embed the chunks, store them in Chroma, and persist to DB_DIR.
        # (Providing persist_directory allows reuse without re-embedding next run.)
        db = Chroma.from_documents(chunks, embeddings, persist_directory=str(DB_DIR))
    else:
        # --- Load path: source unchanged, so open the existing DB from disk as-is ---
        # Queries need to be converted to vectors, so attach embeddings via embedding_function.
        db = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)

    print("Vector DB ready")
    return db
