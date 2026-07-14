"""
Retrieval -> generation core.

Takes a single question and (1) retrieves nearby chunks from the vector DB
, then (2) builds a prompt from those chunks and sends it to the
local LLM (Ollama) to get an answer.

This module contains only the pure logic for "a single query." The
presentation layer, such as HTTP responses (server.py), is the caller's
responsibility.
"""

import json
import os
import requests
from config import MODEL
from prompt import build_prompt

# Ollama local server's generation API
OLLAMA_URL = "http://localhost:11434/api/generate"

# Which LLM backend generates the answer:
#   "ollama" (default) -> local Ollama server, fully offline. Used in local dev.
#   "gemini"           -> Google Gemini API (free tier). Used for the hosted
#                         deployment, where running a local model isn't practical.
# The retrieval/re-ranking below is identical either way; only generation differs.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()

# Gemini model for the hosted backend. Overridable via env without a code change.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _generate_ollama(prompt):
    """Generate an answer with the local Ollama server.

    Never raises: connection/model errors are turned into a human-readable
    string so the caller (and UI) can show them like any other answer.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
        )
        data = json.loads(response.text)
        # A normal response has "response"; a model-side error has "error".
        if "response" in data:
            return data["response"]
        if "error" in data:
            return f"[Ollama error]: {data['error']}"
        return f"[Unknown response]: {response.text[:200]}"
    except requests.exceptions.RequestException as e:
        # Server not running, connection dropped, etc.
        return f"[Ollama connection failed]: {e}"


def _generate_gemini(prompt):
    """Generate an answer with the Google Gemini API (free tier).

    The API key is read from the GEMINI_API_KEY environment variable.
    Like the Ollama path, this returns errors as a string rather than raising.
    """
    try:
        from google import genai
    except ImportError:
        return "[Gemini backend not installed]: run 'pip install google-genai'."

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[Gemini backend not configured]: set GEMINI_API_KEY."

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        # .text concatenates the text parts of the response.
        return response.text or ""
    except Exception as e:
        # The SDK raises various API/network errors; surface them as a message.
        return f"[Gemini API error]: {e}"


def _generate(prompt):
    """Dispatch to the configured backend (Ollama or Gemini)."""
    if LLM_BACKEND == "gemini":
        return _generate_gemini(prompt)
    return _generate_ollama(prompt)


def ask(db, history, search_text, prompt_query=None, k=8, category=None):
    """Returns (answer, list of retrieved chunks) for a single question.

    Retrieval and generation want different text shapes, so the input is
    split in two:
      - search_text: for embedding search. Natural language close to the
        document's tone (no label noise).
      - prompt_query: the question/context to put in the prompt. Can be a
        labeled structured block.
    If prompt_query isn't given, search_text is used for the prompt too
    (backward compatible with callers that only have a single natural
    language question, like eval scripts).

    db: the vector store (Chroma) used for retrieval.
    history: conversation history in the form [{"q": ..., "a": ...}]. Used
        only for prompt construction, not mixed into the search query (since
        that would blur the meaning of the question itself).
    k: number of chunks to retrieve. Adjust this if retrieval quality is off.
    category: topic category (camping/water/wildlife/weather/emergencies) or
        None. If given, re-ranks to prefer chunks of that category (a soft
        weight, not a hard filter, so chunks from other categories remain if
        close enough). If None, searches across everything.

    Returns: (answer: str, results: list[Document]).
        The result chunks can be used by the caller for previews/source
        display. Even if answer generation fails, no exception is raised —
        a human-readable error string is put in answer instead.
    """
    if prompt_query is None:
        prompt_query = search_text

    # Fetch generously more candidates (3x k), then re-rank and keep only the top k.
    candidates = db.similarity_search_with_score(search_text, k=k * 3)

    # Retrieval: embed the natural language query and fetch the top k semantically nearest chunks.
    if category:
        def adjusted(pair):
            doc, dist = pair                       # dist: smaller means more similar
            chunk_category = doc.metadata.get("category")
            # Penalize if it differs from the requested category. Match/no-category is left as-is.
            return dist + (0.15 if chunk_category not in (category, None) else 0.0)
        candidates.sort(key=adjusted)

    results = [doc for doc, _ in candidates[:k]]

    prompt = build_prompt(prompt_query, results, history)

    # Generation: hand the prompt to the configured backend (local Ollama, or
    # the Gemini API for the hosted deployment). Both return the answer as a
    # string and never raise, so retrieval results are always returned.
    answer = _generate(prompt)

    return answer, results
