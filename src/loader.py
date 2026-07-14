"""
Reads Markdown under docs/ and turns it into a list of chunks.
Since a Markdown document's heading structure corresponds to semantic units,
it's first split on headings (heading-aware split), and only sections that
are still too long get a second pass split by character count. This way each
chunk maps to a single topic, improving retrieval accuracy.

Each chunk gets a source+title header and metadata embedded in it, enabling
source tracing of search results.
"""

import os
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from config import DOCS_DIR

def extract_field(text: str, field: str, fallback: str) -> str:
    """Extracts an arbitrary field value from YAML front matter (--- ... ---).

    Uses fallback if there's no front matter or the field is missing.
    This single function reads all front matter fields, including title.
    The field name is matched exactly at the start of a line (^field:), so
    it won't accidentally match a similar field like source_title.
    """
    m = re.search(rf'^---\s*\n.*?^{re.escape(field)}:\s*"?([^"\n]+?)"?\s*$.*?^---\s*\n',
                  text, re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else fallback


def source_label(path: str) -> str:
    """
    Builds a source label from the parent folder name relative to DOCS_DIR.

    docs/ is divided into folders by topic, so the folder a file belongs to
    is effectively the document's category. Also used as the fallback when
    front matter has no category.
    """
    rel = os.path.relpath(path, DOCS_DIR)
    parts = rel.split(os.sep)
    return "/".join(parts[:-1]) if len(parts) > 1 else "root"


# First-pass splitter: splits into sections at Markdown heading (#, ##, ###) boundaries.
# strip_headers=False -> keeps heading lines in the chunk body, so a chunk alone still gives context.
HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ],
    strip_headers=False,  # keep headings in the chunk text
)
# Second-pass (character-count) splitting only applies when a heading section
# exceeds this length. Set with margin above CHUNK_TARGET_SIZE so slightly
# long sections are left whole, and only truly long sections get split
# (reduces small fragment chunks).
SUBSPLIT_THRESHOLD = 1200

# Second-pass splitter: splits sections that exceed the threshold above,
# targeting CHUNK_TARGET_SIZE. chunk_overlap lets adjacent chunks share some
# content, easing context loss at boundaries. Separators are tried in order
# of priority: paragraph -> line -> sentence -> word -> (last resort) character.
CHUNK_TARGET_SIZE = 900
CHAR_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_TARGET_SIZE,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_and_chunk_docs():
    """Loads and chunks all Markdown under docs/, returning a list of Documents."""
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    all_chunks = []
    for doc in docs:
        path = doc.metadata["source"]
        filename = os.path.basename(path)
        title = extract_field(doc.page_content, "title", fallback=filename)
        src = source_label(path)
        # Topic category. Prefers front matter's category, falling back to the folder name (src).
        # Used downstream as a scope filter to narrow search by topic.
        category = extract_field(doc.page_content, "category", src)
        # Source institution family (nps/cdc etc). Taken as-is from front matter's source_type.
        source_type = extract_field(doc.page_content, "source_type", "unknown")
        # Extra metadata used for source attribution/citation.
        publisher = extract_field(doc.page_content, "publisher", "")
        source_url = extract_field(doc.page_content, "source_url", "")
        source_updated_at = extract_field(doc.page_content, "source_updated_at", "")
        scope = extract_field(doc.page_content, "scope", "general")

        # Front matter is just metadata, not searchable body text, so strip it before embedding.
        body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", doc.page_content,
                      count=1, flags=re.DOTALL)

        # First pass: split on headings
        header_chunks = HEADER_SPLITTER.split_text(body)

        for hc in header_chunks:
            # Second pass: only split further by character count if the section exceeds the threshold.
            sub_chunks = CHAR_SPLITTER.split_text(hc.page_content) \
                         if len(hc.page_content) > SUBSPLIT_THRESHOLD \
                         else [hc.page_content]

            crumbs = [
                str(value)
                for k in ("h1", "h2", "h3")
                if (value := hc.metadata.get(k))
            ]
            breadcrumb = " > ".join(crumbs)

            for sc in sub_chunks:
                # Embed a source+title header at the start of the body so it's carried into embedding/prompts.
                header_line = f"[Source: {src}/{filename} | {title}]"
                if breadcrumb:
                    header_line += f"\n[Section: {breadcrumb}]"
                # type(hc)(...) : creates a new chunk of the same Document type as the original.
                hc_copy = type(hc)(
                    page_content=f"{header_line}\n{sc}",
                    # Metadata for post-search filtering/source tracing (not embedded).
                    metadata={
                        **doc.metadata,
                        "title": title,
                        "source_label": src,
                        "filename": filename,
                        "category": category,
                        "source_type": source_type,
                        "source_kind": "doc",
                        "publisher": publisher,
                        "source_url": source_url,
                        "source_updated_at": source_updated_at,
                        "scope": scope,
                    },
                )
                all_chunks.append(hc_copy)

    print(f"Generated {len(all_chunks)} chunks")
    return all_chunks