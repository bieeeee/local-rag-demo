#!/usr/bin/env python3
"""Download selected National Park Service pages and convert them to Markdown.

This is a small, conservative corpus-ingestion utility for a local RAG demo.
It only accepts nps.gov URLs, respects robots.txt, keeps source provenance,
and saves raw HTML so generated Markdown can be reviewed and regenerated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import re
import sys
import time
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as to_markdown
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_USER_AGENT = (
    "local-rag-nps-ingester/0.1 "
    "(+https://github.com/REPLACE_WITH_YOUR_REPO; contact: REPLACE_WITH_YOUR_EMAIL)"
)
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_DELAY_SECONDS = 2.0
MIN_WORDS = 80

STOP_HEADINGS = {
    "you might also like",
    "related content",
    "explore more",
    "more information",
    "tags",
}

REMOVE_SELECTORS = [
    "nav",
    "footer",
    "aside",
    "form",
    "button",
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "iframe",
    "object",
    "embed",
    "video",
    "audio",
    "picture",
    "figure",
    "[role='navigation']",
    "[aria-label*='breadcrumb' i]",
    "[class*='breadcrumb' i]",
    "[id*='breadcrumb' i]",
    "[class*='social' i]",
    "[id*='social' i]",
    "[class*='share' i]",
    "[id*='share' i]",
    "[class*='related' i]",
    "[id*='related' i]",
    "[class*='tags' i]",
    "[id*='tags' i]",
    "[class*='on-this-page' i]",
    "[id*='on-this-page' i]",
]

REVIEW_PATTERNS = {
    "copyright symbol": re.compile(r"©|\bcopyright\b", re.I),
    "third-party credit": re.compile(
        r"\b(?:photo|image|illustration|graphic|map|content)\s+(?:credit|courtesy|by)\b",
        re.I,
    ),
    "permission notice": re.compile(r"\bused with permission\b", re.I),
}


@dataclass
class Source:
    url: str
    output: str
    category: str
    title: str | None = None
    content_selector: str | None = None
    remove_selectors: list[str] = field(default_factory=list)
    stop_headings: list[str] = field(default_factory=list)
    enabled: bool = True


class ScrapeError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape curated NPS pages into reviewable Markdown files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/sources.yaml"),
        help="YAML source list (default: scripts/sources.yaml)",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Markdown output root (default: docs)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("raw/nps"),
        help="Raw HTML output root (default: raw/nps)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/nps-source-manifest.yaml"),
        help="Generated source manifest path",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="Identify your project and provide a real contact address",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="Minimum delay between requests in seconds (default: 2.0)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-raw", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_sources(path: Path) -> list[Source]:
    if not path.exists():
        raise ScrapeError(f"Config file not found: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    groups = config.get("sources")
    if not isinstance(groups, list):
        raise ScrapeError("Config must contain a top-level 'sources' list")

    nps_group = next(
        (g for g in groups if g.get("name") == "nps"),
        None,
    )

    if nps_group is None:
        raise ScrapeError("No source named 'nps' found in config")

    rows = nps_group.get("pages")
    if not isinstance(rows, list):
        raise ScrapeError("'pages' must be a list")

    sources: list[Source] = []

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ScrapeError(f"pages[{index}] must be a mapping")

        try:
            source = Source(**row)
        except TypeError as exc:
            raise ScrapeError(f"Invalid pages[{index}]: {exc}") from exc

        if source.enabled:
            validate_nps_url(source.url)
            validate_relative_output(source.output)
            sources.append(source)

    return sources


def validate_nps_url(url: str) -> None:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        raise ScrapeError(f"Only HTTPS URLs are allowed: {url}")
    if hostname != "nps.gov" and not hostname.endswith(".nps.gov"):
        raise ScrapeError(f"Only nps.gov URLs are allowed: {url}")


def validate_relative_output(output: str) -> None:
    path = Path(output)
    if path.is_absolute() or ".." in path.parts:
        raise ScrapeError(f"Output must be a safe relative path: {output}")
    if path.suffix.lower() != ".md":
        raise ScrapeError(f"Output must end with .md: {output}")


def build_session(user_agent: str) -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    session.mount("https://", adapter)
    return session


def get_robots_parser(
    session: requests.Session,
    url: str,
    cache: dict[str, urllib.robotparser.RobotFileParser],
) -> urllib.robotparser.RobotFileParser:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin in cache:
        return cache[origin]

    robots_url = f"{origin}/robots.txt"
    response = session.get(robots_url, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()

    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(response.text.splitlines())
    cache[origin] = parser
    return parser


def download_html(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if "html" not in content_type:
        raise ScrapeError(f"Expected HTML but received {content_type or 'unknown'}: {url}")
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def find_title(soup: BeautifulSoup, configured_title: str | None) -> str:
    if configured_title:
        return configured_title.strip()
    h1 = soup.find("h1")
    if h1:
        title = normalize_inline_text(h1.get_text(" ", strip=True))
        if title:
            return title
    if soup.title:
        title = normalize_inline_text(soup.title.get_text(" ", strip=True))
        title = re.sub(r"\s*\(U\.S\. National Park Service\)\s*$", "", title)
        if title:
            return title
    raise ScrapeError("Could not determine page title")


def find_updated_at(soup: BeautifulSoup) -> str | None:
    text = soup.get_text(" ", strip=True)
    match = re.search(r"Last updated:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", text, re.I)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%B %d, %Y").date().isoformat()
    except ValueError:
        return None


def choose_content_root(soup: BeautifulSoup, selector: str | None) -> Tag:
    if selector:
        selected = soup.select_one(selector)
        if not isinstance(selected, Tag):
            raise ScrapeError(f"Configured content_selector matched nothing: {selector}")
        return selected

    main = soup.find("main") or soup.select_one("[role='main']") or soup.find("article")
    if not isinstance(main, Tag):
        raise ScrapeError("Could not find <main>, [role=main], or <article>")

    h1 = main.find("h1")
    if not isinstance(h1, Tag):
        return main

    candidate: Tag = h1
    while isinstance(candidate.parent, Tag) and candidate.parent is not main:
        candidate = candidate.parent
        if word_count(candidate.get_text(" ", strip=True)) >= MIN_WORDS:
            return candidate
    return main


def clean_content(root: Tag, page_url: str, extra_selectors: list[str]) -> Tag:
    cleaned = copy.deepcopy(root)

    for selector in [*REMOVE_SELECTORS, *extra_selectors]:
        try:
            matches = cleaned.select(selector)
        except Exception as exc:
            raise ScrapeError(f"Invalid remove selector '{selector}': {exc}") from exc
        for element in matches:
            element.decompose()

    for image in cleaned.find_all("img"):
        image.decompose()

    for anchor in cleaned.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if href.startswith(("javascript:", "data:")):
            anchor.unwrap()
        else:
            anchor["href"] = urljoin(page_url, href)

    for element in cleaned.find_all(True):
        for attribute in list(element.attrs):
            if attribute not in {"href", "title"}:
                del element.attrs[attribute]

    return cleaned


def html_to_markdown(root: Tag, title: str, extra_stop_headings: list[str]) -> str:
    markdown = to_markdown(
        str(root),
        heading_style="ATX",
        bullets="-",
        strong_em_symbol="*",
    )
    markdown = normalize_markdown(markdown)

    lines = markdown.splitlines()
    title_heading = f"# {title}".casefold()
    first_title_index = next(
        (index for index, line in enumerate(lines) if line.strip().casefold() == title_heading),
        None,
    )
    if first_title_index is not None:
        lines = lines[first_title_index + 1 :]

    stops = {item.casefold() for item in STOP_HEADINGS}
    stops.update(item.strip().casefold() for item in extra_stop_headings)

    kept: list[str] = []
    for line in lines:
        heading_match = re.match(r"^#{1,6}\s+(.+?)\s*$", line.strip())
        if heading_match:
            heading_text = normalize_inline_text(heading_match.group(1)).casefold()
            if heading_text in stops:
                break
        if re.match(r"^Last updated:\s*", line.strip(), re.I):
            continue
        if line.strip() in {"Article", "Contact Us", "On This Page Navigation"}:
            continue
        kept.append(line)

    body = normalize_markdown("\n".join(kept)).strip()
    if word_count(body) < MIN_WORDS:
        raise ScrapeError(
            f"Extracted only {word_count(body)} words; inspect the page or set content_selector"
        )
    return f"# {title}\n\n{body}\n"


def normalize_inline_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_markdown(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"^\s*\[?Image[^\n]*\]?\s*$", "", value, flags=re.I | re.M)
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", value)
    return value.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’-]+\b", text, flags=re.UNICODE))


def find_review_flags(root: Tag) -> list[str]:
    text = root.get_text(" ", strip=True)
    flags = [label for label, pattern in REVIEW_PATTERNS.items() if pattern.search(text)]
    return sorted(flags)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_document(
    *,
    title: str,
    source: Source,
    updated_at: str | None,
    retrieved_at: str,
    markdown_body: str,
    raw_sha256: str,
    review_flags: list[str],
) -> str:
    metadata: dict[str, Any] = {
        "title": title,
        "source": "nps",
        "publisher": "National Park Service",
        "source_url": source.url,
        "category": source.category,
        "updated_at": updated_at,
        "retrieved_at": retrieved_at,
        "content_format": "adapted-html-to-markdown",
        "reuse_status": "manual-review-required",
        "source_html_sha256": raw_sha256,
        "modifications": [
            "Converted HTML to Markdown",
            "Removed navigation, branding, images, media, and unrelated page elements",
            "Converted relative hyperlinks to absolute URLs",
        ],
    }
    if review_flags:
        metadata["review_flags"] = review_flags

    frontmatter = yaml.safe_dump(
        metadata,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).strip()
    return f"---\n{frontmatter}\n---\n\n{markdown_body}"


def save_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notice": (
            "Generated files require manual review for extraction accuracy and "
            "third-party rights before publication."
        ),
        "sources": records,
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def scrape_source(
    *,
    source: Source,
    session: requests.Session,
    docs_dir: Path,
    raw_dir: Path,
    save_raw: bool,
    overwrite: bool,
    dry_run: bool,
    retrieved_at: str,
) -> dict[str, Any]:
    output_path = docs_dir / source.output
    raw_path = raw_dir / Path(source.output).with_suffix(".html")

    if output_path.exists() and not overwrite:
        print(f"SKIP  {output_path} already exists")
        return {
            "url": source.url,
            "output": output_path.as_posix(),
            "status": "skipped-existing",
        }

    print(f"GET   {source.url}")
    html = download_html(session, source.url)
    soup = BeautifulSoup(html, "html.parser")

    title = find_title(soup, source.title)
    updated_at = find_updated_at(soup)
    root = choose_content_root(soup, source.content_selector)
    cleaned = clean_content(root, source.url, source.remove_selectors)
    review_flags = find_review_flags(cleaned)
    markdown_body = html_to_markdown(cleaned, title, source.stop_headings)
    document = render_document(
        title=title,
        source=source,
        updated_at=updated_at,
        retrieved_at=retrieved_at,
        markdown_body=markdown_body,
        raw_sha256=sha256_text(html),
        review_flags=review_flags,
    )

    print(
        f"WRITE {output_path} "
        f"({word_count(markdown_body)} words"
        f"{', review: ' + ', '.join(review_flags) if review_flags else ''})"
    )
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(document, encoding="utf-8")
        if save_raw:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(html, encoding="utf-8")

    return {
        "url": source.url,
        "output": output_path.as_posix(),
        "raw_html": raw_path.as_posix() if save_raw else None,
        "title": title,
        "category": source.category,
        "source_updated_at": updated_at,
        "retrieved_at": retrieved_at,
        "word_count": word_count(markdown_body),
        "review_flags": review_flags,
        "status": "dry-run" if dry_run else "written",
    }


def main() -> int:
    args = parse_args()
    if "REPLACE_WITH_" in args.user_agent:
        print(
            "ERROR Replace the placeholder repository and email in --user-agent "
            "or DEFAULT_USER_AGENT before scraping.",
            file=sys.stderr,
        )
        return 2

    try:
        sources = load_sources(args.config)
    except ScrapeError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    if not sources:
        print("No enabled sources found.")
        return 0

    session = build_session(args.user_agent)
    robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
    retrieved_at = datetime.now(timezone.utc).date().isoformat()
    records: list[dict[str, Any]] = []
    failures = 0

    for index, source in enumerate(sources):
        try:
            robots = get_robots_parser(session, source.url, robots_cache)
            if not robots.can_fetch(args.user_agent, source.url):
                raise ScrapeError(f"robots.txt disallows this URL: {source.url}")

            records.append(
                scrape_source(
                    source=source,
                    session=session,
                    docs_dir=args.docs_dir,
                    raw_dir=args.raw_dir,
                    save_raw=not args.no_raw,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    retrieved_at=retrieved_at,
                )
            )
        except (requests.RequestException, ScrapeError) as exc:
            failures += 1
            print(f"FAIL  {source.url}: {exc}", file=sys.stderr)
            records.append(
                {
                    "url": source.url,
                    "output": str(args.docs_dir / source.output),
                    "status": "failed",
                    "error": str(exc),
                }
            )

        if index < len(sources) - 1:
            time.sleep(max(args.delay, 0.0))

    if not args.dry_run:
        save_manifest(args.manifest, records)
        print(f"MANIFEST {args.manifest}")

    print(f"Done: {len(sources) - failures} succeeded, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
