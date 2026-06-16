from __future__ import annotations

import html
import json
import random
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus

import requests


USER_AGENT = "FastBPE/0.1 (+https://local.fastbpe)"

GUTENBERG_BOOKS: list[dict[str, str]] = [
    {"title": "Pride and Prejudice", "author": "Jane Austen", "url": "https://www.gutenberg.org/files/1342/1342-0.txt"},
    {"title": "A Tale of Two Cities", "author": "Charles Dickens", "url": "https://www.gutenberg.org/files/98/98-0.txt"},
    {"title": "Frankenstein", "author": "Mary Wollstonecraft Shelley", "url": "https://www.gutenberg.org/files/84/84-0.txt"},
    {"title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle", "url": "https://www.gutenberg.org/files/1661/1661-0.txt"},
]

CODE_URLS: list[dict[str, str]] = [
    {"repo": "pallets/flask", "path": "src/flask/app.py", "url": "https://raw.githubusercontent.com/pallets/flask/main/src/flask/app.py"},
    {"repo": "psf/requests", "path": "src/requests/sessions.py", "url": "https://raw.githubusercontent.com/psf/requests/main/src/requests/sessions.py"},
    {"repo": "tiangolo/fastapi", "path": "fastapi/routing.py", "url": "https://raw.githubusercontent.com/tiangolo/fastapi/master/fastapi/routing.py"},
    {"repo": "microsoft/TypeScript", "path": "src/compiler/parser.ts", "url": "https://raw.githubusercontent.com/microsoft/TypeScript/main/src/compiler/parser.ts"},
    {"repo": "facebook/react", "path": "packages/react/src/ReactHooks.js", "url": "https://raw.githubusercontent.com/facebook/react/main/packages/react/src/ReactHooks.js"},
    {"repo": "vercel/next.js", "path": "packages/next/src/server/base-server.ts", "url": "https://raw.githubusercontent.com/vercel/next.js/canary/packages/next/src/server/base-server.ts"},
    {"repo": "eslint/eslint", "path": "lib/linter/linter.js", "url": "https://raw.githubusercontent.com/eslint/eslint/main/lib/linter/linter.js"},
    {"repo": "prettier/prettier", "path": "src/language-js/parser-babel.js", "url": "https://raw.githubusercontent.com/prettier/prettier/main/src/language-js/parser-babel.js"},
]


@dataclass
class FetchedDocument:
    domain: str
    source: str
    source_id: str
    title: str
    text: str
    metadata: dict[str, object]


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _clean_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "document"


def _write_documents(documents: Iterable[FetchedDocument], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for index, document in enumerate(documents, start=1):
        suffix = {"english": ".txt", "code": ".py", "web": ".txt", "technical": ".md"}.get(document.domain, ".txt")
        path = output_dir / f"{index:04d}_{_slugify(document.title)}{suffix}"
        path.write_text(document.text, encoding="utf-8")
        written_paths.append(path)
    return written_paths


def _strip_gutenberg_boilerplate(text: str) -> str:
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
    ]
    for marker in start_markers:
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    for marker in end_markers:
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    return _clean_whitespace(text)


def fetch_english_documents(max_docs: int, seed: int = 13) -> list[FetchedDocument]:
    session = _session()
    rng = random.Random(seed)
    documents: list[FetchedDocument] = []
    per_book = max(1, (max_docs + len(GUTENBERG_BOOKS) - 1) // len(GUTENBERG_BOOKS))
    for book in GUTENBERG_BOOKS:
        try:
            response = session.get(book["url"], timeout=60)
            response.raise_for_status()
        except requests.RequestException:
            continue
        text = _strip_gutenberg_boilerplate(response.text)
        paragraphs = [_clean_whitespace(part) for part in re.split(r"\n\s*\n", text)]
        paragraphs = [part for part in paragraphs if len(part) >= 350]
        rng.shuffle(paragraphs)
        for idx, paragraph in enumerate(paragraphs[:per_book], start=1):
            documents.append(
                FetchedDocument(
                    domain="english",
                    source="project_gutenberg",
                    source_id=book["url"],
                    title=f"{book['title']} part {idx}",
                    text=paragraph,
                    metadata={"book_title": book["title"], "author": book["author"]},
                )
            )
            if len(documents) >= max_docs:
                return documents[:max_docs]
    rng.shuffle(documents)
    return documents[:max_docs]


def fetch_code_documents(max_docs: int, seed: int = 13) -> list[FetchedDocument]:
    session = _session()
    rng = random.Random(seed)
    urls = CODE_URLS[:]
    rng.shuffle(urls)
    documents: list[FetchedDocument] = []
    for item in urls[:max_docs]:
        try:
            response = session.get(item["url"], timeout=60)
            response.raise_for_status()
        except requests.RequestException:
            continue
        documents.append(
            FetchedDocument(
                domain="code",
                source="github_raw",
                source_id=item["url"],
                title=f"{item['repo']} {item['path']}",
                text=_clean_whitespace(response.text),
                metadata={"repo": item["repo"], "path": item["path"]},
            )
        )
    return documents


def _strip_html_tags(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<a [^>]+>", "", text)
    text = re.sub(r"</a>", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return _clean_whitespace(text)


def fetch_web_documents(max_docs: int, seed: int = 13) -> list[FetchedDocument]:
    session = _session()
    rng = random.Random(seed)
    response = session.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=60)
    response.raise_for_status()
    story_ids: list[int] = response.json()
    rng.shuffle(story_ids)
    documents: list[FetchedDocument] = []
    for story_id in story_ids[: max_docs * 4]:
        try:
            item_response = session.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json", timeout=60)
            item_response.raise_for_status()
        except requests.RequestException:
            continue
        item = item_response.json() or {}
        parts: list[str] = []
        if item.get("title"):
            parts.append(str(item["title"]))
        if item.get("text"):
            parts.append(_strip_html_tags(str(item["text"])))
        if item.get("url"):
            parts.append(f"URL: {item['url']}")
        combined = _clean_whitespace("\n\n".join(part for part in parts if part))
        if len(combined) < 80:
            continue
        documents.append(
            FetchedDocument(
                domain="web",
                source="hacker_news",
                source_id=str(story_id),
                title=f"HN {story_id}",
                text=combined,
                metadata={"hn_id": story_id, "score": item.get("score"), "by": item.get("by")},
            )
        )
        if len(documents) >= max_docs:
            break
    return documents


def fetch_technical_documents(max_docs: int, seed: int = 13) -> list[FetchedDocument]:
    session = _session()
    query = quote_plus("cat:cs.CL OR cat:cs.LG OR cat:cs.AI")
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={query}&start=0&max_results={max_docs * 2}&sortBy=submittedDate&sortOrder=descending"
    )
    response = session.get(url, timeout=60)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", namespace)
    rng = random.Random(seed)
    rng.shuffle(entries)
    documents: list[FetchedDocument] = []
    for entry in entries:
        title = _clean_whitespace(entry.findtext("atom:title", default="", namespaces=namespace))
        summary = _clean_whitespace(entry.findtext("atom:summary", default="", namespaces=namespace))
        if not title or not summary:
            continue
        authors = [
            _clean_whitespace(author.findtext("atom:name", default="", namespaces=namespace))
            for author in entry.findall("atom:author", namespace)
        ]
        categories = [node.attrib.get("term", "") for node in entry.findall("atom:category", namespace)]
        arxiv_id = _clean_whitespace(entry.findtext("atom:id", default="", namespaces=namespace))
        text = f"# {title}\n\n{summary}\n\nAuthors: {', '.join(author for author in authors if author)}"
        documents.append(
            FetchedDocument(
                domain="technical",
                source="arxiv_api",
                source_id=arxiv_id,
                title=title,
                text=text,
                metadata={"authors": authors, "categories": categories, "arxiv_id": arxiv_id},
            )
        )
        if len(documents) >= max_docs:
            break
    return documents


def fetch_real_datasets(
    output_root: str | Path,
    docs_per_domain: int = 64,
    domains: list[str] | None = None,
    overwrite: bool = False,
    seed: int = 13,
) -> dict[str, object]:
    output_root = Path(output_root)
    selected = domains or ["english", "code", "web", "technical"]
    fetchers = {
        "english": fetch_english_documents,
        "code": fetch_code_documents,
        "web": fetch_web_documents,
        "technical": fetch_technical_documents,
    }
    manifest: dict[str, object] = {
        "generated_at_epoch_s": int(time.time()),
        "docs_per_domain": docs_per_domain,
        "domains": selected,
        "seed": seed,
        "sources": {},
    }
    for domain in selected:
        if domain not in fetchers:
            raise ValueError(f"Unsupported domain: {domain}")
        domain_dir = output_root / domain
        if domain_dir.exists() and overwrite:
            for existing in domain_dir.glob("*"):
                if existing.is_file():
                    existing.unlink()
        domain_dir.mkdir(parents=True, exist_ok=True)
        documents = fetchers[domain](docs_per_domain, seed=seed)
        written_paths = _write_documents(documents, domain_dir)
        manifest["sources"][domain] = [
            {
                "path": str(path.relative_to(output_root)),
                "source": document.source,
                "source_id": document.source_id,
                "title": document.title,
                "metadata": document.metadata,
            }
            for path, document in zip(written_paths, documents)
        ]
    (output_root / "manifest.real.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
