from __future__ import annotations

from pathlib import Path

from fastbpe.datasets.synthetic import SyntheticDatasetConfig, build_synthetic_corpora


def load_domain_texts(data_root: str | Path, max_docs: int | None = None) -> dict[str, list[str]]:
    root = Path(data_root)
    corpora: dict[str, list[str]] = {}
    for domain_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        texts: list[str] = []
        for file_path in sorted(domain_dir.glob("**/*")):
            if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md", ".py", ".js", ".ts", ".json", ".html"}:
                texts.append(file_path.read_text(encoding="utf-8", errors="ignore"))
            if max_docs is not None and len(texts) >= max_docs:
                break
        if texts:
            corpora[domain_dir.name] = texts[:max_docs] if max_docs else texts
    if corpora:
        return corpora
    return build_synthetic_corpora(SyntheticDatasetConfig(docs_per_domain=max_docs or 64))

