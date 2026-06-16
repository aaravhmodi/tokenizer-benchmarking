from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastbpe.datasets.synthetic import SyntheticDatasetConfig, build_synthetic_corpora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize synthetic FastBPE datasets into data/.")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--docs-per-domain", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extension_for_domain(domain: str) -> str:
    return {
        "english": ".txt",
        "code": ".py",
        "web": ".html",
        "technical": ".md",
    }.get(domain, ".txt")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    corpora = build_synthetic_corpora(SyntheticDatasetConfig(docs_per_domain=args.docs_per_domain))
    for domain, texts in corpora.items():
        domain_dir = output_root / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        suffix = extension_for_domain(domain)
        for index, text in enumerate(texts, start=1):
            path = domain_dir / f"{domain}_{index:03d}{suffix}"
            if path.exists() and not args.overwrite:
                continue
            path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
