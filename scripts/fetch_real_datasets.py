from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastbpe.datasets.real_sources import fetch_real_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch real-world datasets into data/ automatically.")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--docs-per-domain", type=int, default=64)
    parser.add_argument("--domains", nargs="+", default=["english", "code", "web", "technical"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fetch_real_datasets(
        output_root=Path(args.output_root),
        docs_per_domain=args.docs_per_domain,
        domains=args.domains,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
