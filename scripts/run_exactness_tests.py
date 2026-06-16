from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastbpe.benchmarks.exactness import compare_to_reference
from fastbpe.benchmarks.runner import build_tokenizers
from fastbpe.datasets.loader import load_domain_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exactness tests against a reference tokenizer.")
    parser.add_argument("--reference", default="tiktoken")
    parser.add_argument("--tokenizers", nargs="+", default=["tiktoken"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--max-docs", type=int, default=64)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpora = load_domain_texts(args.data_root, max_docs=args.max_docs)
    training_corpus = [text for texts in corpora.values() for text in texts[: min(32, len(texts))]]
    tokenizers = build_tokenizers(sorted(set(args.tokenizers + [args.reference])), training_corpus)
    tokenizer_map = {tokenizer.name: tokenizer for tokenizer in tokenizers}
    reference = tokenizer_map[args.reference]
    rows: list[dict[str, object]] = []
    for name in args.tokenizers:
        candidate = tokenizer_map[name]
        if candidate.name == reference.name:
            continue
        for domain, texts in corpora.items():
            result = compare_to_reference(reference, candidate, texts)
            rows.append(
                {
                    "tokenizer": candidate.name,
                    "domain": domain,
                    "compared_docs": result.compared_docs,
                    "exact_matches": result.exact_matches,
                    "exact_match_rate": result.match_rate,
                    "mismatch_examples": result.mismatches,
                }
            )
    output_path = Path(args.output_dir) / "exactness_report.json"
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
