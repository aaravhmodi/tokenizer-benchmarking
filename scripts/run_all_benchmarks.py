from __future__ import annotations

import argparse
from pathlib import Path

from fastbpe.benchmarks.runner import run_benchmarks
from fastbpe.plots.generate_plots import generate_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastBPE tokenizer benchmarks.")
    parser.add_argument("--dataset", nargs="+", default=["all"], help="Dataset domains to benchmark.")
    parser.add_argument("--tokenizers", nargs="+", default=["tiktoken", "hf", "sentencepiece", "naive", "cached"])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-docs", type=int, default=64)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--reference-tokenizer", default="tiktoken")
    parser.add_argument("--enable-memory-profiler", action="store_true")
    parser.add_argument("--data-root", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmarks(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        tokenizer_names=args.tokenizers,
        trials=args.trials,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        dataset_filter=args.dataset,
        reference_tokenizer=args.reference_tokenizer,
        enable_memory_profiler=args.enable_memory_profiler,
    )
    generate_plots(args.output_dir)


if __name__ == "__main__":
    main()

