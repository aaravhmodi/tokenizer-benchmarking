from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _format_table(df: pd.DataFrame, columns: list[str]) -> str:
    subset = df[columns].copy()
    headers = [str(column) for column in subset.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in subset.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def generate_report(results_dir: str | Path = "results", paper_dir: str | Path = "paper") -> Path:
    results_dir = Path(results_dir)
    paper_dir = Path(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(results_dir / "benchmark_summary.csv")
    environment = json.loads((results_dir / "environment.json").read_text(encoding="utf-8"))
    top_speed = summary.sort_values("mb_per_s", ascending=False).head(5)
    match_rows = summary.dropna(subset=["exact_match_rate"])

    report = f"""# FastBPE: Benchmarking Exact Tokenizer Throughput Across Text Domains

## Abstract

FastBPE benchmarks tokenizer throughput, latency, memory, and output exactness across multiple text domains. The suite compares production tokenizer libraries against custom Python baselines and includes a repeated-substring cache prototype to test whether exact tokenization can be accelerated without changing outputs where compatibility is expected.

## Introduction

Tokenizer speed directly affects dataset preprocessing, RAG ingestion, embedding pipelines, and inference server throughput. This project isolates tokenization cost and studies how text type, input length, batching, and warm versus cold execution affect real throughput.

## Background

Byte Pair Encoding combines pre-tokenization with iterative merge logic to map text into token IDs. Faster tokenization is useful only when outputs remain correct for the intended vocabulary. For OpenAI-compatible experiments, `tiktoken` is the reference tokenizer.

## Methodology

- Tokenizers compared: {", ".join(environment["tokenizers"])}
- Domains: {", ".join(environment["domains"])}
- Trials per configuration: {environment["trials"]}
- Batch size: {environment["batch_size"]}
- Platform: {environment["platform"]}
- Python: {environment["python_version"]}

### Benchmark Summary

{_format_table(top_speed, ["tokenizer", "domain", "mode", "mb_per_s", "tokens_per_s", "avg_latency_ms", "peak_memory_bytes"])}

## Results

### Fastest Configurations by MB/s

{_format_table(top_speed, ["tokenizer", "domain", "mb_per_s", "tokens_per_s", "avg_latency_ms"])}

### Exactness

{"No exact-match-compatible non-reference tokenizer results were recorded." if match_rows.empty else _format_table(match_rows, ["tokenizer", "domain", "exact_match_rate"])}

## Discussion

The main result should be interpreted jointly across throughput, latency spread, and memory usage. Domain-specific slowdowns are expected because code, noisy text, and markdown drive different pre-tokenization behavior and token boundary density. The cached prototype is designed to help repeated-token workloads and should be evaluated against its memory overhead and hit rate.

## Proposed Solution

The optimized tokenizer adds a repeated-substring cache on top of a simple Python baseline. This preserves deterministic output for its own vocabulary while reducing repeated lookup work for recurring words or symbols. The benchmark framework records cache hits, misses, and evictions so speedup can be related to workload structure instead of treated as a black box.

## Limitations

- Results are hardware- and Python-runtime-specific.
- Synthetic corpora are used when real datasets are absent.
- Non-compatible vocabularies cannot be judged on exact token ID equality.
- The custom Python tokenizers are research baselines, not production Rust or C++ systems.

## Conclusion

FastBPE provides a reproducible framework for answering when faster exact tokenization is possible, which domains stress tokenizer implementations most, and how caching changes the speed-memory tradeoff.
"""
    output_path = paper_dir / "results_summary.md"
    output_path.write_text(report, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    generate_report()
