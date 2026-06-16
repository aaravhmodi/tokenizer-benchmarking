from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

from fastbpe.adapters.base import TokenizerAdapter
from fastbpe.adapters.cached_bpe import CachedPythonBPEAdapter
from fastbpe.adapters.hf_adapter import HuggingFaceTokenizerAdapter
from fastbpe.adapters.naive_bpe import NaivePythonBPEAdapter, PretokenizationComparator
from fastbpe.adapters.sentencepiece_adapter import SentencePieceAdapter
from fastbpe.adapters.tiktoken_adapter import TikTokenAdapter
from fastbpe.benchmarks.exactness import compare_to_reference
from fastbpe.benchmarks.memory import measure_memory
from fastbpe.benchmarks.metrics import build_trial_metrics, summarize_trials
from fastbpe.datasets.loader import load_domain_texts
from fastbpe.utils.text_stats import length_bucket
from fastbpe.utils.timing import time_single


def available_tokenizer_factories(training_corpus: list[str]) -> dict[str, callable]:
    return {
        "tiktoken": lambda: TikTokenAdapter(),
        "hf": lambda: HuggingFaceTokenizerAdapter(training_corpus=training_corpus),
        "sentencepiece": lambda: SentencePieceAdapter(training_corpus=training_corpus),
        "naive": lambda: NaivePythonBPEAdapter(),
        "cached": lambda: CachedPythonBPEAdapter(),
    }


def build_tokenizers(names: list[str], training_corpus: list[str]) -> list[TokenizerAdapter]:
    factories = available_tokenizer_factories(training_corpus)
    tokenizers: list[TokenizerAdapter] = []
    for name in names:
        try:
            tokenizers.append(factories[name]())
        except KeyError as exc:
            raise ValueError(f"Unknown tokenizer: {name}") from exc
    return tokenizers


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    _ensure_parent(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_benchmarks(
    data_root: str | Path,
    output_dir: str | Path,
    tokenizer_names: list[str],
    trials: int = 5,
    batch_size: int = 8,
    max_docs: int | None = None,
    dataset_filter: list[str] | None = None,
    reference_tokenizer: str = "tiktoken",
    enable_memory_profiler: bool = True,
) -> dict[str, object]:
    corpora = load_domain_texts(data_root, max_docs=max_docs)
    if dataset_filter and "all" not in dataset_filter:
        corpora = {name: texts for name, texts in corpora.items() if name in dataset_filter}
    training_corpus = [text for texts in corpora.values() for text in texts[: min(32, len(texts))]]
    tokenizers = build_tokenizers(tokenizer_names, training_corpus)
    reference = next((tok for tok in tokenizers if tok.name == reference_tokenizer), None)
    if reference is None and reference_tokenizer in available_tokenizer_factories(training_corpus):
        reference = available_tokenizer_factories(training_corpus)[reference_tokenizer]()

    raw_rows: list[dict[str, object]] = []
    mismatch_rows: list[dict[str, object]] = []
    memory_rows: list[dict[str, object]] = []
    length_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []

    for tokenizer in tokenizers:
        tokenizer.warmup(training_corpus[:4])
        for domain, texts in corpora.items():
            tokenizer.reset_stats()
            cold_start_begin = time.perf_counter()
            tokenizer.encode(texts[0])
            cold_start_ms = (time.perf_counter() - cold_start_begin) * 1000
            for trial in range(1, trials + 1):
                if enable_memory_profiler:
                    timing_result, memory = measure_memory(time_single, tokenizer.encode, texts)
                else:
                    timing_result = time_single(tokenizer.encode, texts)
                    from fastbpe.benchmarks.memory import MemoryMeasurement

                    memory = MemoryMeasurement(peak_bytes=0, current_bytes=0)
                outputs = [tokenizer.encode(text) for text in texts]
                exact_match_rate = None
                token_delta = None
                if reference and tokenizer.metadata.compatible_with_reference and tokenizer.name != reference.name:
                    exactness = compare_to_reference(reference, tokenizer, texts)
                    exact_match_rate = exactness.match_rate
                    mismatch_rows.extend(
                        {
                            "tokenizer": tokenizer.name,
                            "domain": domain,
                            "trial": trial,
                            **mismatch,
                        }
                        for mismatch in exactness.mismatches
                    )
                elif reference and tokenizer.name != reference.name:
                    ref_total = sum(len(reference.encode(text)) for text in texts)
                    token_delta = sum(len(output) for output in outputs) - ref_total
                metrics = build_trial_metrics(
                    tokenizer=tokenizer.name,
                    domain=domain,
                    mode="single",
                    trial=trial,
                    texts=texts,
                    outputs=outputs,
                    elapsed_s=timing_result.elapsed_s,
                    latencies_s=timing_result.latencies_s,
                    peak_memory_bytes=memory.peak_bytes,
                    avg_memory_bytes=memory.current_bytes,
                    token_count_vs_reference=token_delta,
                    exact_match_rate=exact_match_rate,
                )
                row = metrics.to_dict()
                row["cold_start_ms"] = cold_start_ms
                row.update(tokenizer.stats())
                raw_rows.append(row)
                memory_rows.append(
                    {
                        "tokenizer": tokenizer.name,
                        "domain": domain,
                        "trial": trial,
                        "peak_memory_bytes": memory.peak_bytes,
                        "avg_memory_bytes": memory.current_bytes,
                    }
                )
                for text, latency_s, output in zip(texts, timing_result.latencies_s, outputs):
                    length_rows.append(
                        {
                            "tokenizer": tokenizer.name,
                            "domain": domain,
                            "trial": trial,
                            "length_bucket": length_bucket(text),
                            "chars": len(text),
                            "latency_ms": latency_s * 1000,
                            "tokens": len(output),
                            "bytes": len(text.encode("utf-8")),
                        }
                    )
                if tokenizer.metadata.supports_batch:
                    batches = [texts[index : index + batch_size] for index in range(0, len(texts), batch_size)]
                    batch_latencies: list[float] = []
                    batch_token_total = 0
                    batch_start = time.perf_counter()
                    for batch in batches:
                        per_batch_start = time.perf_counter()
                        batch_outputs = tokenizer.encode_batch(batch)
                        batch_latencies.append(time.perf_counter() - per_batch_start)
                        batch_token_total += sum(len(ids) for ids in batch_outputs)
                    batch_elapsed = time.perf_counter() - batch_start
                    batch_rows.append(
                        {
                            "tokenizer": tokenizer.name,
                            "domain": domain,
                            "trial": trial,
                            "batch_size": batch_size,
                            "docs": len(texts),
                            "throughput_docs_per_s": len(texts) / batch_elapsed if batch_elapsed else 0.0,
                            "throughput_tokens_per_s": batch_token_total / batch_elapsed if batch_elapsed else 0.0,
                            "avg_batch_latency_ms": (sum(batch_latencies) / len(batch_latencies) * 1000) if batch_latencies else 0.0,
                        }
                    )

    summary_rows = summarize_trials(raw_rows)
    output_path = Path(output_dir)
    write_csv(output_path / "benchmark_raw.csv", raw_rows)
    write_csv(output_path / "benchmark_summary.csv", summary_rows)
    write_csv(output_path / "memory_results.csv", memory_rows)
    write_csv(output_path / "length_scaling.csv", length_rows)
    write_csv(output_path / "batch_results.csv", batch_rows)
    (output_path / "mismatch_examples.json").write_text(json.dumps(mismatch_rows, indent=2), encoding="utf-8")
    metadata = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "tokenizers": tokenizer_names,
        "domains": sorted(corpora.keys()),
        "trials": trials,
        "batch_size": batch_size,
        "max_docs": max_docs,
        "reference_tokenizer": reference_tokenizer,
        "pretokenization_comparison": PretokenizationComparator().compare(training_corpus[:64]) if training_corpus else {},
    }
    (output_path / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "raw_rows": raw_rows,
        "summary_rows": summary_rows,
        "memory_rows": memory_rows,
        "length_rows": length_rows,
        "batch_rows": batch_rows,
        "mismatch_rows": mismatch_rows,
        "metadata": metadata,
    }
