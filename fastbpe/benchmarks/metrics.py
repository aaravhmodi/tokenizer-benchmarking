from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from fastbpe.utils.text_stats import total_megabytes
from fastbpe.utils.timing import summarize_latencies


@dataclass
class TrialMetrics:
    tokenizer: str
    domain: str
    mode: str
    trial: int
    docs: int
    total_runtime_s: float
    total_tokens: int
    mb_processed: float
    docs_per_s: float
    mb_per_s: float
    tokens_per_s: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    peak_memory_bytes: int
    avg_memory_bytes: int
    token_count_vs_reference: int | None = None
    exact_match_rate: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_trial_metrics(
    tokenizer: str,
    domain: str,
    mode: str,
    trial: int,
    texts: list[str],
    outputs: list[list[int]],
    elapsed_s: float,
    latencies_s: list[float],
    peak_memory_bytes: int,
    avg_memory_bytes: int,
    token_count_vs_reference: int | None = None,
    exact_match_rate: float | None = None,
) -> TrialMetrics:
    mb_processed = total_megabytes(texts)
    total_tokens = sum(len(ids) for ids in outputs)
    latency_summary = summarize_latencies(latencies_s)
    docs = len(texts)
    return TrialMetrics(
        tokenizer=tokenizer,
        domain=domain,
        mode=mode,
        trial=trial,
        docs=docs,
        total_runtime_s=elapsed_s,
        total_tokens=total_tokens,
        mb_processed=mb_processed,
        docs_per_s=docs / elapsed_s if elapsed_s else 0.0,
        mb_per_s=mb_processed / elapsed_s if elapsed_s else 0.0,
        tokens_per_s=total_tokens / elapsed_s if elapsed_s else 0.0,
        avg_latency_ms=latency_summary["avg_latency_ms"],
        p50_latency_ms=latency_summary["p50_latency_ms"],
        p95_latency_ms=latency_summary["p95_latency_ms"],
        p99_latency_ms=latency_summary["p99_latency_ms"],
        peak_memory_bytes=peak_memory_bytes,
        avg_memory_bytes=avg_memory_bytes,
        token_count_vs_reference=token_count_vs_reference,
        exact_match_rate=exact_match_rate,
    )


def summarize_trials(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["tokenizer"]), str(row["domain"]), str(row["mode"]))
        grouped.setdefault(key, []).append(row)
    summary: list[dict[str, object]] = []
    numeric_fields = [
        "total_runtime_s",
        "total_tokens",
        "mb_processed",
        "docs_per_s",
        "mb_per_s",
        "tokens_per_s",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "peak_memory_bytes",
        "avg_memory_bytes",
    ]
    for (tokenizer, domain, mode), group in grouped.items():
        row: dict[str, object] = {"tokenizer": tokenizer, "domain": domain, "mode": mode, "trials": len(group)}
        for field in numeric_fields:
            values = [float(item[field]) for item in group]
            row[field] = statistics.mean(values)
            row[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        match_values = [item.get("exact_match_rate") for item in group if item.get("exact_match_rate") is not None]
        row["exact_match_rate"] = statistics.mean(match_values) if match_values else None
        delta_values = [item.get("token_count_vs_reference") for item in group if item.get("token_count_vs_reference") is not None]
        row["token_count_vs_reference"] = statistics.mean(delta_values) if delta_values else None
        summary.append(row)
    return summary

