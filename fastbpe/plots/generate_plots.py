from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def generate_plots(results_dir: str | Path) -> list[Path]:
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    summary = pd.read_csv(results_dir / "benchmark_summary.csv")
    raw = pd.read_csv(results_dir / "benchmark_raw.csv")
    memory = pd.read_csv(results_dir / "memory_results.csv")
    length = pd.read_csv(results_dir / "length_scaling.csv")
    batch = pd.read_csv(results_dir / "batch_results.csv") if (results_dir / "batch_results.csv").exists() else pd.DataFrame()

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary, x="tokenizer", y="mb_per_s", hue="domain")
    plt.title("Tokenizer Throughput (MB/s)")
    path = plots_dir / "throughput_mb_s.png"
    _save_plot(path)
    generated.append(path)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary, x="tokenizer", y="tokens_per_s", hue="domain")
    plt.title("Tokenizer Throughput (tokens/s)")
    path = plots_dir / "throughput_tokens_s.png"
    _save_plot(path)
    generated.append(path)

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=raw, x="tokenizer", y="avg_latency_ms")
    plt.title("Latency Distribution by Tokenizer")
    path = plots_dir / "latency_distribution.png"
    _save_plot(path)
    generated.append(path)

    latency_metrics = summary.groupby("tokenizer", as_index=False).agg(
        avg_latency_ms=("avg_latency_ms", "mean"),
        p50_latency_ms=("p50_latency_ms", "mean"),
        p95_latency_ms=("p95_latency_ms", "mean"),
        p99_latency_ms=("p99_latency_ms", "mean"),
    )
    latency_long = latency_metrics.melt(
        id_vars="tokenizer",
        value_vars=["avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms"],
        var_name="metric",
        value_name="latency_ms",
    )
    metric_labels = {
        "avg_latency_ms": "avg",
        "p50_latency_ms": "p50",
        "p95_latency_ms": "p95",
        "p99_latency_ms": "p99",
    }
    latency_long["metric"] = latency_long["metric"].map(metric_labels)
    plt.figure(figsize=(11, 5))
    ax = sns.barplot(data=latency_long, x="tokenizer", y="latency_ms", hue="metric")
    plt.title("Latency Summary by Tokenizer (avg, p50, p95, p99)")
    plt.ylabel("Latency (ms)")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    path = plots_dir / "latency_percentiles.png"
    _save_plot(path)
    generated.append(path)

    heatmap_data = summary.pivot_table(index="tokenizer", columns="domain", values="mb_per_s", aggfunc="mean")
    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Domain Sensitivity Heatmap")
    path = plots_dir / "domain_heatmap.png"
    _save_plot(path)
    generated.append(path)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=length, x="length_bucket", y="latency_ms", hue="tokenizer", estimator="mean")
    plt.title("Input Length vs Latency")
    path = plots_dir / "length_vs_latency.png"
    _save_plot(path)
    generated.append(path)

    throughput_length = (
        length.groupby(["tokenizer", "length_bucket"], as_index=False)
        .agg(tokens=("tokens", "sum"), bytes=("bytes", "sum"), latency_ms=("latency_ms", "sum"))
    )
    throughput_length["mb_per_s"] = (throughput_length["bytes"] / (1024 * 1024)) / (throughput_length["latency_ms"] / 1000).replace(0, 1)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=throughput_length, x="length_bucket", y="mb_per_s", hue="tokenizer", marker="o")
    plt.title("Input Length vs Throughput")
    path = plots_dir / "length_vs_throughput.png"
    _save_plot(path)
    generated.append(path)

    if not batch.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=batch, x="tokenizer", y="throughput_docs_per_s", hue="domain")
        plt.title("Batch Encoding Throughput")
        path = plots_dir / "batch_throughput.png"
        _save_plot(path)
        generated.append(path)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=memory, x="tokenizer", y="peak_memory_bytes", hue="domain")
    plt.title("Peak Memory Usage by Tokenizer")
    path = plots_dir / "memory_usage.png"
    _save_plot(path)
    generated.append(path)

    exact_summary = summary.dropna(subset=["exact_match_rate"])
    if not exact_summary.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=exact_summary, x="tokenizer", y="exact_match_rate", hue="domain")
        plt.title("Exact Token Match Rate")
        path = plots_dir / "exact_match_rate.png"
        _save_plot(path)
        generated.append(path)

    cache_rows = raw[raw["tokenizer"] == "cached"].copy()
    if not cache_rows.empty and "cache_hit_rate" in cache_rows.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=cache_rows, x="cache_hit_rate", y="tokens_per_s", hue="domain")
        plt.title("Cache Hit Rate vs Speedup Proxy")
        path = plots_dir / "cache_hit_rate_vs_speedup.png"
        _save_plot(path)
        generated.append(path)

    tradeoff = summary.groupby("tokenizer", as_index=False).agg(
        mb_per_s=("mb_per_s", "mean"),
        peak_memory_bytes=("peak_memory_bytes", "mean"),
    )
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=tradeoff, x="peak_memory_bytes", y="mb_per_s", hue="tokenizer", s=100)
    plt.title("Speed-Memory Tradeoff")
    path = plots_dir / "speed_memory_tradeoff.png"
    _save_plot(path)
    generated.append(path)

    tradeoff = tradeoff.sort_values(["peak_memory_bytes", "mb_per_s"], ascending=[True, False])
    frontier = []
    best_speed = -1.0
    for row in tradeoff.to_dict(orient="records"):
        if row["mb_per_s"] > best_speed:
            frontier.append(row)
            best_speed = row["mb_per_s"]
    frontier_df = pd.DataFrame(frontier)
    if not frontier_df.empty:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=frontier_df, x="peak_memory_bytes", y="mb_per_s", marker="o")
        plt.title("Pareto Frontier: Speed vs Memory")
        path = plots_dir / "pareto_frontier.png"
        _save_plot(path)
        generated.append(path)

    return generated
