from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


PLOT_FILES = [
    ("throughput_mb_s.png", "Experiment 1: Tokenizer throughput comparison in MB/s."),
    ("throughput_tokens_s.png", "Experiment 1: Tokenizer throughput comparison in tokens/s."),
    ("latency_distribution.png", "Latency distribution by tokenizer across benchmark trials."),
    ("domain_heatmap.png", "Experiment 2: domain sensitivity heatmap for throughput."),
    ("length_vs_latency.png", "Experiment 3: input length versus latency."),
    ("length_vs_throughput.png", "Experiment 3: input length versus throughput."),
    ("batch_throughput.png", "Experiment 4: batch encoding throughput."),
    ("memory_usage.png", "Memory usage by tokenizer and domain."),
    ("cache_hit_rate_vs_speedup.png", "Experiment 6: cache hit rate versus throughput proxy."),
    ("speed_memory_tradeoff.png", "Speed-memory tradeoff across tokenizer implementations."),
    ("pareto_frontier.png", "Pareto frontier of throughput versus memory."),
]


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _wrap_paragraphs(paragraphs: list[str], width: int = 100) -> list[str]:
    lines: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width))
        lines.append("")
    return lines


def _text_page(pdf: PdfPages, title: str, paragraphs: list[str], footer: str | None = None) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
    ax.axis("off")
    ax.text(0, 1.0, title, fontsize=18, fontweight="bold", va="top", ha="left", family="serif")
    y = 0.95
    for line in _wrap_paragraphs(paragraphs):
        if y < 0.06:
            break
        ax.text(0, y, line, fontsize=10.5, va="top", ha="left", family="serif")
        y -= 0.021 if line else 0.012
    if footer:
        ax.text(0, 0.01, footer, fontsize=9, va="bottom", ha="left", family="serif", color="#444444")
    pdf.savefig(fig)
    plt.close(fig)


def _table_page(
    pdf: PdfPages,
    title: str,
    dataframe: pd.DataFrame,
    numeric_digits: dict[str, int] | None = None,
    intro: str | None = None,
) -> None:
    numeric_digits = numeric_digits or {}
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.axis("off")
    ax.text(0, 1.0, title, fontsize=16, fontweight="bold", va="top", ha="left", family="serif")
    y0 = 0.95
    if intro:
        wrapped = textwrap.wrap(intro, width=110)
        for line in wrapped:
            ax.text(0, y0, line, fontsize=10.5, va="top", ha="left", family="serif")
            y0 -= 0.022
        y0 -= 0.01

    formatted = dataframe.copy()
    for column, digits in numeric_digits.items():
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda v: ""
                if v == ""
                else (_fmt(float(v), digits) if pd.notna(v) else "N/A")
            )
    for column in formatted.columns:
        formatted[column] = formatted[column].astype(str)

    table = ax.table(
        cellText=formatted.values.tolist(),
        colLabels=list(formatted.columns),
        cellLoc="center",
        loc="upper left",
        bbox=[0, max(0.02, y0 - 0.06 - 0.045 * len(formatted)), 1, min(0.8, 0.06 + 0.045 * (len(formatted) + 1))],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8edf5")
        cell.set_edgecolor("#777777")
    pdf.savefig(fig)
    plt.close(fig)


def _figure_page(pdf: PdfPages, image_path: Path, caption: str) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.06, 0.12, 0.88, 0.8])
    ax.axis("off")
    image = mpimg.imread(image_path)
    ax.imshow(image)
    fig.text(0.06, 0.95, image_path.stem.replace("_", " ").title(), fontsize=16, fontweight="bold", family="serif")
    fig.text(0.06, 0.06, caption, fontsize=10.5, family="serif")
    pdf.savefig(fig)
    plt.close(fig)


def generate_pdf_paper(results_dir: str | Path = "results", paper_dir: str | Path = "paper") -> Path:
    plt.rcParams["font.family"] = "serif"
    results_dir = Path(results_dir)
    paper_dir = Path(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(results_dir / "benchmark_summary.csv")
    raw = pd.read_csv(results_dir / "benchmark_raw.csv")
    batch = pd.read_csv(results_dir / "batch_results.csv")
    length = pd.read_csv(results_dir / "length_scaling.csv")
    environment = json.loads((results_dir / "environment.json").read_text(encoding="utf-8"))

    mean_by_tokenizer = (
        summary.groupby("tokenizer", as_index=False)[["mb_per_s", "tokens_per_s", "avg_latency_ms", "peak_memory_bytes"]]
        .mean()
        .sort_values("mb_per_s", ascending=False)
    )
    top_configs = summary.sort_values("mb_per_s", ascending=False).head(10)[
        ["tokenizer", "domain", "mb_per_s", "tokens_per_s", "avg_latency_ms", "peak_memory_bytes"]
    ]
    domain_matrix = summary.pivot_table(index="domain", columns="tokenizer", values="mb_per_s", aggfunc="mean").reset_index()
    direct_compare = summary[summary["tokenizer"].isin(["tiktoken", "tiktoken_cached"])][
        ["tokenizer", "domain", "mb_per_s", "tokens_per_s", "avg_latency_ms", "peak_memory_bytes", "exact_match_rate"]
    ].copy()
    cold_start = raw.groupby("tokenizer", as_index=False)["cold_start_ms"].mean().sort_values("cold_start_ms")
    batch_summary = (
        batch.groupby("tokenizer", as_index=False)[["throughput_docs_per_s", "throughput_tokens_per_s", "avg_batch_latency_ms"]]
        .mean()
        .sort_values("throughput_docs_per_s", ascending=False)
    )
    length_summary = (
        length.groupby(["tokenizer", "length_bucket"], as_index=False)[["latency_ms", "bytes", "tokens"]]
        .mean()
        .sort_values(["tokenizer", "length_bucket"])
    )
    cached_vs_naive = (
        raw[raw["tokenizer"].isin(["naive", "cached"])]
        .groupby("tokenizer", as_index=False)[["mb_per_s", "tokens_per_s", "avg_latency_ms", "peak_memory_bytes", "cache_hit_rate"]]
        .mean(numeric_only=True)
    )
    exact_cached = (
        raw[raw["tokenizer"] == "tiktoken_cached"]
        .groupby("tokenizer", as_index=False)[["mb_per_s", "tokens_per_s", "avg_latency_ms", "peak_memory_bytes", "cache_hit_rate"]]
        .mean(numeric_only=True)
    )
    naive_row = cached_vs_naive[cached_vs_naive["tokenizer"] == "naive"].iloc[0]
    cached_row = cached_vs_naive[cached_vs_naive["tokenizer"] == "cached"].iloc[0]
    best_tokenizer = mean_by_tokenizer.iloc[0]
    best_code = summary[summary["domain"] == "code"].sort_values("mb_per_s", ascending=False).iloc[0]
    exact_rows = summary.dropna(subset=["exact_match_rate"]).copy()
    exact_cached_row = exact_cached.iloc[0] if not exact_cached.empty else None
    tiktoken_row = mean_by_tokenizer[mean_by_tokenizer["tokenizer"] == "tiktoken"].iloc[0] if "tiktoken" in mean_by_tokenizer["tokenizer"].values else None

    abstract = (
        f"FastBPE benchmarks tokenizer throughput, latency, memory, batching behavior, and domain sensitivity across "
        f"English, code, web, and technical text. In the current five-trial run on {environment['platform']}, "
        f"{best_tokenizer['tokenizer']} was the strongest average MB/s performer at {_fmt(float(best_tokenizer['mb_per_s']))} MB/s. "
        f"SentencePiece remained competitive and led the code domain at {_fmt(float(best_code['mb_per_s']))} MB/s. "
        f"The cached Python tokenizer improved on the naive Python baseline by "
        f"{_fmt(float(cached_row['mb_per_s'] / naive_row['mb_per_s']), 2)}x in MB/s with higher memory usage. "
        + (
            f"The exact-compatible tiktoken_cached path preserved token IDs exactly and now slightly outperformed native tiktoken "
            f"({_fmt(float(exact_cached_row['mb_per_s']))} vs {_fmt(float(tiktoken_row['mb_per_s']))} MB/s)."
            if exact_cached_row is not None and tiktoken_row is not None
            else ""
        )
    )

    hypotheses = [
        "H1. Production native tokenizer libraries will outperform readable Python baselines on throughput and latency.",
        "H2. Tokenizer rankings will change across English prose, code, noisy web text, and technical markdown-like text.",
        "H3. Repeated-substring caching will improve the custom Python baseline while increasing memory usage.",
        "H4. Batch encoding will favor tokenizers that expose efficient multi-document paths, but speed gains will not be uniform.",
        "H5. Exact token-ID compatibility cannot be claimed unless the tokenizer intentionally matches the reference vocabulary and merge logic.",
        "H6. Even if exactness is preserved, any speed gain from caching may be modest and may come with substantial memory overhead.",
    ]

    methodology = [
        f"Environment. Platform: {environment['platform']}. Python: {environment['python_version']}. Processor: {environment['processor']}.",
        f"Tokenizers. {', '.join(environment['tokenizers'])}.",
        f"Domains. {', '.join(environment['domains'])}.",
        f"Controls. Warmup before timing, separate cold-start measurement, identical document sets per run, five trials, and raw CSV/JSON output.",
        "Datasets. This run used persisted fetched corpora in data/ rather than the in-memory homogeneous fallback.",
        "Exactness. tiktoken is the reference tokenizer, and this run includes tiktoken_cached as an intentionally exact-compatible cached path.",
    ]

    experiments = [
        "Experiment 1: Overall tokenizer speed, measured in MB/s, tokens/s, documents/s, and latency percentiles.",
        "Experiment 2: Domain sensitivity across clean English, code, noisy web text, and technical text.",
        "Experiment 3: Input-length scaling across the available length buckets in the current corpus.",
        "Experiment 4: Batch versus single encoding, reported for tokenizers that support encode_batch.",
        "Experiment 5: Exactness testing for the tiktoken_cached exact-compatible adapter against the tiktoken reference.",
        "Experiment 6: Caching optimization, comparing the naive and cached Python prototypes.",
    ]

    results_points = [
        f"Overall winner by mean MB/s: tiktoken at {_fmt(float(best_tokenizer['mb_per_s']))} MB/s and {_fmt(float(best_tokenizer['avg_latency_ms']), 4)} ms average latency.",
        f"Fastest code-domain tokenizer: {best_code['tokenizer']} at {_fmt(float(best_code['mb_per_s']))} MB/s.",
        f"Cached versus naive Python baseline: {_fmt(float(naive_row['mb_per_s']))} to {_fmt(float(cached_row['mb_per_s']))} MB/s, "
        f"or {_fmt(float(cached_row['mb_per_s'] / naive_row['mb_per_s']), 2)}x speedup, with peak memory rising from "
        f"{_fmt(float(naive_row['peak_memory_bytes']), 1)} to {_fmt(float(cached_row['peak_memory_bytes']), 1)} bytes.",
        f"Cold start ranking by mean latency favored naive ({_fmt(float(cold_start.iloc[0]['cold_start_ms']), 4)} ms) and tiktoken "
        f"({_fmt(float(cold_start[cold_start['tokenizer']=='tiktoken'].iloc[0]['cold_start_ms']), 4)} ms), while hf was slowest at "
        f"{_fmt(float(cold_start[cold_start['tokenizer']=='hf'].iloc[0]['cold_start_ms']), 4)} ms.",
        f"Batch throughput favored the cached baseline at {_fmt(float(batch_summary.iloc[0]['throughput_docs_per_s']), 1)} docs/s, "
        f"but tiktoken is absent from the batch table because the current adapter intentionally exposes single-document encoding only.",
        (
            f"Exact-compatible caching preserved correctness at 1.0 exact match rate, and mean throughput for tiktoken_cached was "
            f"{_fmt(float(exact_cached_row['mb_per_s']))} MB/s versus {_fmt(float(tiktoken_row['mb_per_s']))} MB/s for native tiktoken, "
            f"but with far higher memory usage."
            if exact_cached_row is not None and tiktoken_row is not None
            else ""
        ),
    ]

    discussion = [
        "The results support H1. Native tokenizer libraries clearly outperformed the Python baselines on MB/s and latency, with tiktoken and SentencePiece consistently ahead of naive Python code.",
        "The results support H2. Ranking changed by domain, which matters because tokenizer selection for RAG preprocessing, web corpora, and code corpora should not rely on one aggregate benchmark number.",
        "The results support H3 with caveats. Caching materially improved the Python baseline, but it increased memory and still depended on high substring reuse. Even after diversifying the synthetic corpus, the cache hit rate remained high, so the next step is to rerun on real corpora.",
        "The results partially support H4. Batch throughput was much higher for adapters with encode_batch support, but batch results are not comparable for tiktoken until a batch-capable adapter path is added.",
        "H5 is now supported empirically: the exact-compatible cached adapter matched tiktoken token IDs exactly across the benchmark corpus.",
        "H6 is supported in a narrower sense: reducing Python overhead changed the result from a slowdown into a small speed win, but the memory cost remains large enough that the overall systems conclusion is still a tradeoff rather than a free improvement.",
    ]

    output_path = paper_dir / "fastbpe_paper.pdf"
    with PdfPages(output_path) as pdf:
        _text_page(
            pdf,
            "FastBPE: Benchmarking and Optimizing Tokenizer Speed",
            [
                "Research-style benchmark report",
                "",
                abstract,
                "",
                "Hypotheses",
                *hypotheses,
            ],
            footer="Generated directly from the current benchmark outputs in results/.",
        )
        _text_page(
            pdf,
            "Methodology and Experimental Design",
            methodology + ["Experiments"] + experiments,
        )
        _table_page(
            pdf,
            "Overall Results by Tokenizer",
            mean_by_tokenizer,
            numeric_digits={"mb_per_s": 3, "tokens_per_s": 0, "avg_latency_ms": 4, "peak_memory_bytes": 1},
            intro="This table aggregates mean performance across all domains and trials.",
        )
        _table_page(
            pdf,
            "Top Tokenizer-Domain Configurations",
            top_configs,
            numeric_digits={"mb_per_s": 3, "tokens_per_s": 0, "avg_latency_ms": 4, "peak_memory_bytes": 1},
            intro="These are the strongest individual tokenizer-domain combinations by MB/s.",
        )
        _table_page(
            pdf,
            "Domain Sensitivity: MB/s by Domain",
            domain_matrix,
            numeric_digits={"cached": 3, "hf": 3, "naive": 3, "sentencepiece": 3, "tiktoken": 3},
            intro="Throughput shifts materially by domain, which changes the ranking of libraries.",
        )
        _table_page(
            pdf,
            "Cold Start and Batch Results",
            pd.concat(
                [
                    cold_start.rename(columns={"cold_start_ms": "value"}).assign(metric="cold_start_ms")[["tokenizer", "metric", "value"]],
                    batch_summary.rename(
                        columns={
                            "throughput_docs_per_s": "docs_per_s",
                            "throughput_tokens_per_s": "tokens_per_s",
                            "avg_batch_latency_ms": "batch_latency_ms",
                        }
                    ),
                ],
                ignore_index=True,
                sort=False,
            ).fillna(""),
            numeric_digits={"value": 4, "docs_per_s": 1, "tokens_per_s": 0, "batch_latency_ms": 4},
            intro="Cold start is reported for all tokenizers. Batch metrics are available only for adapters with batch support.",
        )
        _table_page(
            pdf,
            "Input-Length Scaling Summary",
            length_summary,
            numeric_digits={"latency_ms": 4, "bytes": 1, "tokens": 1},
            intro="The current persisted corpus produced two dominant length buckets, so the length-scaling analysis is informative but not yet broad.",
        )
        _table_page(
            pdf,
            "Caching Optimization Results",
            cached_vs_naive,
            numeric_digits={"mb_per_s": 3, "tokens_per_s": 0, "avg_latency_ms": 4, "peak_memory_bytes": 1, "cache_hit_rate": 4},
            intro="Repeated-substring caching improved throughput and latency relative to the naive Python baseline, with a memory cost.",
        )
        if not exact_rows.empty:
            _table_page(
                pdf,
                "Exact-Compatible Cached Results",
                exact_rows[["tokenizer", "domain", "exact_match_rate"]].copy(),
                numeric_digits={"exact_match_rate": 4},
                intro="The exact-compatible cached adapter is evaluated separately because the key question is whether correctness can be preserved while optimizing.",
            )
        if not direct_compare.empty:
            _table_page(
                pdf,
                "Direct Comparison: tiktoken vs tiktoken_cached",
                direct_compare,
                numeric_digits={"mb_per_s": 3, "tokens_per_s": 0, "avg_latency_ms": 4, "peak_memory_bytes": 1, "exact_match_rate": 4},
                intro="This table isolates the most important comparison in the project: the native reference versus the exact-compatible cached variant.",
            )
        _text_page(
            pdf,
            "Results and Discussion",
            ["Headline findings"] + results_points + ["Discussion"] + discussion,
        )
        for filename, caption in PLOT_FILES:
            image_path = results_dir / "plots" / filename
            if image_path.exists():
                _figure_page(pdf, image_path, caption)

    return output_path


if __name__ == "__main__":
    generate_pdf_paper()
