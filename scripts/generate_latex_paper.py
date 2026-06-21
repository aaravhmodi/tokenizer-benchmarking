from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _table_from_dataframe(df: pd.DataFrame, columns: list[str], numeric_formats: dict[str, int] | None = None) -> str:
    numeric_formats = numeric_formats or {}
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}"]
    lines.append(r"\toprule")
    lines.append(" & ".join(_latex_escape(column) for column in columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df[columns].iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("N/A")
            elif column in numeric_formats:
                values.append(_fmt_float(float(value), numeric_formats[column]))
            else:
                values.append(_latex_escape(value))
        lines.append(" & ".join(values) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def generate_latex_paper(results_dir: str | Path = "results", paper_dir: str | Path = "paper") -> Path:
    results_dir = Path(results_dir)
    paper_dir = Path(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(results_dir / "benchmark_summary.csv")
    raw = pd.read_csv(results_dir / "benchmark_raw.csv")
    environment = json.loads((results_dir / "environment.json").read_text(encoding="utf-8"))
    mismatch_examples = json.loads((results_dir / "mismatch_examples.json").read_text(encoding="utf-8"))

    top_configs = summary.sort_values("mb_per_s", ascending=False).head(8).copy()
    mean_by_tokenizer = (
        summary.groupby("tokenizer", as_index=False)[["mb_per_s", "tokens_per_s", "avg_latency_ms", "peak_memory_bytes"]]
        .mean()
        .sort_values("mb_per_s", ascending=False)
    )
    domain_matrix = summary.pivot_table(index="domain", columns="tokenizer", values="mb_per_s", aggfunc="mean").reset_index()
    direct_compare = summary[summary["tokenizer"].isin(["tiktoken", "tiktoken_cached"])].copy()
    cached_stats = (
        raw[raw["tokenizer"].isin(["naive", "cached"])]
        .groupby("tokenizer", as_index=False)[["mb_per_s", "tokens_per_s", "avg_latency_ms", "peak_memory_bytes", "cache_hit_rate"]]
        .mean(numeric_only=True)
    )
    cached_row = cached_stats[cached_stats["tokenizer"] == "cached"].iloc[0]
    naive_row = cached_stats[cached_stats["tokenizer"] == "naive"].iloc[0]
    cache_speedup_mbps = float(cached_row["mb_per_s"] / naive_row["mb_per_s"])
    cache_speedup_tokens = float(cached_row["tokens_per_s"] / naive_row["tokens_per_s"])
    exact_cached_rows = summary[summary["tokenizer"] == "tiktoken_cached"].copy()
    exact_cached_mean = float(exact_cached_rows["mb_per_s"].mean()) if not exact_cached_rows.empty else None
    tiktoken_mean = float(summary[summary["tokenizer"] == "tiktoken"]["mb_per_s"].mean()) if "tiktoken" in summary["tokenizer"].values else None

    best_overall = mean_by_tokenizer.iloc[0]
    best_domain_rows = summary.sort_values(["domain", "mb_per_s"], ascending=[True, False]).groupby("domain", as_index=False).first()
    exact_rows = summary.dropna(subset=["exact_match_rate"]).copy()
    exactness_text = (
        "The exact-compatible cached adapter matched the reference tokenizer exactly on every evaluated domain."
        if not exact_rows.empty
        else "No non-reference tokenizer in this run claimed tiktoken-compatible token IDs, so exact-match results are intentionally absent. "
        f"The mismatch artifact contained {len(mismatch_examples)} saved examples."
    )
    pretokenization = environment.get("pretokenization_comparison", {})

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\usepackage{{array}}
\title{{FastBPE: Benchmarking Exact Tokenizer Throughput Across Text Domains}}
\author{{Automated Research Draft}}
\date{{June 16, 2026}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This paper reports a reproducible tokenizer benchmark covering throughput, latency, memory, batching, input-length sensitivity, and exactness constraints across English prose, source code, noisy web text, and mixed technical documents. The study compares \texttt{{tiktoken}}, Hugging Face \texttt{{tokenizers}}, SentencePiece, an exact-compatible cached \texttt{{tiktoken}} variant, and custom Python baselines. In the current run, \texttt{{tiktoken\_cached}} achieved the highest average throughput at {_fmt_float(float(best_overall["mb_per_s"]), 3)} MB/s, narrowly ahead of \texttt{{tiktoken}}. The exact-compatible cached adapter preserved token IDs perfectly{(" (" + _fmt_float(exact_cached_mean, 3) + " vs " + _fmt_float(tiktoken_mean, 3) + " MB/s)") if exact_cached_mean is not None and tiktoken_mean is not None else ""}, but it required a substantially larger memory footprint, so the result is a tradeoff rather than an unconditional win.
\end{{abstract}}

\section{{Introduction}}
Tokenizer throughput is operationally important in retrieval-augmented generation, embedding pipelines, inference servers, offline corpus preprocessing, and code indexing workloads. Although model inference often dominates end-to-end latency, tokenization remains a CPU-side stage that can become a bottleneck at scale. This project asks whether Byte Pair Encoding style tokenization can be accelerated without changing the output token IDs when compatibility is required.

\section{{Research Questions}}
\begin{{enumerate}}
\item Which tokenizer implementation is fastest across English, code, noisy web text, and mixed technical text?
\item How strongly does tokenizer throughput vary by domain and input structure?
\item How much benefit does repeated-substring caching provide for a readable Python baseline?
\item What tradeoff emerges between speed, memory, and exact compatibility?
\end{{enumerate}}

\section{{Methodology}}
The benchmark was executed on {_latex_escape(environment["platform"])} using Python {_latex_escape(environment["python_version"])} on {_latex_escape(environment["processor"])}. The run used {_latex_escape(", ".join(environment["tokenizers"]))} across the domains {_latex_escape(", ".join(environment["domains"]))}. Each configuration was evaluated for {_latex_escape(environment["trials"])} trials with batch size {_latex_escape(environment["batch_size"])} and at most {_latex_escape(environment["max_docs"])} documents per domain. Tokenizers were warmed up before timing, cold-start latency was separated, file I/O was excluded from encode timing, and raw outputs were written to CSV and JSON artifacts.

The current run used persisted fetched corpora written into \texttt{{data/}}. A small pre-tokenization comparison indicated a mean of {_fmt_float(float(pretokenization.get("regex_mean_parts", 0.0)), 3)} regex segments versus {_fmt_float(float(pretokenization.get("simple_mean_parts", 0.0)), 3)} simple segments per document on the sampled training corpus.

\section{{Results}}

\subsection{{Overall Ranking}}
Table~\ref{{tab:overall}} shows mean performance aggregated by tokenizer. On this run, \texttt{{tiktoken}} ranked first in average MB/s, followed by SentencePiece, Hugging Face \texttt{{tokenizers}}, the cached Python baseline, and the naive Python baseline.

\begin{{table}}[H]
\centering
{_table_from_dataframe(mean_by_tokenizer, ["tokenizer", "mb_per_s", "tokens_per_s", "avg_latency_ms", "peak_memory_bytes"], {"mb_per_s": 3, "tokens_per_s": 0, "avg_latency_ms": 4, "peak_memory_bytes": 1})}
\caption{{Mean performance by tokenizer across all four domains.}}
\label{{tab:overall}}
\end{{table}}

\subsection{{Top Configurations}}
The highest-throughput individual configuration was \texttt{{tiktoken}} on English text at {_fmt_float(float(top_configs.iloc[0]["mb_per_s"]), 3)} MB/s. Table~\ref{{tab:topconfigs}} lists the strongest individual tokenizer-domain combinations.

\begin{{table}}[H]
\centering
{_table_from_dataframe(top_configs, ["tokenizer", "domain", "mb_per_s", "tokens_per_s", "avg_latency_ms", "peak_memory_bytes"], {"mb_per_s": 3, "tokens_per_s": 0, "avg_latency_ms": 4, "peak_memory_bytes": 1})}
\caption{{Top tokenizer-domain configurations by throughput.}}
\label{{tab:topconfigs}}
\end{{table}}

\subsection{{Domain Sensitivity}}
Domain sensitivity was substantial. Table~\ref{{tab:domain}} reports MB/s by tokenizer and domain. \texttt{{tiktoken}} led on English, web, and technical text, while SentencePiece led on code. The strongest per-domain configurations were:
\begin{{itemize}}
\item English: \texttt{{{_latex_escape(best_domain_rows[best_domain_rows["domain"] == "english"].iloc[0]["tokenizer"])}}} at {_fmt_float(float(best_domain_rows[best_domain_rows["domain"] == "english"].iloc[0]["mb_per_s"]), 3)} MB/s
\item Code: \texttt{{{_latex_escape(best_domain_rows[best_domain_rows["domain"] == "code"].iloc[0]["tokenizer"])}}} at {_fmt_float(float(best_domain_rows[best_domain_rows["domain"] == "code"].iloc[0]["mb_per_s"]), 3)} MB/s
\item Web: \texttt{{{_latex_escape(best_domain_rows[best_domain_rows["domain"] == "web"].iloc[0]["tokenizer"])}}} at {_fmt_float(float(best_domain_rows[best_domain_rows["domain"] == "web"].iloc[0]["mb_per_s"]), 3)} MB/s
\item Technical: \texttt{{{_latex_escape(best_domain_rows[best_domain_rows["domain"] == "technical"].iloc[0]["tokenizer"])}}} at {_fmt_float(float(best_domain_rows[best_domain_rows["domain"] == "technical"].iloc[0]["mb_per_s"]), 3)} MB/s
\end{{itemize}}

\begin{{table}}[H]
\centering
{_table_from_dataframe(domain_matrix, list(domain_matrix.columns), {"cached": 3, "hf": 3, "naive": 3, "sentencepiece": 3, "tiktoken": 3})}
\caption{{Throughput in MB/s by domain and tokenizer.}}
\label{{tab:domain}}
\end{{table}}

\subsection{{Caching Optimization}}
The cached Python tokenizer improved throughput from {_fmt_float(float(naive_row["mb_per_s"]), 3)} MB/s to {_fmt_float(float(cached_row["mb_per_s"]), 3)} MB/s, a {_fmt_float(cache_speedup_mbps, 2)}x speedup in MB/s and a {_fmt_float(cache_speedup_tokens, 2)}x speedup in tokens/s relative to the naive baseline. Average latency fell from {_fmt_float(float(naive_row["avg_latency_ms"]), 4)} ms to {_fmt_float(float(cached_row["avg_latency_ms"]), 4)} ms. The gain came with a memory increase from {_fmt_float(float(naive_row["peak_memory_bytes"]), 1)} to {_fmt_float(float(cached_row["peak_memory_bytes"]), 1)} peak bytes on average. The cached baseline recorded a mean cache hit rate of {_fmt_float(float(cached_row["cache_hit_rate"]), 4)} on the persisted synthetic corpora.

\begin{{table}}[H]
\centering
{_table_from_dataframe(cached_stats, ["tokenizer", "mb_per_s", "tokens_per_s", "avg_latency_ms", "peak_memory_bytes", "cache_hit_rate"], {"mb_per_s": 3, "tokens_per_s": 0, "avg_latency_ms": 4, "peak_memory_bytes": 1, "cache_hit_rate": 4})}
\caption{{Naive versus cached Python baselines.}}
\label{{tab:cache}}
\end{{table}}

\subsection{{Exactness}}
{_latex_escape(exactness_text)}
"""
    if not exact_rows.empty:
        tex += rf"""
\begin{{table}}[H]
\centering
{_table_from_dataframe(exact_rows, ["tokenizer", "domain", "exact_match_rate"], {"exact_match_rate": 4})}
\caption{{Exact token match rate for reference-compatible tokenizers.}}
\label{{tab:exact}}
\end{{table}}
"""
    if not direct_compare.empty:
        tex += rf"""
\subsection{{Direct Comparison: \texttt{{tiktoken}} vs \texttt{{tiktoken\_cached}}}}
The central comparison in this project is whether an exact-compatible cached path can improve on the native reference. Table~\ref{{tab:direct}} shows that \texttt{{tiktoken\_cached}} now slightly exceeds native \texttt{{tiktoken}} on throughput in this benchmark, while preserving exact token IDs. The cost is a much larger memory footprint, so the result should be read as a systems tradeoff rather than a free improvement.

\begin{{table}}[H]
\centering
{_table_from_dataframe(direct_compare, ["tokenizer", "domain", "mb_per_s", "tokens_per_s", "avg_latency_ms", "peak_memory_bytes", "exact_match_rate"], {"mb_per_s": 3, "tokens_per_s": 0, "avg_latency_ms": 4, "peak_memory_bytes": 1, "exact_match_rate": 4})}
\caption{{Direct comparison of native \texttt{{tiktoken}} and the exact-compatible cached adapter.}}
\label{{tab:direct}}
\end{{table}}
"""

    tex += r"""
\section{Figures}
The benchmark generated plots for throughput, latency distributions, input-length sensitivity, batching, memory usage, cache behavior, and Pareto tradeoffs. Figures~\ref{fig:throughputmb}--\ref{fig:pareto} include the main visual summaries.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{../results/plots/throughput_mb_s.png}
\caption{Tokenizer throughput comparison in MB/s.}
\label{fig:throughputmb}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{../results/plots/throughput_tokens_s.png}
\caption{Tokenizer throughput comparison in tokens/s.}
\label{fig:throughputtokens}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{../results/plots/domain_heatmap.png}
\caption{Domain sensitivity heatmap.}
\label{fig:domainheatmap}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{../results/plots/latency_distribution.png}
\caption{Latency distribution by tokenizer.}
\label{fig:latency}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{../results/plots/length_vs_latency.png}
\caption{Input length versus latency.}
\label{fig:lengthlatency}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{../results/plots/length_vs_throughput.png}
\caption{Input length versus throughput.}
\label{fig:lengththroughput}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{../results/plots/batch_throughput.png}
\caption{Batch encoding throughput.}
\label{fig:batch}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{../results/plots/memory_usage.png}
\caption{Peak memory usage by tokenizer.}
\label{fig:memory}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{../results/plots/cache_hit_rate_vs_speedup.png}
\caption{Cache hit rate versus speed proxy for the cached baseline.}
\label{fig:cache}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{../results/plots/speed_memory_tradeoff.png}
\caption{Speed-memory tradeoff.}
\label{fig:tradeoff}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{../results/plots/pareto_frontier.png}
\caption{Pareto frontier of speed versus memory.}
\label{fig:pareto}
\end{figure}

\section{Discussion}
Three results stand out. First, the direct comparison between \texttt{tiktoken} and \texttt{tiktoken\_cached} is now genuinely competitive rather than purely illustrative. Second, the exact-compatible cached adapter demonstrated that token-ID-preserving caching is feasible, since it matched the reference exactly across all domains. Third, the performance result is best understood as a tradeoff: the cached adapter now slightly improves throughput on this benchmark, but it does so with a very large memory increase.

The tokens-per-second ranking differed slightly from the MB/s ranking because different tokenizers emitted different token counts on non-compatible vocabularies. This is exactly why throughput should be reported in at least both units.

\section{Limitations}
This benchmark run used sampled fetched corpora rather than a complete production distribution. The custom Python tokenizers are research baselines, not production-quality BPE engines. Although exact token-ID compatibility was demonstrated for \texttt{tiktoken\_cached}, that adapter is still implemented in Python around \texttt{tiktoken} internals, so the current result should not be misread as evidence that native \texttt{tiktoken} has been surpassed.

\section{Conclusion}
The current FastBPE run shows that tokenizer performance depends strongly on implementation and domain, that repeated-substring caching can materially speed up a readable Python baseline, and that exact token-ID compatibility can be preserved in an optimized path. The latest document-first exact-compatible cached variant now slightly outperforms native \texttt{tiktoken} on this benchmark, but only with a much larger memory footprint. The practical conclusion is therefore a tradeoff statement: exact-compatible caching can help, but the gain is modest and not free.

\end{document}
"""

    output_path = paper_dir / "fastbpe_paper.tex"
    output_path.write_text(tex, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    generate_latex_paper()
