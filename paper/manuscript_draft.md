# FastBPE: Benchmarking Exact Tokenizer Throughput Across Text Domains

## Abstract

Tokenization is usually treated as a lightweight preprocessing step, but in retrieval-augmented generation, embedding pipelines, offline corpus preparation, and inference-serving systems, tokenizer throughput can become a measurable CPU bottleneck. This paper presents FastBPE, a reproducible benchmark suite for measuring tokenizer throughput, latency, memory usage, batching behavior, and exact token-ID compatibility across English prose, source code, noisy web text, and technical text. FastBPE compares production tokenizers with custom Python baselines and evaluates an exact-compatible cached `tiktoken` prototype. On the current benchmark corpus, the exact-compatible `tiktoken_cached` adapter achieves perfect token-ID agreement with `tiktoken` across all tested domains and slightly improves average throughput, but only with a substantially larger memory footprint. These results show that exact tokenizer acceleration is possible, but the practical outcome is a systems tradeoff rather than a free performance win.

## 1. Introduction

Large language model systems do not begin with matrix multiplication; they begin with tokenization. Before an inference server can run a prompt, before an embedding job can index a corpus, and before a retrieval pipeline can chunk a knowledge base, raw text must be converted into token IDs. At small scale this step is easy to ignore. At large scale it becomes a recurring CPU workload whose latency and throughput can materially affect end-to-end system behavior.

This matters in several settings. Retrieval-augmented generation pipelines tokenize large document collections during ingestion. Embedding services tokenize millions of chunks prior to vectorization. Offline training and preprocessing jobs tokenize corpora whose total size is measured in gigabytes or terabytes. Interactive inference systems tokenize user prompts on the critical path. In all of these cases, even modest differences in tokenizer throughput compound into real operational cost.

Despite this, tokenizer benchmarking is often informal. Many comparisons focus on a single throughput metric, a single dataset, or a single implementation language. Fewer evaluations ask the harder systems question: can tokenization be made faster without changing the output token IDs when exact compatibility is required?

FastBPE addresses that question directly. The project provides a common adapter interface for multiple tokenizers, runs them on the same corpora, measures throughput, latency, memory, batch behavior, and correctness, and logs raw outputs for reproducibility. Most importantly, it investigates an exact-compatible cached path for `tiktoken`, the reference tokenizer used in this project for OpenAI-style experiments.

The main contributions of this work are:

1. A reproducible tokenizer benchmark suite spanning English prose, code, noisy web text, and technical text.
2. A unified evaluation methodology covering MB/s, tokens/s, latency, cold start, memory usage, batch behavior, and exactness.
3. A progression from synthetic validation to automatically fetched real-world corpora.
4. An exact-compatible cached `tiktoken` prototype that preserves token IDs perfectly on the evaluated corpus.
5. Evidence that exact-compatible caching can slightly improve throughput on this benchmark, but with a substantial memory cost.

The central claim of this paper is therefore deliberately nuanced: exact tokenizer optimization is possible, but it is a correctness-constrained systems tradeoff rather than a pure speed problem.

## 2. Background

### 2.1 Byte Pair Encoding

Byte Pair Encoding (BPE) and closely related subword tokenization schemes represent text as sequences of learned token units rather than as raw characters or whitespace-delimited words. In practice, modern tokenizer implementations combine a vocabulary with merge behavior or equivalent segmentation rules so that repeated substrings can be encoded as stable token IDs.

For LLM systems, the tokenizer is not merely a parsing convenience. The exact integer token IDs determine the input seen by the model. Two tokenizers that produce different IDs are not interchangeable, even if they produce similar token counts.

### 2.2 Pre-tokenization

Real tokenizer implementations rarely operate on arbitrary strings with a single flat merge loop. They usually begin with a pre-tokenization or segmentation stage that breaks text into regex-defined or byte-level pieces. That stage matters because text domains differ dramatically:

- English prose has relatively regular word and punctuation structure.
- Code contains identifiers, symbols, delimiters, and repeated syntax.
- Technical text mixes natural language with markup, equations, and lists.
- Noisy web text includes URLs, HTML fragments, punctuation irregularities, and inconsistent formatting.

These differences affect both token boundaries and cache reuse behavior.

### 2.3 Exact Token Compatibility

Fast tokenizer comparisons are easy to overstate when different vocabularies are involved. A tokenizer can be “fast” simply because it emits different token sequences. For that reason, this project treats `tiktoken` as the reference tokenizer for OpenAI-style compatibility experiments and only interprets exact token-ID equality when a candidate tokenizer is explicitly intended to match the same vocabulary and segmentation semantics.

### 2.4 Metrics

FastBPE measures:

- throughput in MB/s
- throughput in tokens/s
- documents per second
- average latency
- p50, p95, and p99 latency
- peak memory usage
- token count differences for non-compatible tokenizers
- exact match rate for compatible tokenizers
- cache hit and miss statistics for cached variants

This mix is important because no single metric is sufficient. A tokenizer can be faster in tokens/s yet slower in MB/s, or slightly faster overall while using dramatically more memory.

## 3. System Design

### 3.1 Benchmark Architecture

FastBPE is organized around a common `TokenizerAdapter` interface. Every tokenizer exposes an `encode` method for single texts and an `encode_batch` method for batched inputs. This design allows benchmark logic to remain independent of individual libraries.

The system includes:

- tokenizer adapters under `fastbpe/adapters/`
- benchmark logic under `fastbpe/benchmarks/`
- dataset loading and fetching under `fastbpe/datasets/`
- plotting utilities under `fastbpe/plots/`
- command-line entry points under `scripts/`

### 3.2 Tokenizers Evaluated

The current project evaluates:

- `tiktoken`
- `tiktoken_cached`
- Hugging Face `tokenizers`
- SentencePiece
- `naive` Python tokenizer
- `cached` Python tokenizer

Only `tiktoken_cached` is intended to be exact-compatible with the `tiktoken` reference. The other libraries are throughput comparators or research baselines rather than vocabulary-compatible replacements.

### 3.3 Dataset Handling

The benchmark supports four domains:

- `english`
- `code`
- `web`
- `technical`

The project began with synthetic corpora for pipeline validation, then added an automated fetcher for public real-world sources:

- Project Gutenberg for English prose
- curated GitHub raw files for code
- Hacker News items for noisy web text
- arXiv abstracts for technical text

Each fetched corpus is written into `data/` and tracked with a provenance manifest.

### 3.4 Result Logging and Paper Generation

FastBPE writes raw and aggregated outputs to CSV and JSON, generates plots automatically, records environment metadata, and produces both Markdown and LaTeX/PDF paper artifacts. This matters because benchmarking claims are only useful when the underlying measurements are reproducible and inspectable.

## 4. Methodology

### 4.1 Research Question

The primary research question is:

**Can BPE tokenization be accelerated without changing the output token IDs?**

Supporting questions are:

1. Which tokenizer is fastest across English, code, web, and technical text?
2. How much does performance change by domain?
3. Does batching materially improve throughput?
4. Can caching improve tokenizer speed while preserving exact compatibility?
5. What tradeoff appears between throughput and memory usage?

### 4.2 Benchmark Setup

The current benchmark run uses:

- platform: Windows 10 (`environment.json`)
- Python: 3.11.9
- tokenizer set: `tiktoken`, `tiktoken_cached`, `hf`, `sentencepiece`, `naive`, `cached`
- domains: `code`, `english`, `technical`, `web`
- trials per configuration: 5
- batch size: 8
- maximum documents per domain in the latest large run: 256

### 4.3 Fairness Controls

The benchmark uses the following controls:

- identical input texts for all tokenizers in a given run
- tokenizer warmup before timing
- cold-start timing recorded separately
- file loading excluded from timed encoding
- five repeated trials per configuration
- raw per-trial outputs saved to disk
- environment metadata saved alongside results
- exactness validated where compatibility is claimed

An important methodological correction was made during the project: cache state is now reset between trials so cached adapters are not given an unfair advantage from previous-trial whole-document reuse.

### 4.4 Exact-Compatible Cached Design

The first exact-compatible cached prototype wrapped `tiktoken` piece-by-piece using the same vocabulary and piece boundaries. That proved exactness but introduced enough Python overhead to trail native `tiktoken`.

The current exact-compatible design is document-first:

1. On a cache hit, return the cached token IDs for the full document.
2. On a cache miss, call native `tiktoken.encode()` directly.
3. Store the resulting exact token IDs in a bounded document cache.

This design preserves exact token IDs while avoiding slower Python-side piece orchestration on the miss path.

## 5. Results

### 5.1 Overall Throughput

In the latest 256-document-per-domain benchmark run, the mean MB/s ranking is:

1. `tiktoken_cached`: 5.253 MB/s
2. `tiktoken`: 5.167 MB/s
3. `sentencepiece`: 3.058 MB/s
4. `hf`: 3.015 MB/s
5. `naive`: 1.365 MB/s
6. `cached`: 0.788 MB/s

The top result is therefore no longer a simple “native reference wins” story. The exact-compatible cached path now slightly exceeds `tiktoken` on this benchmark.

### 5.2 Direct `tiktoken` vs `tiktoken_cached` Comparison

This is the most important comparison in the paper because it isolates the exactness-constrained optimization question.

Per domain:

- `technical`: `tiktoken_cached` 6.376 MB/s vs `tiktoken` 6.291 MB/s
- `code`: `tiktoken_cached` 5.134 MB/s vs `tiktoken` 4.731 MB/s
- `english`: `tiktoken_cached` 4.985 MB/s vs `tiktoken` 5.138 MB/s
- `web`: `tiktoken_cached` 4.517 MB/s vs `tiktoken` 4.509 MB/s

This pattern suggests that the cached exact-compatible path does not uniformly dominate the native reference. It wins in some domains, loses slightly in others, and remains very close overall.

### 5.3 Exactness

Exactness results are strong and unambiguous for the compatible path:

- `tiktoken_cached` exact match rate: 1.0 on `code`
- `tiktoken_cached` exact match rate: 1.0 on `english`
- `tiktoken_cached` exact match rate: 1.0 on `technical`
- `tiktoken_cached` exact match rate: 1.0 on `web`

This matters because it answers the core correctness question directly. The optimized path is not merely “similar”; it is exact on the evaluated corpus.

### 5.4 Domain Sensitivity

Performance depends on domain:

- technical text produces the highest MB/s for both `tiktoken` and `tiktoken_cached`
- code remains competitive but carries more structural complexity
- web text is noisier and less uniform
- English prose remains strong but not always dominant

This supports the claim that tokenizer performance should not be summarized by a single aggregate number without considering workload structure.

### 5.5 Naive vs Cached Python Baselines

The custom Python baselines remain useful because they separate the “can caching help in principle?” question from the exact-compatibility question.

In the latest run:

- `naive`: 1.365 MB/s
- `cached`: 0.788 MB/s

So the earlier cached Python baseline is no longer the strongest caching story in the repo. It is dominated by overhead in the current corpus and implementation. The more meaningful result is now the exact-compatible cached path.

### 5.6 Memory Tradeoff

The most important caveat in the current result is memory:

- `tiktoken` mean peak memory: about 120,948 bytes
- `tiktoken_cached` mean peak memory: about 2,233,979 bytes

That is a very large increase for a relatively small throughput gain. Any practical claim about tokenizer acceleration must therefore be framed as a speed-memory tradeoff, not a pure speed win.

## 6. Discussion

### 6.1 What We Wanted to Find Out

The project began with a systems question: can exact BPE tokenization be accelerated without changing token IDs?

That question turned out to require two separate achievements:

1. preserving exact correctness
2. improving speed under realistic workloads

The first achievement has now been demonstrated clearly. The second has been demonstrated only modestly, and only with a sizable memory cost.

### 6.2 Why Exact-Compatible Caching Matters

Without an exact-compatible tokenizer, the benchmark would only answer whether “some cache makes some tokenizer faster.” That is not enough for model-serving systems that require drop-in tokenizer equivalence.

`tiktoken_cached` matters because it shows that an optimization layer can preserve exact token IDs while still participating in a throughput comparison against the native reference.

### 6.3 Why the Result Is a Tradeoff

The exact-compatible cached path now slightly outperforms `tiktoken` on this benchmark, but the gain is small and the memory increase is large. This means the practical question is not “is caching faster?” but “when is this tradeoff worth it?”

Caching is most attractive when:

- documents repeat
- prompts or chunks have boilerplate structure
- the workload is dominated by recurring text templates
- available memory is less constrained than CPU time

Caching is less attractive when:

- inputs are highly diverse
- memory is constrained
- the gain from reuse is small relative to cache cost

### 6.4 Why the Earlier Result Changed

Earlier in the project, the exact-compatible cached path was slower. That was not only because caching was a bad idea, but because the implementation path was too expensive on the Python side. Once the design moved to a document-first strategy backed by native `tiktoken.encode()` on misses, much of that overhead was removed.

This is a useful research result on its own: the location of the optimization matters. A cache wrapped around the wrong abstraction level can lose even when reuse exists.

### 6.5 Implications

For RAG and preprocessing pipelines, the project suggests:

- exact-compatible caching is plausible when repeated text structure is present
- native tokenizer implementations remain extremely strong baselines
- small throughput gains must be evaluated alongside memory cost
- implementation language and abstraction level matter as much as the optimization idea itself

## 7. Limitations

This work has several limitations:

1. Results are hardware- and runtime-specific.
2. The fetched corpora are real but still sampled rather than exhaustive production distributions.
3. The exact-compatible cached path is implemented in Python around `tiktoken`, not in a lower-level language.
4. The measured gain is modest and may shift with different corpora or cache policies.
5. Hugging Face and SentencePiece are throughput comparators, not exact ID replacements for `tiktoken`.
6. Larger or more repetitive industrial workloads may change the observed tradeoff further.

## 8. Future Work

The strongest next steps are:

1. Reduce the memory cost of document-level exact-compatible caching.
2. Add cache-aware batching for repeated-document workloads.
3. Evaluate larger corpora and more repeated production-like prompt templates.
4. Move the exact-compatible cached path into a lower-level implementation such as Rust or C++.
5. Explore hybrid strategies that combine document-level and piece-level reuse without reintroducing high Python overhead.

## 9. Conclusion

FastBPE began as a tokenizer benchmark and became a correctness-constrained systems study. The project now shows that exact token-ID compatibility can be preserved in an optimized path and that a document-first cached strategy can slightly outperform native `tiktoken` on the current benchmark. However, the gain is modest and the memory cost is large. The right conclusion is therefore not that tokenizer acceleration is solved, but that exact-compatible acceleration is possible and worth pursuing when the throughput-memory tradeoff matches the workload.

## Appendix: Repo Artifacts

The following artifacts support the paper:

- `results/benchmark_summary.csv`
- `results/benchmark_raw.csv`
- `results/memory_results.csv`
- `results/length_scaling.csv`
- `results/batch_results.csv`
- `results/exactness_report.json`
- `results/environment.json`
- `results/plots/`
- `paper/project_journal.md`
- `paper/equations.tex`
