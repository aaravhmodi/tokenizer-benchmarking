# FastBPE Project Journal

## Phase 1: Framework Build-Out

The project started as a benchmarking framework rather than an application. The original objective was to measure tokenizer throughput, latency, memory behavior, and output exactness across multiple text domains while keeping the benchmark reproducible.

The first milestone was structural:

- unified `TokenizerAdapter` interface
- benchmark runner with warmup, multiple trials, cold-start measurement, and CSV/JSON outputs
- synthetic dataset fallback across `english`, `code`, `web`, and `technical`
- plotting pipeline and report generation
- naive Python tokenizer and cached Python tokenizer prototypes

At this stage, the project could run end to end, but the results were only meaningful as framework validation.

## Phase 2: Synthetic Results and Early Optimization Signal

The initial benchmark runs used synthetic corpora. Those runs suggested:

- `tiktoken` performed strongly overall
- the cached Python tokenizer could outperform the naive Python tokenizer
- domain differences were measurable even in the synthetic setup

This phase was useful for verifying instrumentation, but it also exposed a methodological weakness: repeated synthetic documents artificially inflated cache hit rates and made the caching optimization look stronger than it really was.

## Phase 3: Better Synthetic Sampling

To reduce overfitting to repeated text, the synthetic corpus generator was improved:

- more varied document templates
- persisted synthetic files under `data/`
- clearer distinction among English prose, code, noisy web text, and technical markdown-like text

This changed the observed cache behavior and made it clear that dataset composition strongly affects optimization conclusions.

## Phase 4: Real Dataset Ingestion

The next milestone was replacing hand-prepared corpora with an automated real-data pipeline. A unified fetch script was added to populate `data/` from public online sources:

- Project Gutenberg for English prose
- curated GitHub raw files for code
- Hacker News public items for noisy web text
- arXiv API abstracts for technical text

This added a provenance manifest and moved the project from toy inputs to a realistic reproducible benchmark corpus.

## Phase 5: Real-Data Benchmarking

Once the benchmark was run on real datasets, the conclusions became more credible and more nuanced:

- `tiktoken` remained the strongest overall tokenizer
- domain effects remained real
- the Python substring cache did not consistently help on small real corpora
- after scaling the real corpus, the cached Python baseline again showed a measurable speedup over the naive Python baseline, but at much higher memory cost

This phase established a core systems result: optimization gains depend on corpus scale and repetition structure.

## Phase 6: Exact-Compatible Optimization

The hardest open problem was not just speed, but exactness. The earlier Python cached tokenizer was not vocabulary-compatible with `tiktoken`, so it could not answer the main research question directly.

To address that, a new adapter was added:

- `tiktoken_cached`

This adapter uses:

- `tiktoken`'s own `cl100k_base` vocabulary
- `tiktoken`'s regex piece boundaries
- `encode_single_piece` from the core BPE implementation
- an LRU cache for repeated pieces

This changed the project qualitatively. Exactness was no longer hypothetical; it became testable.

## Phase 7: Exactness Confirmed on Real Data

The exactness benchmark on the 128-document real corpus showed:

- `tiktoken_cached` matched `tiktoken` exactly
- exact match rate was `1.0` on `code`, `english`, `technical`, and `web`
- no mismatch examples were recorded

This is a major research milestone because the project can now support a real statement about exact-token compatibility for an optimized path.

## Phase 8: Corrected Benchmark Methodology

An important benchmarking error was discovered after the first exact-compatible caching experiments. Cache state had been allowed to persist across repeated benchmark trials, which made later trials unrealistically fast for whole-document caching.

The methodology was corrected by:

- clearing cache state between trials
- rerunning the benchmark on the full fetched corpus
- regenerating the paper and report after the correction

This matters because it changed the interpretation of the optimization results. The corrected benchmark showed that exact-compatible caching was still correct, but no longer produced the implausibly large speedup seen under the flawed setup.

## Phase 9: Document-First Exact-Compatible Caching

Profiling showed that the main bottleneck in the exact-compatible cached adapter was not cache misses, but Python-side overhead:

- regex traversal
- per-piece iteration
- cache bookkeeping around individual pieces

To address that, the exact-compatible adapter was revised to prefer a document-level cache backed by native `tiktoken.encode()` on cache misses. This design keeps exact token IDs while removing the slowest Python-side hot path. The piece-level cache remains in the codebase as an experimental mechanism, but the practical path now prioritizes native encoding plus memoization.

## Current State

The current evidence supports the following conclusions:

- exact compatibility is achievable with the `tiktoken_cached` design
- the latest document-first exact-compatible design slightly outperforms native `tiktoken` on the current benchmark
- that speed gain comes with a much larger memory footprint
- caching still improves the naive Python baseline
- domain and corpus size materially affect tokenizer behavior

So the project has progressed from:

1. benchmark scaffold
2. synthetic validation
3. automated real-data ingestion
4. real-data benchmarking
5. exact-compatible optimization
6. exactness verification
7. corrected fair benchmarking
8. document-first exact-compatible caching

## Next Stage

The next technical objective is no longer to prove exactness. That has been demonstrated for the current corpus. The next objective is to improve the exact-compatible optimized path enough to preserve or widen the current throughput edge while reducing memory cost.

That likely requires:

- reducing memory overhead in document caching
- improving cache representation and lookup cost
- testing cache-aware batching
- possibly moving exact-compatible caching into a lower-level implementation
