# FastBPE: Benchmarking and Optimizing Tokenizer Speed

FastBPE is a reproducible ML systems benchmark for studying whether Byte Pair Encoding style tokenization can be accelerated without changing token outputs where exact compatibility is required. The project compares multiple tokenizer backends across text domains, measures throughput and latency, tracks memory usage, and prototypes optimization ideas such as repeated-substring caching.

## Why tokenizer speed matters

Tokenizer throughput is a real systems bottleneck in:

- large-scale corpus preprocessing
- embedding and RAG ingestion pipelines
- inference servers that tokenize user prompts online
- code and document indexing pipelines

Small per-document latency differences compound quickly at scale, especially when tokenization runs on CPUs in front of expensive GPU inference.

## Research question

Can BPE tokenization be accelerated without changing the output token IDs?

Supporting questions:

- Which tokenizer is fastest across English, code, web, and technical text?
- How sensitive is throughput to text domain and input length?
- How much time is spent in pre-tokenization versus token lookup?
- When does repeated-substring caching provide useful speedup?

## Repository layout

```text
data/
  english/
  code/
  web/
  technical/
fastbpe/
  adapters/
  benchmarks/
  datasets/
  plots/
  utils/
results/
  plots/
scripts/
paper/
```

## Supported tokenizers

- `tiktoken`
- Hugging Face `tokenizers`
- SentencePiece
- custom naive Python tokenizer baseline
- custom cached Python tokenizer baseline

The project currently treats `tiktoken` as the OpenAI-compatible reference. Exact token ID comparison is only meaningful for tokenizers that intentionally match the same vocabulary and merge behavior.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset setup

Place real text samples in:

```text
data/english
data/code
data/web
data/technical
```

Supported file types include `.txt`, `.md`, `.py`, `.js`, `.ts`, `.json`, and `.html`.

If no real files are present, FastBPE falls back to synthetic corpora so the benchmark pipeline remains runnable end to end.
For controlled local runs, `scripts/prepare_datasets.py` materializes the synthetic corpora into files under `data/` so the exact same document set can be inspected and reused.

## Benchmark commands

```bash
python scripts/prepare_datasets.py --docs-per-domain 64 --overwrite
python scripts/run_all_benchmarks.py --dataset all --trials 5 --enable-memory-profiler
python scripts/run_all_benchmarks.py --dataset code web --tokenizers tiktoken hf naive cached --trials 3 --max-docs 32
python scripts/run_exactness_tests.py --reference tiktoken --tokenizers tiktoken
python scripts/generate_report.py
```

## Methodology

- Warm up each tokenizer before timing.
- Separate cold-start timing from steady-state timing.
- Use the same input set for every tokenizer in a given run.
- Run multiple trials and save raw results.
- Exclude file loading from timing.
- Report throughput, latency, token counts, and memory.

## Outputs

Benchmark artifacts are written to `results/`:

- `benchmark_raw.csv`
- `benchmark_summary.csv`
- `memory_results.csv`
- `length_scaling.csv`
- `batch_results.csv`
- `mismatch_examples.json`
- `environment.json`
- `plots/*.png`

## Example result categories

The benchmark generates:

- throughput comparison in MB/s and tokens/s
- latency distribution plots
- tokenizer-by-domain heatmaps
- input-length scaling plots
- memory usage plots
- speed-memory tradeoff and Pareto frontier plots

## Custom tokenizer prototypes

### Naive baseline

The naive adapter is intentionally simple and readable. It uses regex pre-tokenization and a character-level token assignment scheme so performance instrumentation is easy to inspect.

### Cached baseline

The cached adapter adds repeated-substring memoization with LRU eviction. It reports:

- cache hits
- cache misses
- cache evictions
- cache hit rate

This is an optimization testbed, not a production-faithful reimplementation of OpenAI BPE.

## Limitations

- Synthetic corpora are used by default when real data is unavailable.
- Exact token ID matching currently requires an intentionally compatible tokenizer.
- Python baselines are useful for instrumentation but will not match optimized native implementations.
- SentencePiece and Hugging Face adapters are benchmark comparators, not vocabulary-compatible replacements for `tiktoken`.

## Reproducibility

Do not trust single-run benchmark numbers. For publishable results, record:

- hardware and OS details
- Python version
- installed tokenizer library versions
- warmup policy
- trial count
- dataset composition

FastBPE writes environment metadata alongside benchmark results to support this.
