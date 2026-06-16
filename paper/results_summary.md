# FastBPE: Benchmarking Exact Tokenizer Throughput Across Text Domains

## Abstract

FastBPE benchmarks tokenizer throughput, latency, memory, and output exactness across multiple text domains. The suite compares production tokenizer libraries against custom Python baselines and includes a repeated-substring cache prototype to test whether exact tokenization can be accelerated without changing outputs where compatibility is expected.

## Introduction

Tokenizer speed directly affects dataset preprocessing, RAG ingestion, embedding pipelines, and inference server throughput. This project isolates tokenization cost and studies how text type, input length, batching, and warm versus cold execution affect real throughput.

## Background

Byte Pair Encoding combines pre-tokenization with iterative merge logic to map text into token IDs. Faster tokenization is useful only when outputs remain correct for the intended vocabulary. For OpenAI-compatible experiments, `tiktoken` is the reference tokenizer.

## Methodology

- Tokenizers compared: tiktoken, hf, sentencepiece, naive, cached
- Domains: code, english, technical, web
- Trials per configuration: 5
- Batch size: 8
- Platform: Windows-10-10.0.26200-SP0
- Python: 3.11.9

### Benchmark Summary

| tokenizer | domain | mode | mb_per_s | tokens_per_s | avg_latency_ms | peak_memory_bytes |
| --- | --- | --- | --- | --- | --- | --- |
| sentencepiece | code | single | 6.325406985833624 | 1575259.114449651 | 0.0234781249673687 | 3192.0 |
| sentencepiece | web | single | 6.272883106886823 | 1715894.263483554 | 0.0202893750156363 | 3000.0 |
| tiktoken | english | single | 6.263450213790594 | 944375.6769298372 | 0.0226637499054049 | 3285.6 |
| sentencepiece | english | single | 5.989888638739555 | 862077.9271526426 | 0.0241687504967558 | 3179.2 |
| sentencepiece | technical | single | 5.238176013694884 | 1234297.8992665452 | 0.0316499999826191 | 3544.0 |

## Results

### Fastest Configurations by MB/s

| tokenizer | domain | mb_per_s | tokens_per_s | avg_latency_ms |
| --- | --- | --- | --- | --- |
| sentencepiece | code | 6.325406985833624 | 1575259.114449651 | 0.0234781249673687 |
| sentencepiece | web | 6.272883106886823 | 1715894.263483554 | 0.0202893750156363 |
| tiktoken | english | 6.263450213790594 | 944375.6769298372 | 0.0226637499054049 |
| sentencepiece | english | 5.989888638739555 | 862077.9271526426 | 0.0241687504967558 |
| sentencepiece | technical | 5.238176013694884 | 1234297.8992665452 | 0.0316499999826191 |

### Exactness

No exact-match-compatible non-reference tokenizer results were recorded.

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
