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
| tiktoken | technical | single | 5.5417095090741855 | 1106086.7430081873 | 0.2630000003591703 | 14575.2 |
| tiktoken | code | single | 4.988159436529007 | 1168625.5351913825 | 0.4092462498192617 | 19938.4 |
| tiktoken | english | single | 4.784466855633031 | 1180900.284363235 | 0.1878293744994152 | 21719.2 |
| hf | code | single | 3.9643637908973326 | 870805.8940640491 | 0.5172899996068736 | 20040.0 |
| hf | technical | single | 3.908434535651848 | 838216.3012122038 | 0.3738787509064423 | 16108.0 |

## Results

### Fastest Configurations by MB/s

| tokenizer | domain | mb_per_s | tokens_per_s | avg_latency_ms |
| --- | --- | --- | --- | --- |
| tiktoken | technical | 5.5417095090741855 | 1106086.7430081873 | 0.2630000003591703 |
| tiktoken | code | 4.988159436529007 | 1168625.5351913825 | 0.4092462498192617 |
| tiktoken | english | 4.784466855633031 | 1180900.284363235 | 0.1878293744994152 |
| hf | code | 3.9643637908973326 | 870805.8940640491 | 0.5172899996068736 |
| hf | technical | 3.908434535651848 | 838216.3012122038 | 0.3738787509064423 |

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
