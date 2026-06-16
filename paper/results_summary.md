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
| tiktoken | technical | single | 5.909159736932676 | 1225606.419004629 | 0.2545349991123657 | 12927.2 |
| tiktoken | code | single | 4.850739983926603 | 1144326.0583846294 | 23.281377142744272 | 3254357.6 |
| tiktoken | english | single | 4.643709831299247 | 1147159.5712394137 | 0.1863049994426546 | 12651.2 |
| tiktoken | web | single | 3.7154968729576607 | 997306.2319129856 | 0.0303625012747943 | 2155.2 |
| hf | technical | single | 3.1346939596567718 | 917873.3578391684 | 0.4803649993846193 | 16944.0 |

## Results

### Fastest Configurations by MB/s

| tokenizer | domain | mb_per_s | tokens_per_s | avg_latency_ms |
| --- | --- | --- | --- | --- |
| tiktoken | technical | 5.909159736932676 | 1225606.419004629 | 0.2545349991123657 |
| tiktoken | code | 4.850739983926603 | 1144326.0583846294 | 23.281377142744272 |
| tiktoken | english | 4.643709831299247 | 1147159.5712394137 | 0.1863049994426546 |
| tiktoken | web | 3.7154968729576607 | 997306.2319129856 | 0.0303625012747943 |
| hf | technical | 3.1346939596567718 | 917873.3578391684 | 0.4803649993846193 |

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
