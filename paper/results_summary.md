# FastBPE: Benchmarking Exact Tokenizer Throughput Across Text Domains

## Abstract

FastBPE benchmarks tokenizer throughput, latency, memory, and output exactness across multiple text domains. The suite compares production tokenizer libraries against custom Python baselines and includes a repeated-substring cache prototype to test whether exact tokenization can be accelerated without changing outputs where compatibility is expected.

## Introduction

Tokenizer speed directly affects dataset preprocessing, RAG ingestion, embedding pipelines, and inference server throughput. This project isolates tokenization cost and studies how text type, input length, batching, and warm versus cold execution affect real throughput.

## Background

Byte Pair Encoding combines pre-tokenization with iterative merge logic to map text into token IDs. Faster tokenization is useful only when outputs remain correct for the intended vocabulary. For OpenAI-compatible experiments, `tiktoken` is the reference tokenizer.

## Methodology

- Tokenizers compared: tiktoken, tiktoken_cached, hf, sentencepiece, naive, cached
- Domains: code, english, technical, web
- Trials per configuration: 5
- Batch size: 8
- Platform: Windows-10-10.0.26200-SP0
- Python: 3.11.9

### Benchmark Summary

| tokenizer | domain | mode | mb_per_s | tokens_per_s | avg_latency_ms | peak_memory_bytes |
| --- | --- | --- | --- | --- | --- | --- |
| tiktoken | technical | single | 5.715343951355498 | 1153083.6369878757 | 0.2562895316714275 | 20807.0 |
| tiktoken | code | single | 5.035151178451908 | 1174885.5791157917 | 0.40453921851622 | 64324.0 |
| tiktoken | english | single | 4.479161483633531 | 1101243.4409602892 | 0.1689214062935207 | 31258.2 |
| tiktoken | web | single | 3.891763019061671 | 1078596.119244086 | 0.0482285938460336 | 22503.2 |
| tiktoken_cached | code | single | 2.782343231865716 | 649222.7985644138 | 0.7399565627338234 | 276448.6 |

## Results

### Fastest Configurations by MB/s

| tokenizer | domain | mb_per_s | tokens_per_s | avg_latency_ms |
| --- | --- | --- | --- | --- |
| tiktoken | technical | 5.715343951355498 | 1153083.6369878757 | 0.2562895316714275 |
| tiktoken | code | 5.035151178451908 | 1174885.5791157917 | 0.40453921851622 |
| tiktoken | english | 4.479161483633531 | 1101243.4409602892 | 0.1689214062935207 |
| tiktoken | web | 3.891763019061671 | 1078596.119244086 | 0.0482285938460336 |
| tiktoken_cached | code | 2.782343231865716 | 649222.7985644138 | 0.7399565627338234 |

### Exactness

| tokenizer | domain | exact_match_rate |
| --- | --- | --- |
| tiktoken_cached | code | 1.0 |
| tiktoken_cached | english | 1.0 |
| tiktoken_cached | technical | 1.0 |
| tiktoken_cached | web | 1.0 |

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
