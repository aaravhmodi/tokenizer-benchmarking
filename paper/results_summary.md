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
| tiktoken_cached | technical | single | 6.376365956809692 | 1290529.145360713 | 0.2312619541044114 | 2472672.0 |
| tiktoken | technical | single | 6.291333413663785 | 1273319.1897249734 | 0.2343800001654017 | 35843.8 |
| tiktoken | english | single | 5.137940945588106 | 1267727.9277666814 | 0.1387139845519414 | 46248.0 |
| tiktoken_cached | code | single | 5.133743281579646 | 1219410.2639649643 | 0.4308817969103984 | 4470460.0 |
| tiktoken_cached | english | single | 4.98460850642769 | 1229894.9091674387 | 0.1431261719517351 | 1456340.0 |

## Results

### Fastest Configurations by MB/s

| tokenizer | domain | mb_per_s | tokens_per_s | avg_latency_ms |
| --- | --- | --- | --- | --- |
| tiktoken_cached | technical | 6.376365956809692 | 1290529.145360713 | 0.2312619541044114 |
| tiktoken | technical | 6.291333413663785 | 1273319.1897249734 | 0.2343800001654017 |
| tiktoken | english | 5.137940945588106 | 1267727.9277666814 | 0.1387139845519414 |
| tiktoken_cached | code | 5.133743281579646 | 1219410.2639649643 | 0.4308817969103984 |
| tiktoken_cached | english | 4.98460850642769 | 1229894.9091674387 | 0.1431261719517351 |

### Exactness

| tokenizer | domain | exact_match_rate |
| --- | --- | --- |
| tiktoken_cached | code | 1.0 |
| tiktoken_cached | english | 1.0 |
| tiktoken_cached | technical | 1.0 |
| tiktoken_cached | web | 1.0 |

### Direct Comparison: `tiktoken` vs `tiktoken_cached`

| tokenizer | domain | mb_per_s | tokens_per_s | avg_latency_ms | peak_memory_bytes | exact_match_rate |
| --- | --- | --- | --- | --- | --- | --- |
| tiktoken | code | 4.731345113298107 | 1123829.23669314 | 0.4681530470861617 | 370000.8 | nan |
| tiktoken | english | 5.137940945588106 | 1267727.9277666814 | 0.1387139845519414 | 46248.0 | nan |
| tiktoken | technical | 6.291333413663785 | 1273319.1897249734 | 0.2343800001654017 | 35843.8 | nan |
| tiktoken | web | 4.509071481018413 | 1181075.777731079 | 0.0518363277024036 | 31697.6 | nan |
| tiktoken_cached | code | 5.133743281579646 | 1219410.2639649643 | 0.4308817969103984 | 4470460.0 | 1.0 |
| tiktoken_cached | english | 4.98460850642769 | 1229894.9091674387 | 0.1431261719517351 | 1456340.0 | 1.0 |
| tiktoken_cached | technical | 6.376365956809692 | 1290529.145360713 | 0.2312619541044114 | 2472672.0 | 1.0 |
| tiktoken_cached | web | 4.516598645626086 | 1183047.3924705086 | 0.0517535152994241 | 536444.0 | 1.0 |

## Discussion

The main result should be interpreted jointly across throughput, latency spread, and memory usage. Domain-specific slowdowns are expected because code, noisy text, and markdown drive different pre-tokenization behavior and token boundary density. The cached prototype is designed to help repeated-token workloads and should be evaluated against its memory overhead and hit rate.

The exact-compatible result is now more concrete: `tiktoken_cached` preserves `tiktoken` token IDs exactly on the benchmark corpus. Its mean throughput was 5.253 MB/s versus 5.167 MB/s for `tiktoken`. The speed advantage is modest, while memory usage is dramatically higher, so the right interpretation is a tradeoff rather than a free optimization.

## Proposed Solution

The optimized tokenizer adds a repeated-substring cache on top of a simple Python baseline. This preserves deterministic output for its own vocabulary while reducing repeated lookup work for recurring words or symbols. The benchmark framework records cache hits, misses, and evictions so speedup can be related to workload structure instead of treated as a black box.

## Limitations

- Results are hardware- and Python-runtime-specific.
- Real fetched corpora are more representative than the synthetic fallback, but they are still a sampled benchmark corpus rather than a complete production distribution.
- Non-compatible vocabularies cannot be judged on exact token ID equality.
- The exact-compatible cached tokenizer is implemented in Python around `tiktoken` internals, so it is not yet a fair substitute for a lower-level native optimization.

## Conclusion

FastBPE now shows two distinct outcomes: exact token-ID compatibility can be preserved in an optimized path, and with a document-first caching strategy that path can slightly outperform native `tiktoken` on this benchmark. The cost is much higher memory usage, so the benchmark’s main value is in clarifying when that tradeoff might or might not be worth it.
