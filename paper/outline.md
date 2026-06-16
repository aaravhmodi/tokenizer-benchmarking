# FastBPE Paper Outline

## Abstract

Benchmark exact tokenizer throughput and evaluate a cached BPE-inspired optimization.

## Introduction

- Why tokenization speed matters for LLM systems
- Why exactness matters for reproducible inference and preprocessing

## Background

- Byte Pair Encoding
- Pre-tokenization and merge tables
- Throughput, latency, memory, and exactness metrics

## Methodology

- Tokenizers compared
- Datasets and domains
- Trial design, warmup, and cold-start controls
- Hardware/software environment

## Results

- Overall throughput
- Domain sensitivity
- Input-length scaling
- Batch versus single encoding
- Exactness compatibility
- Cache optimization tradeoffs

## Discussion

- Why specific tokenizers are faster
- Why domains differ
- When caching helps
- Speed-memory tradeoffs

## Limitations

- Runtime and hardware specificity
- Dataset realism
- Python baseline constraints

## Conclusion

- Conditions where exact faster tokenization is feasible

