from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SyntheticDatasetConfig:
    docs_per_domain: int = 64


def build_synthetic_corpora(config: SyntheticDatasetConfig | None = None) -> dict[str, list[str]]:
    config = config or SyntheticDatasetConfig()
    english = [
        (
            "Tokenizer benchmarks need reproducible prose samples with stable punctuation, "
            "moderate sentence length, and realistic vocabulary shifts across documents."
        )
        for _ in range(config.docs_per_domain)
    ]
    code = [
        (
            "def benchmark_tokenizer(texts):\n"
            "    total = 0\n"
            "    for item in texts:\n"
            "        total += len(item.encode('utf-8'))\n"
            "    return {'bytes': total, 'docs': len(texts)}\n"
        )
        for _ in range(config.docs_per_domain)
    ]
    web = [
        (
            "<div class='post'>lol this tokenizer is FAST!!! 😅😅 visit /benchmarks?id=42 "
            "&amp; tell me why punctuation...breaks??? #nlp #bpe</div>"
        )
        for _ in range(config.docs_per_domain)
    ]
    technical = [
        (
            "# FastBPE Notes\n"
            "- Throughput matters for RAG preprocessing.\n"
            "- Let T(n) denote total tokenization time.\n"
            "- Compare p95 latency for markdown, equations like O(n log n), and tables.\n"
        )
        for _ in range(config.docs_per_domain)
    ]
    return {
        "english": english,
        "code": code,
        "web": web,
        "technical": technical,
    }

