from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle


@dataclass
class SyntheticDatasetConfig:
    docs_per_domain: int = 64


def _repeat_to_length(items: list[str], count: int) -> list[str]:
    return [item for _, item in zip(range(count), cycle(items))]


def build_synthetic_corpora(config: SyntheticDatasetConfig | None = None) -> dict[str, list[str]]:
    config = config or SyntheticDatasetConfig()
    english = _repeat_to_length(
        [
            (
                "Tokenizer benchmarks need reproducible prose samples with stable punctuation, "
                "moderate sentence length, and vocabulary that shifts as topics move from economics "
                "to science reporting and contemporary nonfiction."
            ),
            (
                "A clean English corpus should include declarative sentences, quotations, commas, "
                "and named entities so that token boundary behavior is not biased toward trivial text."
            ),
            (
                "Public-domain style narrative prose remains useful because it mixes dialogue, "
                "descriptions, and paragraph structure without the markup noise seen in web archives."
            ),
            (
                "Benchmark fairness depends on holding the document set fixed while running repeated "
                "trials, separating file I/O from encode time, and recording the environment exactly."
            ),
        ],
        config.docs_per_domain,
    )
    code = _repeat_to_length(
        [
            (
                "def benchmark_tokenizer(texts):\n"
                "    total_bytes = 0\n"
                "    for item in texts:\n"
                "        total_bytes += len(item.encode('utf-8'))\n"
                "    return {'bytes': total_bytes, 'docs': len(texts)}\n"
            ),
            (
                "export function longestPrefixLookup(input: string, trie: TrieNode): number[] {\n"
                "  const result: number[] = [];\n"
                "  let node = trie;\n"
                "  for (const ch of input) {\n"
                "    if (!node.children[ch]) break;\n"
                "    node = node.children[ch];\n"
                "    if (node.tokenId !== undefined) result.push(node.tokenId);\n"
                "  }\n"
                "  return result;\n"
                "}\n"
            ),
            (
                "{\n"
                "  \"experiment\": \"fastbpe\",\n"
                "  \"domains\": [\"english\", \"code\", \"web\", \"technical\"],\n"
                "  \"metrics\": {\"latency_ms\": [0.2, 0.4], \"throughput_mb_s\": 12.5},\n"
                "  \"cache\": {\"enabled\": true, \"capacity\": 4096}\n"
                "}\n"
            ),
            (
                "async function encodeBatch(docs) {\n"
                "  return Promise.all(docs.map(async (doc) => ({\n"
                "    bytes: Buffer.byteLength(doc, 'utf8'),\n"
                "    preview: doc.slice(0, 32),\n"
                "  })));\n"
                "}\n"
            ),
        ],
        config.docs_per_domain,
    )
    web = _repeat_to_length(
        [
            (
                "<div class='post'>lol this tokenizer is FAST!!! visit /benchmarks?id=42 "
                "&amp; tell me why punctuation...breaks??? #nlp #bpe</div>"
            ),
            (
                "Reddit-style reply: idk if this is cursed formatting or just mobile copy/paste -- "
                "but the text has CAPS, broken apostrophes, URLs like https://x.y/z?a=1&b=2, and emoji [robot][fire]"
            ),
            (
                "<p>cached tokenization might help repeated boilerplate headers, nav labels, and quote "
                "blocks, but messy user-generated text often destroys regularity.</p>"
            ),
            (
                "@someone this benchmark looks legit??? maybe?? newline weirdness, unicode dashes, "
                "ASCII fallbacks, repeated slang, and HTML entities &lt; &gt; should all be measured."
            ),
        ],
        config.docs_per_domain,
    )
    technical = _repeat_to_length(
        [
            (
                "# FastBPE Notes\n"
                "- Throughput matters for RAG preprocessing.\n"
                "- Let T(n) denote total tokenization time.\n"
                "- Compare p95 latency for markdown, equations like O(n log n), and tables.\n"
            ),
            (
                "## Abstract\n"
                "We evaluate exact tokenizer throughput across arXiv-style abstracts, documentation, "
                "lists, and math-heavy notation such as x_i, sum_j, and KL-divergence approximations.\n"
            ),
            (
                "| Metric | Meaning |\n"
                "| --- | --- |\n"
                "| MB/s | bytes processed divided by elapsed wall time |\n"
                "| tokens/s | emitted token ids divided by elapsed wall time |\n"
            ),
            (
                "Documentation excerpt: configure the benchmark runner with --dataset technical "
                "--trials 5 --batch-size 8, then capture environment metadata for reproducibility.\n"
            ),
        ],
        config.docs_per_domain,
    )
    return {
        "english": english,
        "code": code,
        "web": web,
        "technical": technical,
    }
