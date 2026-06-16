from __future__ import annotations

import re
from collections import Counter

from fastbpe.adapters.base import AdapterMetadata, TokenizerAdapter


class NaivePythonBPEAdapter(TokenizerAdapter):
    name = "naive"
    metadata = AdapterMetadata(
        compatible_with_reference=False,
        vocabulary_name="fastbpe-naive-char",
        supports_batch=True,
    )

    def __init__(self) -> None:
        self.pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
        self.vocab: dict[str, int] = {}
        self.lookup_ops = 0
        self.pretokenize_calls = 0

    def _token_id(self, piece: str) -> int:
        self.lookup_ops += 1
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab)
        return self.vocab[piece]

    def _pretokenize(self, text: str) -> list[str]:
        self.pretokenize_calls += 1
        return self.pattern.findall(text)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in self._pretokenize(text):
            chars = list(token)
            for char in chars:
                ids.append(self._token_id(char))
        return ids

    def reset_stats(self) -> None:
        self.lookup_ops = 0
        self.pretokenize_calls = 0

    def stats(self) -> dict[str, int]:
        return {
            "lookup_ops": self.lookup_ops,
            "pretokenize_calls": self.pretokenize_calls,
            "vocab_size": len(self.vocab),
        }


class PretokenizationComparator:
    def __init__(self) -> None:
        self.regex_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def regex_split(self, text: str) -> list[str]:
        return self.regex_pattern.findall(text)

    def simple_split(self, text: str) -> list[str]:
        parts = re.split(r"(\s+)", text)
        return [part for part in parts if part and not part.isspace()]

    def compare(self, texts: list[str]) -> dict[str, float]:
        regex_lengths = Counter(len(self.regex_split(text)) for text in texts)
        simple_lengths = Counter(len(self.simple_split(text)) for text in texts)
        return {
            "regex_mean_parts": sum(k * v for k, v in regex_lengths.items()) / max(1, sum(regex_lengths.values())),
            "simple_mean_parts": sum(k * v for k, v in simple_lengths.items()) / max(1, sum(simple_lengths.values())),
        }

