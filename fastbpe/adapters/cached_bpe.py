from __future__ import annotations

from collections import OrderedDict

from fastbpe.adapters.base import AdapterMetadata
from fastbpe.adapters.naive_bpe import NaivePythonBPEAdapter


class CachedPythonBPEAdapter(NaivePythonBPEAdapter):
    name = "cached"
    metadata = AdapterMetadata(
        compatible_with_reference=False,
        vocabulary_name="fastbpe-cached-char",
        supports_batch=True,
    )

    def __init__(self, cache_size: int = 4096) -> None:
        super().__init__()
        self.cache_size = cache_size
        self.cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

    def _encode_token(self, token: str) -> list[int]:
        if token in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(token)
            return list(self.cache[token])
        self.cache_misses += 1
        encoded = tuple(self._token_id(char) for char in token)
        self.cache[token] = encoded
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
            self.cache_evictions += 1
        return list(encoded)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in self._pretokenize(text):
            ids.extend(self._encode_token(token))
        return ids

    def reset_stats(self) -> None:
        super().reset_stats()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

    def stats(self) -> dict[str, int | float]:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total else 0.0
        return {
            **super().stats(),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_evictions": self.cache_evictions,
            "cache_size": len(self.cache),
            "cache_capacity": self.cache_size,
            "cache_hit_rate": hit_rate,
        }

