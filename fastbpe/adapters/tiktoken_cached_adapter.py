from __future__ import annotations

from collections import OrderedDict

import regex

from fastbpe.adapters.base import AdapterMetadata, TokenizerAdapter

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


class CachedTikTokenAdapter(TokenizerAdapter):
    name = "tiktoken_cached"
    metadata = AdapterMetadata(
        compatible_with_reference=True,
        vocabulary_name="cl100k_base",
        supports_batch=True,
    )

    def __init__(self, encoding_name: str = "cl100k_base", cache_size: int = 16384) -> None:
        if tiktoken is None:
            raise RuntimeError("tiktoken is not installed")
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.pattern = regex.compile(self.encoding._pat_str)
        self.cache_size = cache_size
        self.cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.piece_count = 0

    def _encode_piece(self, piece: str) -> list[int]:
        self.piece_count += 1
        if piece in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(piece)
            return list(self.cache[piece])
        self.cache_misses += 1
        encoded = tuple(self.encoding._core_bpe.encode_single_piece(piece.encode("utf-8")))
        self.cache[piece] = encoded
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
            self.cache_evictions += 1
        return list(encoded)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self.pattern.findall(text):
            ids.extend(self._encode_piece(piece))
        return ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def reset_stats(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.piece_count = 0

    def stats(self) -> dict[str, int | float]:
        total = self.cache_hits + self.cache_misses
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_evictions": self.cache_evictions,
            "cache_size": len(self.cache),
            "cache_capacity": self.cache_size,
            "cache_hit_rate": (self.cache_hits / total) if total else 0.0,
            "piece_count": self.piece_count,
        }
