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

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        cache_size: int = 16384,
        doc_cache_size: int = 2048,
    ) -> None:
        if tiktoken is None:
            raise RuntimeError("tiktoken is not installed")
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.pattern = regex.compile(self.encoding._pat_str)
        self.cache_size = cache_size
        self.doc_cache_size = doc_cache_size
        self.cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self.doc_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.doc_cache_hits = 0
        self.doc_cache_misses = 0
        self.doc_cache_evictions = 0
        self.piece_count = 0
        self.native_encode_calls = 0

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
        if text in self.doc_cache:
            self.doc_cache_hits += 1
            self.doc_cache.move_to_end(text)
            return list(self.doc_cache[text])
        self.doc_cache_misses += 1
        # Native encode on cache miss is materially faster than Python-side
        # piece orchestration while still preserving exact token IDs.
        self.native_encode_calls += 1
        encoded = tuple(self.encoding.encode(text))
        self.doc_cache[text] = encoded
        if len(self.doc_cache) > self.doc_cache_size:
            self.doc_cache.popitem(last=False)
            self.doc_cache_evictions += 1
        return list(encoded)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def reset_stats(self) -> None:
        self.cache.clear()
        self.doc_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.doc_cache_hits = 0
        self.doc_cache_misses = 0
        self.doc_cache_evictions = 0
        self.piece_count = 0
        self.native_encode_calls = 0

    def stats(self) -> dict[str, int | float]:
        total = self.cache_hits + self.cache_misses
        doc_total = self.doc_cache_hits + self.doc_cache_misses
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_evictions": self.cache_evictions,
            "cache_size": len(self.cache),
            "cache_capacity": self.cache_size,
            "cache_hit_rate": (self.cache_hits / total) if total else 0.0,
            "doc_cache_hits": self.doc_cache_hits,
            "doc_cache_misses": self.doc_cache_misses,
            "doc_cache_evictions": self.doc_cache_evictions,
            "doc_cache_size": len(self.doc_cache),
            "doc_cache_capacity": self.doc_cache_size,
            "doc_cache_hit_rate": (self.doc_cache_hits / doc_total) if doc_total else 0.0,
            "piece_count": self.piece_count,
            "native_encode_calls": self.native_encode_calls,
        }
