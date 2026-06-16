from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdapterMetadata:
    compatible_with_reference: bool = False
    vocabulary_name: str | None = None
    supports_batch: bool = True
    notes: dict[str, Any] = field(default_factory=dict)


class TokenizerAdapter:
    name: str = "base"
    metadata: AdapterMetadata = AdapterMetadata()

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def warmup(self, texts: list[str]) -> None:
        for text in texts[:3]:
            self.encode(text)

    def reset_stats(self) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {}

