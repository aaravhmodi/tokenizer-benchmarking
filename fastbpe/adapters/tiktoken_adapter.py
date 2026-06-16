from __future__ import annotations

from fastbpe.adapters.base import AdapterMetadata, TokenizerAdapter

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


class TikTokenAdapter(TokenizerAdapter):
    name = "tiktoken"
    metadata = AdapterMetadata(
        compatible_with_reference=True,
        vocabulary_name="cl100k_base",
        supports_batch=False,
    )

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        if tiktoken is None:
            raise RuntimeError("tiktoken is not installed")
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)

