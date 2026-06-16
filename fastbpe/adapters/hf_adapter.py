from __future__ import annotations

from fastbpe.adapters.base import AdapterMetadata, TokenizerAdapter

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
except ImportError:  # pragma: no cover
    Tokenizer = None
    BPE = None
    ByteLevel = None
    BpeTrainer = None


class HuggingFaceTokenizerAdapter(TokenizerAdapter):
    name = "hf"
    metadata = AdapterMetadata(
        compatible_with_reference=False,
        vocabulary_name="bytelevel-bpe-demo",
        supports_batch=True,
    )

    def __init__(self, training_corpus: list[str] | None = None, vocab_size: int = 8192) -> None:
        if Tokenizer is None:
            raise RuntimeError("tokenizers is not installed")
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]"],
            initial_alphabet=ByteLevel.alphabet(),
        )
        corpus = training_corpus or [
            "Tokenizer benchmarking needs consistent byte-level segmentation.",
            "Code paths, markdown, and web text all produce different token shapes.",
        ]
        self.tokenizer.train_from_iterator(corpus, trainer)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [encoding.ids for encoding in self.tokenizer.encode_batch(texts)]

