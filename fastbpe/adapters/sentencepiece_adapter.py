from __future__ import annotations

import tempfile
from pathlib import Path

from fastbpe.adapters.base import AdapterMetadata, TokenizerAdapter

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover
    spm = None


class SentencePieceAdapter(TokenizerAdapter):
    name = "sentencepiece"
    metadata = AdapterMetadata(
        compatible_with_reference=False,
        vocabulary_name="sentencepiece-unigram-demo",
        supports_batch=True,
    )

    def __init__(self, training_corpus: list[str] | None = None, vocab_size: int = 512) -> None:
        if spm is None:
            raise RuntimeError("sentencepiece is not installed")
        corpus = training_corpus or [
            "SentencePiece is included for speed comparison, not exact ID compatibility.",
            "The project benchmarks throughput across heterogeneous text domains.",
        ]
        temp_dir = Path(tempfile.mkdtemp(prefix="fastbpe_spm_"))
        input_path = temp_dir / "corpus.txt"
        model_prefix = temp_dir / "spm"
        input_path.write_text("\n".join(corpus), encoding="utf-8")
        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            hard_vocab_limit=False,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
        )
        self.processor = spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")

    def encode(self, text: str) -> list[int]:
        return list(self.processor.encode(text, out_type=int))

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [list(ids) for ids in self.processor.encode(texts, out_type=int)]
