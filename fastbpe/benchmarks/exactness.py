from __future__ import annotations

from dataclasses import dataclass

from fastbpe.adapters.base import TokenizerAdapter


@dataclass
class ExactnessResult:
    compared_docs: int
    exact_matches: int
    mismatches: list[dict[str, object]]

    @property
    def match_rate(self) -> float:
        return self.exact_matches / self.compared_docs if self.compared_docs else 0.0


def compare_to_reference(
    reference: TokenizerAdapter,
    candidate: TokenizerAdapter,
    texts: list[str],
    max_examples: int = 25,
) -> ExactnessResult:
    exact_matches = 0
    mismatches: list[dict[str, object]] = []
    for text in texts:
        ref_ids = reference.encode(text)
        cand_ids = candidate.encode(text)
        if ref_ids == cand_ids:
            exact_matches += 1
        elif len(mismatches) < max_examples:
            mismatches.append(
                {
                    "text_preview": text[:240],
                    "reference_tokens": ref_ids[:80],
                    "candidate_tokens": cand_ids[:80],
                    "reference_count": len(ref_ids),
                    "candidate_count": len(cand_ids),
                }
            )
    return ExactnessResult(compared_docs=len(texts), exact_matches=exact_matches, mismatches=mismatches)

