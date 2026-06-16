from __future__ import annotations


def total_megabytes(texts: list[str]) -> float:
    total_bytes = sum(len(text.encode("utf-8")) for text in texts)
    return total_bytes / (1024 * 1024)


def length_bucket(text: str) -> str:
    length = len(text)
    if length <= 64:
        return "16-64"
    if length <= 256:
        return "65-256"
    if length <= 1024:
        return "257-1024"
    if length <= 4096:
        return "1025-4096"
    return "4097+"

