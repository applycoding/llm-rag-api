"""Offline hashing embedder and the lexical score that stabilizes it.

Hashed bag-of-words vectors are not a semantic model. Retrieval therefore
mixes the cosine of those vectors with token overlap, which is the signal
this demo can actually trust without downloading a model.
"""

import hashlib
import math
import re
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "each",
        "for",
        "from",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "of",
        "on",
        "or",
        "per",
        "should",
        "than",
        "that",
        "the",
        "then",
        "these",
        "this",
        "those",
        "to",
        "via",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "would",
    }
)


def tokenize(text: str, *, drop_stopwords: bool = False) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    if drop_stopwords:
        return [token for token in tokens if token not in STOPWORDS]
    return tokens


def lexical_overlap(query: str, text: str) -> float:
    """Share of query content-words that also appear in text. Range 0 to 1."""
    query_terms = set(tokenize(query, drop_stopwords=True))
    if not query_terms:
        return 0.0
    text_terms = set(tokenize(text, drop_stopwords=True))
    return len(query_terms & text_terms) / len(query_terms)


class HashingEmbedder:
    """Signed feature hashing over tokens and adjacent token pairs."""

    def __init__(self, dim: int = 384) -> None:
        if dim < 8:
            raise ValueError("embedding dim must be at least 8")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            _accumulate(vector, token, 1.0)
        for left, right in zip(tokens, tokens[1:], strict=False):
            _accumulate(vector, f"{left}_{right}", 0.5)
        return _l2_normalize(vector)


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _accumulate(vector: list[float], feature: str, weight: float) -> None:
    digest = hashlib.blake2s(feature.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest[:4], "little") % len(vector)
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    vector[bucket] += sign * weight


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
