import asyncio
import hashlib
import math
from collections.abc import Sequence

from app.textutil import tokenize


class LocalEmbeddingProvider:
    """Lexical hashing embedder. Similar wording scores higher; meaning does not."""

    name = "local"

    def __init__(self, *, dimensions: int) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_many, list(texts))

    async def aclose(self) -> None:
        return None

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        return [embed_text(text, self.dimensions) for text in texts]


def embed_text(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    tokens = tokenize(text)
    if not tokens:
        vector[0] = 1.0
        return vector
    for token in tokens:
        encoded = token.encode("utf-8")
        for salt in (b"a", b"b"):
            digest = hashlib.sha256(salt + encoded).digest()
            bucket = int.from_bytes(digest[:4], "little") % dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]
