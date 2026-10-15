from app.embeddings.local import embed_text


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def test_local_embeddings_are_deterministic_unit_vectors() -> None:
    first = embed_text("Chroma stores embeddings", 64)
    second = embed_text("Chroma stores embeddings", 64)
    assert first == second
    assert len(first) == 64
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


def test_empty_text_still_has_a_unit_vector() -> None:
    vector = embed_text("???", 32)
    assert vector[0] == 1.0
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-6


def test_related_text_outranks_unrelated_text() -> None:
    dimensions = 384
    query = embed_text("Where are embeddings stored?", dimensions)
    stored = embed_text("Embeddings are stored in Chroma on disk.", dimensions)
    streamed = embed_text("The query API streams tokens and returns citations.", dimensions)
    assert _cosine(query, stored) > _cosine(query, streamed)
