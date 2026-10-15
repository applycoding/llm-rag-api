import math

from app.embeddings import HashingEmbedder, lexical_overlap


def test_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dim=64)
    first = embedder.embed("audit logs for 180 days")
    assert first == embedder.embed("audit logs for 180 days")
    norm = math.sqrt(sum(value * value for value in first))
    assert abs(norm - 1) < 1e-9


def test_empty_text_is_a_zero_vector() -> None:
    assert HashingEmbedder(dim=16).embed("   ") == [0.0] * 16


def test_lexical_overlap_ignores_question_words() -> None:
    score = lexical_overlap(
        "How long does Lumen Archive retain audit logs?",
        "Lumen Archive retains audit logs for 180 days.",
    )
    assert score > 0.5
    assert lexical_overlap("What is the capital of France?", "Supported regions are Iceland.") == 0
