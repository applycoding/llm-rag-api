import pytest

from app.ingestion.chunker import TextSegment, chunk_segments, chunk_text


def test_empty_text_has_no_chunks() -> None:
    assert chunk_text("  \n\t ", chunk_size=100, overlap=10) == []


def test_overlap_is_exact_when_there_is_no_whitespace() -> None:
    chunks = chunk_text("a" * 250, chunk_size=100, overlap=20)
    assert len(chunks) >= 3
    assert chunks[0][-20:] == chunks[1][:20]
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunks_stay_within_the_requested_size() -> None:
    chunks = chunk_text("alpha beta " * 80, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_overlap_must_be_smaller_than_the_chunk() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("hello", chunk_size=20, overlap=20)


def test_pages_are_preserved_across_segments() -> None:
    drafts = chunk_segments(
        [
            TextSegment(text="first page", page=1),
            TextSegment(text="second page", page=2),
        ],
        chunk_size=100,
        overlap=10,
    )
    assert [draft.page for draft in drafts] == [1, 2]
    assert [draft.chunk_index for draft in drafts] == [0, 1]
