import pytest

from app.chunking import chunk_text


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("  \n\t ", 200, 20) == []


def test_short_text_is_one_chunk() -> None:
    chunks = chunk_text("Harbor rotation.", 200, 20)
    assert len(chunks) == 1
    assert chunks[0].text == "Harbor rotation."
    assert chunks[0].start == 0


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", 20, 20)


def test_long_text_overlaps_and_prefers_paragraphs() -> None:
    text = ("A" * 60) + "\n\n" + ("B" * 60)
    chunks = chunk_text(text, size=80, overlap=10)
    assert chunks[0].text == "A" * 60
    assert "B" in chunks[1].text
    assert chunks[1].start < chunks[0].end


def test_windows_advance() -> None:
    text = "word " * 200
    chunks = chunk_text(text.strip(), size=80, overlap=20)
    assert len(chunks) > 2
    assert all(chunk.text for chunk in chunks)
    assert all(chunks[index].start < chunks[index + 1].start for index in range(len(chunks) - 1))
