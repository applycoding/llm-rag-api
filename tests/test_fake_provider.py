from datetime import UTC, datetime

import pytest

from app.domain import ChunkRecord, RetrievedChunk
from app.llm.fake import FakeProvider
from app.prompting import build_messages


def _chunk(text: str, filename: str) -> RetrievedChunk:
    record = ChunkRecord(
        chunk_id="doc:0",
        document_id="a" * 32,
        filename=filename,
        media_type="text/plain",
        chunk_index=0,
        page=None,
        text=text,
        uploaded_at=datetime.now(UTC),
    )
    return RetrievedChunk(record=record, score=0.5)


@pytest.mark.asyncio
async def test_fake_provider_cites_the_matching_source() -> None:
    retrieved = [
        _chunk("The query API streams tokens and returns citations.", "api.md"),
        _chunk("Embeddings are stored in Chroma on disk.", "storage.md"),
    ]
    messages = build_messages("Where are embeddings stored?", retrieved)
    provider = FakeProvider(model="fake-extractive")

    answer = await provider.complete(messages)
    pieces: list[str] = []
    async for piece in provider.stream(messages):
        pieces.append(piece)

    assert "Chroma" in answer
    assert "[2]" in answer
    assert "".join(pieces) == answer
    assert len(pieces) > 1


@pytest.mark.asyncio
async def test_fake_provider_does_not_quote_a_markdown_heading() -> None:
    note = (
        "# Retrieval notes\n\n"
        "This demo stores embeddings in Chroma. "
        "Embeddings are stored on disk for similarity search."
    )
    messages = build_messages("Where are embeddings stored?", [_chunk(note, "notes.md")])
    answer = await FakeProvider(model="fake-extractive").complete(messages)
    assert answer.startswith("Embeddings are stored on disk for similarity search.")
    assert "# Retrieval notes" not in answer


@pytest.mark.asyncio
async def test_fake_provider_admits_when_sources_do_not_answer() -> None:
    messages = build_messages(
        "How do you feed a sourdough starter?",
        [_chunk("Embeddings are stored in Chroma on disk.", "storage.md")],
    )
    answer = await FakeProvider(model="fake-extractive").complete(messages)
    assert "do not contain an answer" in answer
