import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from uuid import uuid4

from app.config import Settings
from app.domain import Answer, ChunkRecord, Citation, DocumentSummary, QueryEvent, RetrievedChunk
from app.embeddings.local import LocalEmbeddingProvider
from app.embeddings.openai import OpenAIEmbeddingProvider
from app.errors import IngestError, ProviderError
from app.ingestion.chunker import chunk_segments
from app.ingestion.loaders import load_document, media_type_for
from app.llm.types import LLMProvider
from app.prompting import build_messages
from app.store.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)

NO_SOURCES_ANSWER = "No documents have been ingested yet, so there is nothing to cite."


class RagService:
    def __init__(
        self,
        *,
        settings: Settings,
        embedder: LocalEmbeddingProvider | OpenAIEmbeddingProvider,
        store: ChromaVectorStore,
        llm: LLMProvider,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._llm = llm

    async def aclose(self) -> None:
        await self._llm.aclose()
        await self._embedder.aclose()

    async def ping(self) -> None:
        await asyncio.to_thread(self._store.ping)

    async def ingest(self, *, filename: str, data: bytes) -> DocumentSummary:
        if not data:
            raise IngestError("The file is empty.", status_code=400)
        media_type = media_type_for(filename)
        drafts = chunk_segments(
            load_document(filename, data),
            chunk_size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )
        if not drafts:
            raise IngestError("The file has no extractable text.", status_code=400)
        if len(drafts) > self._settings.max_chunks:
            limit = self._settings.max_chunks
            raise IngestError(
                f"Document produced {len(drafts)} chunks, above the demo limit of {limit}.",
                status_code=413,
            )

        document_id = uuid4().hex
        uploaded_at = datetime.now(UTC)
        records = [
            ChunkRecord(
                chunk_id=f"{document_id}:{draft.chunk_index}",
                document_id=document_id,
                filename=filename,
                media_type=media_type,
                chunk_index=draft.chunk_index,
                page=draft.page,
                text=draft.text,
                uploaded_at=uploaded_at,
            )
            for draft in drafts
        ]
        vectors = await self._embedder.embed([record.text for record in records])
        if len(vectors) != len(records):
            raise ProviderError("The embedding provider returned an unexpected number of vectors.")
        await asyncio.to_thread(self._store.upsert, records, vectors)
        logger.info("ingested document %s chunks=%s", document_id, len(records))
        return DocumentSummary(
            document_id=document_id,
            filename=filename,
            media_type=media_type,
            chunk_count=len(records),
            uploaded_at=uploaded_at,
        )

    async def list_documents(self) -> list[DocumentSummary]:
        return await asyncio.to_thread(self._store.list_documents)

    async def delete(self, document_id: str) -> bool:
        return await asyncio.to_thread(self._store.delete_document, document_id)

    async def retrieve(self, question: str, top_k: int | None) -> list[RetrievedChunk]:
        limit = self._settings.default_top_k if top_k is None else top_k
        vectors = await self._embedder.embed([question])
        if len(vectors) != 1:
            raise ProviderError("The embedding provider returned an unexpected number of vectors.")
        return await asyncio.to_thread(self._store.query, vectors[0], limit)

    async def answer(self, question: str, top_k: int | None = None) -> Answer:
        return await self.complete_answer(question, await self.retrieve(question, top_k))

    async def complete_answer(self, question: str, retrieved: Sequence[RetrievedChunk]) -> Answer:
        citations = _citations(retrieved)
        if not retrieved:
            text = NO_SOURCES_ANSWER
        else:
            text = await self._llm.complete(build_messages(question, retrieved))
        return Answer(
            text=text,
            citations=citations,
            provider=self._llm.name,
            model=self._llm.model,
        )

    async def stream_answer(
        self,
        question: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> AsyncIterator[QueryEvent]:
        citations = _citations(retrieved)
        yield QueryEvent(
            kind="citations",
            data={"citations": [citation.to_dict() for citation in citations]},
        )
        if not retrieved:
            yield QueryEvent(kind="token", data={"text": NO_SOURCES_ANSWER})
            yield _done(NO_SOURCES_ANSWER, self._llm.name, self._llm.model)
            return

        parts: list[str] = []
        async for token in self._llm.stream(build_messages(question, retrieved)):
            parts.append(token)
            yield QueryEvent(kind="token", data={"text": token})
        yield _done("".join(parts), self._llm.name, self._llm.model)


def _citations(retrieved: Sequence[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            index=index,
            document_id=item.record.document_id,
            filename=item.record.filename,
            chunk_index=item.record.chunk_index,
            page=item.record.page,
            score=item.score,
            text=item.record.text,
        )
        for index, item in enumerate(retrieved, start=1)
    ]


def _done(answer: str, provider: str, model: str) -> QueryEvent:
    return QueryEvent(
        kind="done",
        data={"answer": answer, "provider": provider, "model": model},
    )
