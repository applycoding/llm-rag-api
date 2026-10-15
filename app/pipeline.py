"""Ingest documents and answer questions from the local index."""

import hashlib
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from app.chunking import chunk_text
from app.config import Settings
from app.documents import (
    ALLOWED_SUFFIXES,
    MAX_CHUNKS,
    IngestError,
    LoadedDocument,
    infer_title,
    load_bytes,
)
from app.embeddings import HashingEmbedder
from app.llm.base import LLMProvider
from app.llm.factory import build_provider
from app.models import IngestOutcome, RetrievedChunk
from app.prompts import build_messages
from app.schemas import AnswerResponse, Citation
from app.store import IndexedChunk, LocalVectorIndex

logger = logging.getLogger(__name__)

_EXCERPT_CHARS = 320


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        provider: LLMProvider | None = None,
        embedder: HashingEmbedder | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or HashingEmbedder(settings.embedding_dim)
        self.store = LocalVectorIndex(settings.data_dir / "index.json", self.embedder.dim)
        self.provider = provider or build_provider(settings)

    def ingest_loaded(self, loaded: LoadedDocument) -> IngestOutcome:
        return self.ingest_text(loaded.text, source=loaded.source, title=loaded.title)

    def ingest_text(self, text: str, *, source: str, title: str | None = None) -> IngestOutcome:
        cleaned = text.replace("\x00", "").strip()
        if not cleaned:
            raise IngestError("Document is empty.")
        filename = Path(source).name or "pasted.txt"
        resolved_title = (title or infer_title(cleaned, filename)).strip() or filename
        spans = chunk_text(cleaned, self.settings.chunk_size, self.settings.chunk_overlap)
        if not spans:
            raise IngestError("Document is empty.")
        if len(spans) > MAX_CHUNKS:
            raise IngestError(
                f"Document produced {len(spans)} chunks, above the demo limit of {MAX_CHUNKS}."
            )
        indexed = [
            IndexedChunk(
                text=span.text,
                start=span.start,
                end=span.end,
                embedding=self.embedder.embed(span.text),
            )
            for span in spans
        ]
        record, unchanged = self.store.upsert(
            source=filename,
            title=resolved_title,
            sha256=self._fingerprint(resolved_title, cleaned),
            chunks=indexed,
        )
        logger.info(
            "Indexed %s (%s chunks, unchanged=%s)",
            record.source,
            len(record.chunk_ids),
            unchanged,
        )
        return IngestOutcome(
            document_id=record.id,
            source=record.source,
            title=record.title,
            chunk_count=len(record.chunk_ids),
            unchanged=unchanged,
        )

    def ingest_directory(self, directory: Path) -> list[IngestOutcome]:
        if not directory.is_dir():
            raise IngestError(f"Sample directory not found: {directory}")
        paths = [
            path
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
        ]
        if not paths:
            raise IngestError(f"No PDF, Markdown, or text files found in {directory}.")
        return [self.ingest_loaded(load_bytes(path.name, path.read_bytes())) for path in paths]

    def list_documents(self) -> list[tuple[str, str, str, int]]:
        rows = []
        for record in self.store.list_documents():
            rows.append((record.id, record.source, record.title, len(record.chunk_ids)))
        return rows

    def delete(self, document_id: str) -> bool:
        return self.store.delete(document_id)

    def reset(self) -> None:
        self.store.clear()

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        limit = self.settings.top_k if top_k is None else top_k
        hits = [
            hit
            for hit in self.store.search(self.embedder.embed(question), question, limit)
            if hit.score >= self.settings.min_score
        ]
        return [
            RetrievedChunk(
                index=index,
                document_id=hit.chunk.document_id,
                chunk_id=hit.chunk.id,
                source=hit.chunk.source,
                title=hit.chunk.title,
                score=round(hit.score, 4),
                text=hit.chunk.text,
            )
            for index, hit in enumerate(hits, start=1)
        ]

    async def answer(self, question: str, top_k: int | None = None) -> AnswerResponse:
        retrieved, messages = self._prepare(question, top_k)
        text = (await self.provider.complete(messages)).strip()
        return self._package(text, retrieved)

    async def stream_events(
        self,
        question: str,
        top_k: int | None = None,
    ) -> AsyncIterator[dict[str, object]]:
        retrieved, messages = self._prepare(question, top_k)
        citations = [item.model_dump() for item in self._citations(retrieved)]
        yield {"event": "citations", "data": {"citations": citations}}
        parts: list[str] = []
        async for token in self.provider.stream(messages):
            parts.append(token)
            yield {"event": "token", "data": {"text": token}}
        packaged = self._package("".join(parts).strip(), retrieved)
        yield {"event": "done", "data": packaged.model_dump()}

    def _prepare(self, question: str, top_k: int | None):
        retrieved = self.retrieve(question, top_k)
        return retrieved, build_messages(question, retrieved)

    def _package(self, text: str, retrieved: list[RetrievedChunk]) -> AnswerResponse:
        return AnswerResponse(
            answer=text,
            citations=self._citations(retrieved),
            provider=self.provider.name,
            model=self.provider.model,
        )

    def _citations(self, retrieved: list[RetrievedChunk]) -> list[Citation]:
        return [
            Citation(
                index=item.index,
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                source=item.source,
                title=item.title,
                score=item.score,
                excerpt=_excerpt(item.text),
            )
            for item in retrieved
        ]

    def _fingerprint(self, title: str, text: str) -> str:
        raw = (
            f"{self.embedder.dim}:{self.settings.chunk_size}:"
            f"{self.settings.chunk_overlap}:{title}:{text}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."
