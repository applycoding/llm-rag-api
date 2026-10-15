"""JSON-backed cosine index.

This is a single-process store: embeddings and chunk text live in one file.
It is the swap point if a later version wants Chroma or FAISS. The demo keeps
the index local so tests and Docker do not need a native vector library.
"""

import json
import logging
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.embeddings import dot, lexical_overlap

logger = logging.getLogger(__name__)

INDEX_VERSION = 1
COSINE_WEIGHT = 0.35
LEXICAL_WEIGHT = 0.65


@dataclass
class DocumentRecord:
    id: str
    source: str
    title: str
    sha256: str
    chunk_ids: list[str]


@dataclass
class ChunkRecord:
    id: str
    document_id: str
    source: str
    title: str
    text: str
    start: int
    end: int
    embedding: list[float]


@dataclass(frozen=True)
class IndexedChunk:
    text: str
    start: int
    end: int
    embedding: list[float]


@dataclass(frozen=True)
class SearchHit:
    chunk: ChunkRecord
    score: float


@dataclass(frozen=True)
class IndexCounts:
    documents: int
    chunks: int


class LocalVectorIndex:
    """Cosine search over hashed embeddings, blended with lexical overlap."""

    def __init__(self, path: Path, dim: int) -> None:
        self.path = path
        self.dim = dim
        self._lock = threading.Lock()
        self.documents: dict[str, DocumentRecord] = {}
        self.chunks: dict[str, ChunkRecord] = {}
        self._load()

    def counts(self) -> IndexCounts:
        with self._lock:
            return IndexCounts(documents=len(self.documents), chunks=len(self.chunks))

    def list_documents(self) -> list[DocumentRecord]:
        with self._lock:
            return list(self.documents.values())

    def upsert(
        self,
        *,
        source: str,
        title: str,
        sha256: str,
        chunks: list[IndexedChunk],
    ) -> tuple[DocumentRecord, bool]:
        with self._lock:
            existing = self._find_source(source)
            if existing is not None and existing.sha256 == sha256:
                return existing, True
            if existing is not None:
                self._drop(existing.id)
            document_id = uuid.uuid4().hex
            chunk_ids: list[str] = []
            for offset, chunk in enumerate(chunks):
                if len(chunk.embedding) != self.dim:
                    raise ValueError("embedding dimension does not match the index")
                chunk_id = f"{document_id}:{offset}"
                self.chunks[chunk_id] = ChunkRecord(
                    id=chunk_id,
                    document_id=document_id,
                    source=source,
                    title=title,
                    text=chunk.text,
                    start=chunk.start,
                    end=chunk.end,
                    embedding=chunk.embedding,
                )
                chunk_ids.append(chunk_id)
            record = DocumentRecord(
                id=document_id,
                source=source,
                title=title,
                sha256=sha256,
                chunk_ids=chunk_ids,
            )
            self.documents[document_id] = record
            self._save()
            return record, False

    def delete(self, document_id: str) -> bool:
        with self._lock:
            if document_id not in self.documents:
                return False
            self._drop(document_id)
            self._save()
            return True

    def clear(self) -> None:
        with self._lock:
            self.documents.clear()
            self.chunks.clear()
            self._save()

    def search(self, vector: list[float], query: str, top_k: int) -> list[SearchHit]:
        if top_k <= 0:
            return []
        with self._lock:
            snapshot = list(self.chunks.values())
        ranked: list[SearchHit] = []
        for chunk in snapshot:
            lexical = lexical_overlap(query, chunk.text)
            if lexical <= 0:
                continue
            cosine = max(dot(vector, chunk.embedding), 0.0)
            score = COSINE_WEIGHT * cosine + LEXICAL_WEIGHT * lexical
            ranked.append(SearchHit(chunk=chunk, score=score))
        ranked.sort(key=lambda hit: hit.score, reverse=True)
        return ranked[:top_k]

    def _find_source(self, source: str) -> DocumentRecord | None:
        for record in self.documents.values():
            if record.source == source:
                return record
        return None

    def _drop(self, document_id: str) -> None:
        record = self.documents.pop(document_id)
        for chunk_id in record.chunk_ids:
            self.chunks.pop(chunk_id, None)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("version") != INDEX_VERSION or payload.get("dim") != self.dim:
                raise ValueError("index metadata does not match this process")
            documents: dict[str, DocumentRecord] = {}
            chunks: dict[str, ChunkRecord] = {}
            for raw in payload["documents"]:
                record = DocumentRecord(
                    id=raw["id"],
                    source=raw["source"],
                    title=raw["title"],
                    sha256=raw["sha256"],
                    chunk_ids=list(raw["chunk_ids"]),
                )
                documents[record.id] = record
            for raw in payload["chunks"]:
                embedding = [float(value) for value in raw["embedding"]]
                if len(embedding) != self.dim:
                    raise ValueError("stored embedding has the wrong dimension")
                chunk = ChunkRecord(
                    id=raw["id"],
                    document_id=raw["document_id"],
                    source=raw["source"],
                    title=raw["title"],
                    text=raw["text"],
                    start=int(raw["start"]),
                    end=int(raw["end"]),
                    embedding=embedding,
                )
                chunks[chunk.id] = chunk
            self.documents = documents
            self.chunks = chunks
        except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring unreadable index at %s: %s", self.path, exc)
            self.documents = {}
            self.chunks = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "dim": self.dim,
            "documents": [
                {
                    "id": record.id,
                    "source": record.source,
                    "title": record.title,
                    "sha256": record.sha256,
                    "chunk_ids": record.chunk_ids,
                }
                for record in self.documents.values()
            ],
            "chunks": [
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "title": chunk.title,
                    "text": chunk.text,
                    "start": chunk.start,
                    "end": chunk.end,
                    "embedding": chunk.embedding,
                }
                for chunk in self.chunks.values()
            ],
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)
