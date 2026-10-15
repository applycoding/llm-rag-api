import logging
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings
from app.domain import ChunkRecord, DocumentSummary, RetrievedChunk
from app.errors import AppError, StoreError

logger = logging.getLogger(__name__)

_BATCH_SIZE = 256
_PAGE_MISSING = 0


class ChromaVectorStore:
    """Chroma collection that stores caller-supplied embeddings."""

    def __init__(self, settings: Settings) -> None:
        self._lock = threading.Lock()
        self._client = build_chroma_client(settings)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def ping(self) -> None:
        with self._lock, _guard("answer a health check"):
            self._collection.count()

    def upsert(self, records: Sequence[ChunkRecord], embeddings: Sequence[Sequence[float]]) -> None:
        if len(records) != len(embeddings):
            raise StoreError("Refusing to store chunks without one embedding each.")
        with self._lock, _guard("store embeddings"):
            for start in range(0, len(records), _BATCH_SIZE):
                batch = list(records[start : start + _BATCH_SIZE])
                vectors = [list(vector) for vector in embeddings[start : start + _BATCH_SIZE]]
                self._collection.upsert(
                    ids=[record.chunk_id for record in batch],
                    documents=[record.text for record in batch],
                    embeddings=vectors,
                    metadatas=[_metadata(record) for record in batch],
                )

    def query(self, embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        with self._lock, _guard("search embeddings"):
            count = self._collection.count()
            if count == 0 or top_k <= 0:
                return []
            result = self._collection.query(
                query_embeddings=[list(embedding)],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
        return _retrieved(result)

    def delete_document(self, document_id: str) -> bool:
        with self._lock, _guard("delete a document"):
            found = self._collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            ids = found.get("ids") or []
            if not ids:
                return False
            self._collection.delete(ids=list(ids))
            return True

    def list_documents(self) -> list[DocumentSummary]:
        with self._lock, _guard("list documents"):
            payload = self._collection.get(include=["metadatas"])
        return _summaries(payload.get("metadatas") or [])


def build_chroma_client(settings: Settings) -> chromadb.ClientAPI:
    chroma_settings = ChromaSettings(
        anonymized_telemetry=False,
        chroma_product_telemetry_impl="app.store.telemetry.NoopTelemetry",
    )
    if settings.chroma_host:
        return chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=chroma_settings,
        )
    path = Path(settings.chroma_persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=chroma_settings)


@contextmanager
def _guard(action: str) -> Iterator[None]:
    try:
        yield
    except AppError:
        raise
    except Exception as exc:
        logger.exception("chroma failed to %s", action)
        raise StoreError(f"The vector store could not {action}.") from exc


def _metadata(record: ChunkRecord) -> dict[str, str | int]:
    return {
        "document_id": record.document_id,
        "filename": record.filename,
        "media_type": record.media_type,
        "chunk_index": record.chunk_index,
        "page": record.page if record.page is not None else _PAGE_MISSING,
        "uploaded_at": record.uploaded_at.isoformat(),
    }


def _retrieved(result: object) -> list[RetrievedChunk]:
    if not isinstance(result, dict):
        raise StoreError("The vector store returned an unexpected search result.")
    ids = result.get("ids") or []
    if not ids or not ids[0]:
        return []
    documents = (result.get("documents") or [[]])[0] or []
    metadatas = (result.get("metadatas") or [[]])[0] or []
    distances = (result.get("distances") or [[]])[0] or []
    row_ids = ids[0]
    if not (len(row_ids) == len(documents) == len(metadatas) == len(distances)):
        raise StoreError("The vector store returned an incomplete search result.")

    found: list[RetrievedChunk] = []
    for chunk_id, document, metadata, distance in zip(
        row_ids,
        documents,
        metadatas,
        distances,
        strict=True,
    ):
        if (
            not isinstance(chunk_id, str)
            or not isinstance(document, str)
            or not isinstance(metadata, dict)
        ):
            continue
        record = _record(chunk_id, document, metadata)
        if record is None:
            continue
        found.append(RetrievedChunk(record=record, score=_similarity(distance)))
    return found


def _summaries(metadatas: Sequence[object]) -> list[DocumentSummary]:
    grouped: dict[str, dict[str, object]] = {}
    for metadata in metadatas:
        if not isinstance(metadata, dict):
            continue
        document_id = metadata.get("document_id")
        filename = metadata.get("filename")
        media_type = metadata.get("media_type")
        uploaded_at = _uploaded_at(metadata.get("uploaded_at"))
        if not isinstance(document_id, str) or not isinstance(filename, str):
            continue
        if not isinstance(media_type, str) or uploaded_at is None:
            continue
        current = grouped.get(document_id)
        if current is None:
            grouped[document_id] = {
                "filename": filename,
                "media_type": media_type,
                "uploaded_at": uploaded_at,
                "chunk_count": 1,
            }
            continue
        count = current["chunk_count"]
        current["chunk_count"] = count + 1 if isinstance(count, int) else 1

    summaries: list[DocumentSummary] = []
    for document_id, fields in grouped.items():
        uploaded_at = fields["uploaded_at"]
        chunk_count = fields["chunk_count"]
        if not isinstance(uploaded_at, datetime) or not isinstance(chunk_count, int):
            continue
        summaries.append(
            DocumentSummary(
                document_id=document_id,
                filename=str(fields["filename"]),
                media_type=str(fields["media_type"]),
                chunk_count=chunk_count,
                uploaded_at=uploaded_at,
            )
        )
    summaries.sort(key=lambda item: (item.uploaded_at, item.filename, item.document_id))
    return summaries


def _record(chunk_id: str, document: str, metadata: dict[str, object]) -> ChunkRecord | None:
    document_id = metadata.get("document_id")
    filename = metadata.get("filename")
    media_type = metadata.get("media_type")
    chunk_index = _as_int(metadata.get("chunk_index"))
    uploaded_at = _uploaded_at(metadata.get("uploaded_at"))
    if not isinstance(document_id, str) or not isinstance(filename, str):
        return None
    if not isinstance(media_type, str) or chunk_index is None or uploaded_at is None:
        return None
    page = _as_int(metadata.get("page"))
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=filename,
        media_type=media_type,
        chunk_index=chunk_index,
        page=None if page in (None, _PAGE_MISSING) else page,
        text=document,
        uploaded_at=uploaded_at,
    )


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _uploaded_at(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _similarity(distance: object) -> float:
    # Chroma's cosine space reports distance as 1 - cosine similarity.
    try:
        value = float(distance)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, 1.0 - value))
