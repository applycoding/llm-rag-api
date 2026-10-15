from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    filename: str
    media_type: str
    chunk_index: int
    page: int | None
    text: str
    uploaded_at: datetime


@dataclass(frozen=True)
class RetrievedChunk:
    record: ChunkRecord
    score: float


@dataclass(frozen=True)
class DocumentSummary:
    document_id: str
    filename: str
    media_type: str
    chunk_count: int
    uploaded_at: datetime


@dataclass(frozen=True)
class Citation:
    index: int
    document_id: str
    filename: str
    chunk_index: int
    page: int | None
    score: float
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "page": self.page,
            "score": round(self.score, 4),
            "text": self.text,
        }


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    provider: str
    model: str


@dataclass(frozen=True)
class QueryEvent:
    kind: str
    data: dict[str, object]
