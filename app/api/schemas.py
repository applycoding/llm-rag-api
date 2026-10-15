from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, field_validator

from app.domain import Answer, DocumentSummary


class HealthOut(BaseModel):
    status: str
    demo: bool
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    vector_store: str
    vector_store_mode: str


class DocumentOut(BaseModel):
    id: str
    filename: str
    media_type: str
    chunk_count: int
    uploaded_at: datetime

    @classmethod
    def from_summary(cls, document: DocumentSummary) -> Self:
        return cls(
            id=document.document_id,
            filename=document.filename,
            media_type=document.media_type,
            chunk_count=document.chunk_count,
            uploaded_at=document.uploaded_at,
        )


class DocumentListOut(BaseModel):
    documents: list[DocumentOut]


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty")
        return stripped


class CitationOut(BaseModel):
    index: int
    document_id: str
    filename: str
    chunk_index: int
    page: int | None
    score: float
    text: str


class QueryOut(BaseModel):
    answer: str
    citations: list[CitationOut]
    provider: str
    model: str

    @classmethod
    def from_answer(cls, answer: Answer) -> Self:
        return cls(
            answer=answer.text,
            citations=[
                CitationOut.model_validate(citation.to_dict()) for citation in answer.citations
            ],
            provider=answer.provider,
            model=answer.model,
        )
