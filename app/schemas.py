"""HTTP request and response models."""

from pydantic import BaseModel, Field, field_validator

from app.models import IngestOutcome


class IngestResponse(BaseModel):
    document_id: str
    source: str
    title: str
    chunk_count: int
    unchanged: bool

    @classmethod
    def from_outcome(cls, outcome: IngestOutcome) -> "IngestResponse":
        return cls(
            document_id=outcome.document_id,
            source=outcome.source,
            title=outcome.title,
            chunk_count=outcome.chunk_count,
            unchanged=outcome.unchanged,
        )


class TextIngestRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    source: str = "pasted.txt"
    title: str | None = Field(default=None, max_length=120)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be blank")
        return stripped

    @field_validator("source")
    @classmethod
    def source_is_a_filename(cls, value: str) -> str:
        name = value.strip().replace("\\", "/").split("/")[-1]
        if not name or name in {".", ".."}:
            return "pasted.txt"
        return name[:180]


class SampleIngestResponse(BaseModel):
    documents: list[IngestResponse]


class DocumentResponse(BaseModel):
    id: str
    source: str
    title: str
    chunk_count: int


class Citation(BaseModel):
    index: int
    document_id: str
    chunk_id: str
    source: str
    title: str
    score: float
    excerpt: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    provider: str
    model: str


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    credentials_configured: bool
    documents: int
    chunks: int


class ProviderResponse(BaseModel):
    name: str
    default_model: str
    requires_api_key: bool
    active: bool
    credentials_configured: bool


class ResetResponse(BaseModel):
    documents: int
    chunks: int
