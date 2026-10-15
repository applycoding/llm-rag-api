"""HTTP API for the demo."""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from app import __version__
from app.config import (
    API_KEY_ENV,
    DEFAULT_MODELS,
    ProviderName,
    Settings,
    missing_key_message,
)
from app.documents import IngestError, load_bytes
from app.llm.base import MissingCredentialsError, ProviderError
from app.pipeline import Pipeline
from app.schemas import (
    AnswerResponse,
    DocumentResponse,
    HealthResponse,
    IngestResponse,
    ProviderResponse,
    QueryRequest,
    ResetResponse,
    SampleIngestResponse,
    TextIngestRequest,
)

logger = logging.getLogger(__name__)
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"
UPLOAD_FILE = File(...)


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings()
    pipeline = Pipeline(resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if resolved.auto_ingest and pipeline.store.counts().documents == 0:
            try:
                ingested = pipeline.ingest_directory(resolved.sample_dir)
            except IngestError as exc:
                logger.warning("Auto-ingest skipped: %s", exc)
            else:
                logger.info("Auto-ingested %s sample documents", len(ingested))
        yield

    app = FastAPI(
        title="Lumen Archive RAG Demo",
        version=__version__,
        description=(
            "Portfolio demo of document ingestion, local retrieval, and cited answers. "
            "Not a production search system."
        ),
        lifespan=lifespan,
    )
    app.state.pipeline = pipeline
    app.state.settings = resolved

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(STATIC_INDEX)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        current = _pipeline(request)
        counts = current.store.counts()
        return HealthResponse(
            status="ok",
            provider=current.provider.name,
            model=current.provider.model,
            credentials_configured=current.provider.credentials_configured(),
            documents=counts.documents,
            chunks=counts.chunks,
        )

    @app.get("/v1/providers", response_model=list[ProviderResponse])
    def providers(request: Request) -> list[ProviderResponse]:
        current = _settings(request)
        rows: list[ProviderResponse] = []
        for name in ProviderName:
            requires_key = API_KEY_ENV[name] is not None
            if requires_key:
                configured = current.api_key_for(name) is not None
            else:
                configured = True
            rows.append(
                ProviderResponse(
                    name=name.value,
                    default_model=DEFAULT_MODELS[name],
                    requires_api_key=requires_key,
                    active=name == current.llm_provider,
                    credentials_configured=configured,
                )
            )
        return rows

    @app.get("/v1/documents", response_model=list[DocumentResponse])
    def list_documents(request: Request) -> list[DocumentResponse]:
        return [
            DocumentResponse(id=doc_id, source=source, title=title, chunk_count=chunks)
            for doc_id, source, title, chunks in _pipeline(request).list_documents()
        ]

    @app.post("/v1/documents", response_model=IngestResponse)
    async def upload_document(request: Request, file: UploadFile = UPLOAD_FILE) -> IngestResponse:
        raw = await file.read()
        if len(raw) > _settings(request).max_upload_bytes:
            raise HTTPException(status_code=413, detail="File exceeds the demo upload limit.")
        filename = Path(file.filename or "upload.txt").name
        try:
            outcome = _pipeline(request).ingest_loaded(load_bytes(filename, raw))
        except IngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return IngestResponse.from_outcome(outcome)

    @app.post("/v1/documents/text", response_model=IngestResponse)
    def ingest_text(body: TextIngestRequest, request: Request) -> IngestResponse:
        try:
            outcome = _pipeline(request).ingest_text(
                body.text,
                source=body.source,
                title=body.title,
            )
        except IngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return IngestResponse.from_outcome(outcome)

    @app.post("/v1/documents/sample", response_model=SampleIngestResponse)
    def ingest_sample(request: Request) -> SampleIngestResponse:
        try:
            outcomes = _pipeline(request).ingest_directory(_settings(request).sample_dir)
        except IngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SampleIngestResponse(
            documents=[IngestResponse.from_outcome(outcome) for outcome in outcomes]
        )

    @app.delete("/v1/documents/{document_id}", status_code=204)
    def delete_document(document_id: str, request: Request) -> Response:
        if not _pipeline(request).delete(document_id):
            raise HTTPException(status_code=404, detail="Document not found.")
        return Response(status_code=204)

    @app.post("/v1/reset", response_model=ResetResponse)
    def reset_index(request: Request) -> ResetResponse:
        current = _pipeline(request)
        current.reset()
        counts = current.store.counts()
        return ResetResponse(documents=counts.documents, chunks=counts.chunks)

    @app.post("/v1/query", response_model=AnswerResponse)
    async def query(body: QueryRequest, request: Request) -> AnswerResponse:
        _require_credentials(request)
        try:
            return await _pipeline(request).answer(body.question, body.top_k)
        except MissingCredentialsError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/query/stream")
    async def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
        _require_credentials(request)
        current = _pipeline(request)

        async def generate() -> AsyncIterator[str]:
            try:
                async for event in current.stream_events(body.question, body.top_k):
                    yield _sse(str(event["event"]), event["data"])
            except MissingCredentialsError as exc:
                yield _sse("error", {"detail": str(exc)})
            except ProviderError as exc:
                yield _sse("error", {"detail": str(exc)})

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _require_credentials(request: Request) -> None:
    provider = _pipeline(request).provider
    if provider.credentials_configured():
        return
    raise HTTPException(
        status_code=503,
        detail=missing_key_message(_settings(request).llm_provider),
    )


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
