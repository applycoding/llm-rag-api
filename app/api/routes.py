import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from app.api.schemas import DocumentListOut, DocumentOut, HealthOut, QueryIn, QueryOut
from app.config import Settings
from app.domain import QueryEvent, RetrievedChunk
from app.errors import AppError, IngestError, NotFoundError
from app.rag.service import RagService

logger = logging.getLogger(__name__)

router = APIRouter()
_DOCUMENT_ID = re.compile(r"^[a-f0-9]{32}$")


@router.get("/health", response_model=HealthOut, tags=["meta"])
async def health(request: Request) -> HealthOut:
    settings: Settings = request.app.state.settings
    rag: RagService = request.app.state.rag
    try:
        await rag.ping()
    except AppError as exc:
        raise AppError("Vector store is unavailable.", status_code=503) from exc
    return HealthOut(
        status="ok",
        demo=True,
        llm_provider=settings.llm_provider.value,
        llm_model=settings.resolved_llm_model,
        embedding_provider=settings.embedding_provider.value,
        embedding_model=settings.resolved_embedding_model,
        vector_store="chroma",
        vector_store_mode=settings.vector_store_mode,
    )


@router.post("/documents", response_model=DocumentOut, status_code=201, tags=["documents"])
async def upload_document(request: Request, file: UploadFile = File(...)) -> DocumentOut:
    settings: Settings = request.app.state.settings
    rag: RagService = request.app.state.rag
    filename = Path(file.filename or "").name
    if not filename or filename in {".", ".."}:
        raise IngestError("Upload needs a filename.", status_code=400)
    try:
        data = await file.read()
    finally:
        await file.close()
    if len(data) > settings.max_upload_bytes:
        raise IngestError(
            f"File exceeds the {settings.max_upload_bytes} byte upload limit.",
            status_code=413,
        )
    document = await rag.ingest(filename=filename, data=data)
    return DocumentOut.from_summary(document)


@router.get("/documents", response_model=DocumentListOut, tags=["documents"])
async def list_documents(request: Request) -> DocumentListOut:
    rag: RagService = request.app.state.rag
    documents = await rag.list_documents()
    return DocumentListOut(documents=[DocumentOut.from_summary(item) for item in documents])


@router.delete("/documents/{document_id}", status_code=204, tags=["documents"])
async def delete_document(document_id: str, request: Request) -> Response:
    if _DOCUMENT_ID.fullmatch(document_id) is None:
        raise NotFoundError("Document was not found.")
    rag: RagService = request.app.state.rag
    deleted = await rag.delete(document_id)
    if not deleted:
        raise NotFoundError("Document was not found.")
    return Response(status_code=204)


@router.post("/query", response_model=QueryOut, tags=["query"])
async def query(body: QueryIn, request: Request) -> QueryOut:
    rag: RagService = request.app.state.rag
    answer = await rag.answer(body.question, body.top_k)
    return QueryOut.from_answer(answer)


@router.post("/query/stream", tags=["query"])
async def query_stream(body: QueryIn, request: Request) -> StreamingResponse:
    rag: RagService = request.app.state.rag
    retrieved = await rag.retrieve(body.question, body.top_k)
    return StreamingResponse(
        _event_stream(rag, body.question, retrieved),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _event_stream(
    rag: RagService,
    question: str,
    retrieved: list[RetrievedChunk],
) -> AsyncIterator[str]:
    try:
        async for event in rag.stream_answer(question, retrieved):
            yield _encode(event)
    except AppError as exc:
        logger.warning("stream failed: %s", exc)
        payload = json.dumps({"detail": str(exc)})
        yield f"event: error\ndata: {payload}\n\n"


def _encode(event: QueryEvent) -> str:
    payload = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.kind}\ndata: {payload}\n\n"
