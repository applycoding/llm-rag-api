from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.errors import IngestError
from app.ingestion.chunker import TextSegment

MEDIA_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
}


def media_type_for(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    media_type = MEDIA_TYPES.get(suffix)
    if media_type is None:
        raise IngestError(
            "Unsupported file type. Upload a .pdf, .md, .markdown, or .txt file.",
            status_code=415,
        )
    return media_type


def load_document(filename: str, data: bytes) -> list[TextSegment]:
    media_type = media_type_for(filename)
    if media_type == "application/pdf":
        return _load_pdf(data)
    return _load_text(data)


def _load_text(data: bytes) -> list[TextSegment]:
    text = data.decode("utf-8", errors="replace").replace("\x00", "").strip()
    if not text:
        raise IngestError("The file has no extractable text.", status_code=400)
    return [TextSegment(text=text, page=None)]


def _load_pdf(data: bytes) -> list[TextSegment]:
    try:
        reader = PdfReader(BytesIO(data))
    except PdfReadError as exc:
        raise IngestError("Could not read that PDF.", status_code=400) from exc
    if reader.is_encrypted:
        raise IngestError("Encrypted PDFs are not supported.", status_code=400)

    segments: list[TextSegment] = []
    for page_number, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        text = extracted.replace("\x00", "").strip()
        if text:
            segments.append(TextSegment(text=text, page=page_number))
    if not segments:
        raise IngestError("The PDF has no extractable text.", status_code=400)
    return segments
