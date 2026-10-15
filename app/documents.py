"""Load PDF, Markdown, and plain text into a title plus a body."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

ALLOWED_SUFFIXES = frozenset({".pdf", ".md", ".markdown", ".txt"})
MAX_CHUNKS = 2_000


class IngestError(ValueError):
    """A document the demo cannot index, safe to show to the caller."""


@dataclass(frozen=True)
class LoadedDocument:
    source: str
    title: str
    text: str


def load_bytes(filename: str, data: bytes) -> LoadedDocument:
    source = Path(filename).name or "upload.txt"
    suffix = Path(source).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise IngestError(f"Unsupported file type '{suffix or 'none'}'. Use one of: {allowed}.")
    if suffix == ".pdf":
        text = _pdf_text(data)
    else:
        text = data.decode("utf-8-sig", errors="replace")
    text = text.replace("\x00", "").strip()
    if not text:
        raise IngestError("No text found in the document. Scanned PDFs are not supported.")
    return LoadedDocument(source=source, title=infer_title(text, source), text=text)


def infer_title(text: str, source: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if heading:
                return heading[:120]
        return stripped[:120]
    return Path(source).stem


def render_plaintext_pdf(lines: list[str]) -> bytes:
    """Build a one-page Helvetica PDF. ASCII lines only; this is a demo fixture."""
    if not lines:
        raise ValueError("lines must not be empty")
    commands = ["BT", "/F1 12 Tf", "72 740 Td", "16 TL"]
    for index, line in enumerate(lines):
        if index:
            commands.append("T*")
        safe = line.encode("ascii", errors="replace").decode("ascii")
        commands.append(f"({_pdf_escape(safe)}) Tj")
    commands.append("ET")
    stream = "\n".join(commands).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        (
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    return _assemble_pdf(objects)


def _pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(data))
    except PdfReadError as exc:
        raise IngestError("The PDF could not be read.") from exc
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _assemble_pdf(objects: list[bytes]) -> bytes:
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    parts: list[bytes] = [header]
    offsets: list[int] = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(sum(len(part) for part in parts))
        parts.append(f"{number} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref_at = sum(len(part) for part in parts)
    size = len(objects) + 1
    xref_lines = [f"xref\n0 {size}\n", "0000000000 65535 f \n"]
    xref_lines.extend(f"{offset:010d} 00000 n \n" for offset in offsets)
    trailer = f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    return b"".join(parts) + "".join(xref_lines).encode("ascii") + trailer.encode("ascii")
