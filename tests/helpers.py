import json
from pathlib import Path

from app.config import Settings


def make_settings(tmp_path: Path | None = None, **overrides: object) -> Settings:
    values: dict[str, object] = {}
    if tmp_path is not None:
        values["chroma_persist_dir"] = str(tmp_path / "chroma")
    values.update(overrides)
    return Settings(_env_file=None, **values)


def parse_sse(raw: str) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        name = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        payload = json.loads("\n".join(data_lines))
        if not isinstance(payload, dict):
            raise AssertionError("SSE data was not an object")
        events.append((name, payload))
    return events


def minimal_pdf(text: str) -> bytes:
    """A one-page PDF whose text extraction returns ``text`` for ASCII input."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 120 Td ({escaped}) Tj ET\n".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    header = b"%PDF-1.4\n"
    body = bytearray(header)
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body.extend(f"{number} 0 obj ".encode("ascii"))
        body.extend(obj)
        body.extend(b"\nendobj\n")
    xref_at = len(body)
    total = len(objects) + 1
    body.extend(f"xref\n0 {total}\n".encode("ascii"))
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = f"trailer << /Size {total} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    body.extend(trailer.encode("ascii"))
    return bytes(body)
