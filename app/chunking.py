"""Character chunker with a preference for paragraph and sentence breaks."""

from dataclasses import dataclass

_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", " ")


@dataclass(frozen=True)
class TextChunk:
    text: str
    start: int
    end: int


def chunk_text(text: str, size: int, overlap: int) -> list[TextChunk]:
    """Split text into windows of about `size` characters.

    Offsets refer to the stripped source string, after newlines are normalized.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk overlap must be >= 0 and smaller than chunk size")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    if len(normalized) <= size:
        return [TextChunk(normalized, 0, len(normalized))]

    chunks: list[TextChunk] = []
    start = 0
    length = len(normalized)
    while start < length:
        hard_end = min(start + size, length)
        end = hard_end
        if hard_end < length:
            end = start + _break_offset(normalized[start:hard_end])
        piece, piece_start, piece_end = _trim(normalized, start, end)
        if piece:
            chunks.append(TextChunk(piece, piece_start, piece_end))
        if end >= length:
            break
        next_start = max(end - overlap, start + 1)
        start = next_start
    return chunks


def _break_offset(window: str) -> int:
    floor = max(1, len(window) // 2)
    for separator in _SEPARATORS:
        index = window.rfind(separator, floor)
        if index != -1:
            return index + len(separator)
    return len(window)


def _trim(text: str, start: int, end: int) -> tuple[str, int, int]:
    raw = text[start:end]
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw) - len(raw.rstrip())
    return raw.strip(), start + leading, end - trailing
