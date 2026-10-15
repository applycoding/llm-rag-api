from dataclasses import dataclass


@dataclass(frozen=True)
class TextSegment:
    text: str
    page: int | None


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    page: int | None
    chunk_index: int


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Split text into windows, preferring a newline or space near the end."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window = normalized[start:end]
            break_at = window.rfind("\n")
            if break_at < chunk_size // 2:
                space_at = window.rfind(" ")
                if space_at >= chunk_size // 2:
                    break_at = space_at
            if break_at >= chunk_size // 2:
                end = start + break_at
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def chunk_segments(
    segments: list[TextSegment],
    *,
    chunk_size: int,
    overlap: int,
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    index = 0
    for segment in segments:
        for piece in chunk_text(segment.text, chunk_size=chunk_size, overlap=overlap):
            drafts.append(ChunkDraft(text=piece, page=segment.page, chunk_index=index))
            index += 1
    return drafts
