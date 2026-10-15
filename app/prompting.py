from collections.abc import Sequence

from app.domain import RetrievedChunk
from app.llm.types import ChatMessage

SOURCE_PREFIX = "Sources:\n"
QUESTION_MARKER = "\n\nQuestion: "
SYSTEM_PROMPT = (
    "You answer questions using only the numbered sources. "
    "Cite sources inline as [1], [2], and so on. "
    "If the sources do not contain the answer, say you do not know. "
    "Do not invent citations or facts."
)


def build_messages(question: str, chunks: Sequence[RetrievedChunk]) -> list[ChatMessage]:
    if not chunks:
        body = "(none)"
    else:
        blocks: list[str] = []
        for index, item in enumerate(chunks, start=1):
            record = item.record
            location = record.filename
            if record.page is not None:
                location = f"{location}, page {record.page}"
            blocks.append(f"[{index}] {location} (chunk {record.chunk_index})\n{record.text}")
        body = "\n\n".join(blocks)
    user = f"{SOURCE_PREFIX}{body}{QUESTION_MARKER}{question.strip()}"
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user),
    ]
