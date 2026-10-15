import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from app.llm.types import ChatMessage
from app.prompting import QUESTION_MARKER, SOURCE_PREFIX
from app.textutil import tokenize

_HEADER = re.compile(r"^\[(\d+)\] .+\(chunk \d+\)$")
_HEADING = re.compile(r"#{1,6}\s+.+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH = re.compile(r"\n\s*\n")
_NO_ANSWER = "The supplied sources do not contain an answer to that question."


class FakeProvider:
    """Extractive offline generator. It quotes retrieved sources and cites them."""

    name = "fake"

    def __init__(self, *, model: str) -> None:
        self.model = model

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        return render_answer(messages)

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        for piece in stream_pieces(render_answer(messages)):
            yield piece

    async def aclose(self) -> None:
        return None


@dataclass(frozen=True)
class _ScoredSentence:
    overlap: int
    source_index: int
    sentence_index: int
    sentence: str


def render_answer(messages: Sequence[ChatMessage]) -> str:
    question, sources = parse_prompt(_last_user(messages))
    if not sources:
        return _NO_ANSWER
    question_tokens = set(tokenize(question))
    scored: list[_ScoredSentence] = []
    for source_index, text in sources:
        for sentence_index, sentence in enumerate(_sentences(text)):
            overlap = len(set(tokenize(sentence)) & question_tokens)
            if overlap:
                scored.append(
                    _ScoredSentence(
                        overlap=overlap,
                        source_index=source_index,
                        sentence_index=sentence_index,
                        sentence=sentence,
                    )
                )
    if not scored:
        return _NO_ANSWER
    scored.sort(key=lambda item: (-item.overlap, item.source_index, item.sentence_index))
    chosen: list[_ScoredSentence] = []
    seen: set[tuple[str, int]] = set()
    for item in scored:
        key = (item.sentence, item.source_index)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(item)
        if len(chosen) == 2:
            break
    return " ".join(f"{_finish_sentence(item.sentence)} [{item.source_index}]" for item in chosen)


def parse_prompt(user_content: str) -> tuple[str, list[tuple[int, str]]]:
    """Read the source layout produced by ``build_messages``."""
    if QUESTION_MARKER not in user_content:
        return user_content.strip(), []
    source_part, question = user_content.rsplit(QUESTION_MARKER, 1)
    body = source_part.removeprefix(SOURCE_PREFIX).strip()
    if not body or body == "(none)":
        return question.strip(), []

    sources: list[tuple[int, str]] = []
    current_index: int | None = None
    current_lines: list[str] = []
    for line in body.split("\n"):
        header = _HEADER.match(line.strip())
        if header:
            if current_index is not None:
                sources.append((current_index, "\n".join(current_lines).strip()))
            current_index = int(header.group(1))
            current_lines = []
            continue
        if current_index is not None:
            current_lines.append(line)
    if current_index is not None:
        sources.append((current_index, "\n".join(current_lines).strip()))
    return question.strip(), sources


def stream_pieces(text: str) -> list[str]:
    if not text:
        return []
    words = text.split(" ")
    pieces: list[str] = []
    for index, word in enumerate(words):
        if index < len(words) - 1:
            pieces.append(f"{word} ")
        elif word:
            pieces.append(word)
    return pieces


def _last_user(messages: Sequence[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _sentences(text: str) -> list[str]:
    pieces: list[str] = []
    for paragraph in _PARAGRAPH.split(text):
        paragraph = paragraph.strip()
        if not paragraph or _HEADING.fullmatch(paragraph):
            continue
        parts = [part.strip() for part in _SENTENCE.split(paragraph) if part.strip()]
        pieces.extend(parts)
    if pieces:
        return pieces
    stripped = text.strip()
    return [stripped] if stripped else []


def _finish_sentence(sentence: str) -> str:
    if sentence.endswith((".", "!", "?")):
        return sentence
    return f"{sentence}."
