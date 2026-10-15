"""Extractive offline provider.

It reads the same prompt the hosted models see and quotes the retrieved
sentences that share the most words with the question. That keeps the demo
and the tests runnable with no API key, and it makes citations real spans
from the index rather than generated prose.
"""

import asyncio
import re
from collections.abc import AsyncIterator, Sequence

from app.embeddings import tokenize
from app.llm.base import LLMProvider
from app.models import ChatMessage
from app.prompts import excerpts_from_messages, question_from_messages

UNKNOWN = "I do not know."
_WORD = re.compile(r"\S+\s*")


class FakeProvider(LLMProvider):
    def __init__(self, model: str = "fake-extractive") -> None:
        super().__init__(model)

    @property
    def name(self) -> str:
        return "fake"

    def credentials_configured(self) -> bool:
        return True

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        answer = extractive_answer(
            question_from_messages(messages),
            excerpts_from_messages(messages),
        )
        for match in _WORD.finditer(answer):
            yield match.group(0)
            await asyncio.sleep(0)


def extractive_answer(question: str, excerpts: Sequence[tuple[int, str]]) -> str:
    if not excerpts:
        return UNKNOWN
    question_terms = set(tokenize(question, drop_stopwords=True))
    ranked: list[tuple[int, int, str]] = []
    for index, text in excerpts:
        for sentence in _sentences(text):
            overlap = len(question_terms & set(tokenize(sentence, drop_stopwords=True)))
            if overlap:
                ranked.append((overlap, index, sentence))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not ranked:
        fallback = _sentences(excerpts[0][1])[0]
        return _cite(fallback, excerpts[0][0])
    best = ranked[0][0]
    chosen: list[str] = []
    seen: set[str] = set()
    for overlap, index, sentence in ranked:
        if overlap < best:
            break
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(_cite(sentence, index))
        if len(chosen) == 2:
            break
    return " ".join(chosen)


def _sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for block in re.split(r"\n+", text.strip()):
        parts = re.split(r"(?<=[.!?])\s+", block.strip())
        sentences.extend(part.strip() for part in parts if part.strip())
    return sentences or [text.strip()]


def _cite(sentence: str, index: int) -> str:
    trimmed = sentence.strip()
    if trimmed.endswith((".", "!", "?")):
        trimmed = trimmed[:-1].rstrip()
    return f"{trimmed} [{index}]."
