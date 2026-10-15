"""Shared records that are not HTTP schemas."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved passage, numbered the way the prompt cites it."""

    index: int
    document_id: str
    chunk_id: str
    source: str
    title: str
    score: float
    text: str


@dataclass(frozen=True)
class IngestOutcome:
    document_id: str
    source: str
    title: str
    chunk_count: int
    unchanged: bool
