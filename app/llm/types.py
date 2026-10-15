from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


class LLMProvider(Protocol):
    name: str
    model: str

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Return the full assistant text."""

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Yield assistant text as it arrives."""

    async def aclose(self) -> None:
        """Release HTTP clients owned by the provider."""
