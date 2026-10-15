"""Provider interface. Live adapters speak HTTP; the fake one never leaves the process."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

from app.models import ChatMessage

USER_AGENT = "llm-rag-api-demo/0.1"

__all__ = [
    "ChatMessage",
    "LLMProvider",
    "MissingCredentialsError",
    "ProviderError",
    "USER_AGENT",
]


class ProviderError(RuntimeError):
    """The selected model host failed or returned a response we cannot use."""


class MissingCredentialsError(ProviderError):
    """A hosted provider was selected without its API key."""


class LLMProvider(ABC):
    def __init__(self, model: str) -> None:
        self.model = model

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider id, matching ProviderName values."""

    @abstractmethod
    def credentials_configured(self) -> bool:
        """True when a call can be attempted. Ollama and fake are always true."""

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        parts: list[str] = []
        async for token in self.stream(messages):
            parts.append(token)
        return "".join(parts)

    @abstractmethod
    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Yield text deltas. Implementations are async generators."""
