"""Local Llama (and any other model tag) through Ollama's chat API."""

from collections.abc import AsyncIterator, Sequence

import httpx

from app.config import ProviderName
from app.llm.base import LLMProvider
from app.llm.http import post_lines
from app.llm.parse import parse_ollama_line
from app.models import ChatMessage


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        timeout_s: float,
        max_tokens: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(model)
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._transport = transport

    @property
    def name(self) -> str:
        return ProviderName.OLLAMA.value

    def credentials_configured(self) -> bool:
        return True

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "stream": True,
            "options": {"num_predict": self._max_tokens},
        }
        async for line in post_lines(
            url=f"{self._base_url}/api/chat",
            headers={},
            payload=payload,
            timeout_s=self._timeout_s,
            transport=self._transport,
            provider_name=self.name,
        ):
            token = parse_ollama_line(line)
            if token:
                yield token
