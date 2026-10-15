"""Chat Completions adapters for OpenAI, Mistral, and DeepSeek."""

from collections.abc import AsyncIterator, Sequence

import httpx

from app.config import ProviderName, missing_key_message
from app.llm.base import LLMProvider, MissingCredentialsError
from app.llm.http import post_lines
from app.llm.parse import parse_openai_data_line
from app.models import ChatMessage


class OpenAICompatibleProvider(LLMProvider):
    """Hosts that accept the OpenAI chat-completions streaming body."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key: str | None,
        base_url: str,
        timeout_s: float,
        max_tokens: int,
        provider_id: str,
        token_field: str = "max_tokens",
        chat_path: str = "/chat/completions",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(model)
        self._name = name
        self._provider_id = provider_id
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._token_field = token_field
        self._chat_path = chat_path if chat_path.startswith("/") else f"/{chat_path}"
        self._transport = transport

    @property
    def name(self) -> str:
        return self._name

    def credentials_configured(self) -> bool:
        return self._api_key is not None

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        if self._api_key is None:
            raise MissingCredentialsError(missing_key_message(ProviderName(self._provider_id)))
        chat_messages = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        payload: dict[str, object] = {
            "model": self.model,
            "messages": chat_messages,
            "stream": True,
            self._token_field: self._max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        url = f"{self._base_url}{self._chat_path}"
        async for line in post_lines(
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=self._timeout_s,
            transport=self._transport,
            provider_name=self._name,
        ):
            token = parse_openai_data_line(line)
            if token:
                yield token
