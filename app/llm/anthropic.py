"""Anthropic Messages API adapter."""

from collections.abc import AsyncIterator, Sequence

import httpx

from app.config import ProviderName, missing_key_message
from app.llm.base import LLMProvider, MissingCredentialsError
from app.llm.http import post_lines
from app.llm.parse import parse_anthropic_data_line
from app.models import ChatMessage


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str,
        version: str,
        timeout_s: float,
        max_tokens: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(model)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._version = version
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._transport = transport

    @property
    def name(self) -> str:
        return ProviderName.ANTHROPIC.value

    def credentials_configured(self) -> bool:
        return self._api_key is not None

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        if self._api_key is None:
            raise MissingCredentialsError(missing_key_message(ProviderName.ANTHROPIC))
        system, conversation = _conversation(messages)
        payload: dict[str, object] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "stream": True,
            "messages": conversation,
        }
        if system:
            payload["system"] = system
        headers = {"x-api-key": self._api_key, "anthropic-version": self._version}
        async for line in post_lines(
            url=f"{self._base_url}/v1/messages",
            headers=headers,
            payload=payload,
            timeout_s=self._timeout_s,
            transport=self._transport,
            provider_name=self.name,
        ):
            token = parse_anthropic_data_line(line)
            if token:
                yield token


def _conversation(messages: Sequence[ChatMessage]) -> tuple[str | None, list[dict[str, str]]]:
    systems = [message.content for message in messages if message.role == "system"]
    conversation = [
        {"role": message.role, "content": message.content}
        for message in messages
        if message.role != "system"
    ]
    if not conversation or conversation[0]["role"] != "user":
        conversation.insert(0, {"role": "user", "content": "Answer using the excerpts."})
    system = "\n\n".join(systems) if systems else None
    return system, conversation
