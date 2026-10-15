"""Gemini streamGenerateContent adapter."""

from collections.abc import AsyncIterator, Sequence
from urllib.parse import quote

import httpx

from app.config import ProviderName, missing_key_message
from app.llm.base import LLMProvider, MissingCredentialsError
from app.llm.http import post_lines
from app.llm.parse import parse_gemini_data_line
from app.models import ChatMessage


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str,
        timeout_s: float,
        max_tokens: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(model)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._transport = transport

    @property
    def name(self) -> str:
        return ProviderName.GEMINI.value

    def credentials_configured(self) -> bool:
        return self._api_key is not None

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        if self._api_key is None:
            raise MissingCredentialsError(missing_key_message(ProviderName.GEMINI))
        system, contents = _contents(messages)
        payload: dict[str, object] = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": self._max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        model = quote(self.model, safe="")
        url = f"{self._base_url}/models/{model}:streamGenerateContent?alt=sse"
        async for line in post_lines(
            url=url,
            headers={"x-goog-api-key": self._api_key},
            payload=payload,
            timeout_s=self._timeout_s,
            transport=self._transport,
            provider_name=self.name,
        ):
            token = parse_gemini_data_line(line)
            if token:
                yield token


def _contents(messages: Sequence[ChatMessage]) -> tuple[str | None, list[dict[str, object]]]:
    systems = [message.content for message in messages if message.role == "system"]
    contents: list[dict[str, object]] = []
    for message in messages:
        if message.role == "system":
            continue
        role = "model" if message.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message.content}]})
    if not contents:
        contents.append({"role": "user", "parts": [{"text": "Answer using the excerpts."}]})
    system = "\n\n".join(systems) if systems else None
    return system, contents
