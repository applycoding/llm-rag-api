import json
from collections.abc import AsyncIterator, Sequence

import httpx

from app.errors import ProviderError
from app.llm.http import read_error, request_failed
from app.llm.types import ChatMessage

_DONE = object()


class OpenAICompatibleProvider:
    """Chat Completions client used for OpenAI, Mistral, and DeepSeek."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float,
        timeout: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self._temperature = temperature
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        payload = await self._post(messages, stream=False)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError(f"{self.name} returned no choices.")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content:
            raise ProviderError(f"{self.name} returned no message content.")
        return content

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        async for delta in self._stream(messages):
            yield delta

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _body(self, messages: Sequence[ChatMessage], *, stream: bool) -> dict[str, object]:
        return {
            "model": self.model,
            "temperature": self._temperature,
            "stream": stream,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
        }

    async def _post(self, messages: Sequence[ChatMessage], *, stream: bool) -> dict[str, object]:
        try:
            response = await self._client.post(
                self._url,
                headers=self._headers,
                json=self._body(messages, stream=stream),
            )
        except httpx.HTTPError as exc:
            raise request_failed(self.name, exc) from exc
        if response.status_code >= 400:
            raise read_error(self.name, response)
        payload = response.json()
        if not isinstance(payload, dict):
            raise ProviderError(f"{self.name} returned a response that is not an object.")
        return payload

    async def _stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            async with self._client.stream(
                "POST",
                self._url,
                headers=self._headers,
                json=self._body(messages, stream=True),
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")[:300]
                    raise ProviderError(f"{self.name} returned HTTP {response.status_code}: {body}")
                async for line in response.aiter_lines():
                    delta = _delta_from_sse_line(line)
                    if delta is _DONE:
                        break
                    if isinstance(delta, str):
                        yield delta
        except ProviderError:
            raise
        except httpx.HTTPError as exc:
            raise request_failed(self.name, exc) from exc


def _delta_from_sse_line(line: str) -> str | object | None:
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    data = stripped[5:].strip()
    if data == "[DONE]":
        return _DONE
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return None
    content = delta.get("content")
    if isinstance(content, str) and content:
        return content
    return None
