import json
from collections.abc import AsyncIterator, Sequence

import httpx

from app.errors import ProviderError
from app.llm.http import read_error, request_failed
from app.llm.types import ChatMessage


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        temperature: float,
        timeout: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._headers = {"Content-Type": "application/json"}
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        try:
            response = await self._client.post(
                self._url,
                headers=self._headers,
                json=self._body(messages, stream=False),
            )
        except httpx.HTTPError as exc:
            raise request_failed(self.name, exc) from exc
        if response.status_code >= 400:
            raise read_error(self.name, response)
        return _content_from_payload(response.json())

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            async with self._client.stream(
                "POST",
                self._url,
                headers=self._headers,
                json=self._body(messages, stream=True),
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")[:300]
                    raise ProviderError(f"ollama returned HTTP {response.status_code}: {body}")
                async for line in response.aiter_lines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        parsed = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    text = _content_from_payload(parsed, allow_empty=True)
                    if text:
                        yield text
                    if isinstance(parsed, dict) and parsed.get("done") is True:
                        break
        except ProviderError:
            raise
        except httpx.HTTPError as exc:
            raise request_failed(self.name, exc) from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _body(self, messages: Sequence[ChatMessage], *, stream: bool) -> dict[str, object]:
        return {
            "model": self.model,
            "stream": stream,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "options": {"temperature": self._temperature},
        }


def _content_from_payload(payload: object, *, allow_empty: bool = False) -> str:
    if not isinstance(payload, dict):
        raise ProviderError("ollama returned a response that is not an object.")
    message = payload.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or (not content and not allow_empty):
        raise ProviderError("ollama returned no message content.")
    return content
