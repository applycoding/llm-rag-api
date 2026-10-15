import json
from collections.abc import AsyncIterator, Sequence

import httpx

from app.errors import ProviderError
from app.llm.http import read_error, request_failed
from app.llm.messages import split_system
from app.llm.types import ChatMessage


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._url = f"{base_url.rstrip('/')}/v1/messages"
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
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
        payload = response.json()
        blocks = payload.get("content") if isinstance(payload, dict) else None
        if not isinstance(blocks, list):
            raise ProviderError("anthropic returned no content blocks.")
        texts = [
            block["text"]
            for block in blocks
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if not texts:
            raise ProviderError("anthropic returned no text content.")
        return "".join(texts)

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
                    raise ProviderError(f"anthropic returned HTTP {response.status_code}: {body}")
                async for line in response.aiter_lines():
                    text = _text_from_sse_line(line)
                    if text:
                        yield text
        except ProviderError:
            raise
        except httpx.HTTPError as exc:
            raise request_failed(self.name, exc) from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _body(self, messages: Sequence[ChatMessage], *, stream: bool) -> dict[str, object]:
        system, rest = split_system(messages)
        body: dict[str, object] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": stream,
            "messages": [{"role": message.role, "content": message.content} for message in rest],
        }
        if system is not None:
            body["system"] = system
        return body


def _text_from_sse_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    try:
        parsed = json.loads(stripped[5:].strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    delta = parsed.get("delta")
    if not isinstance(delta, dict) or delta.get("type") != "text_delta":
        return None
    text = delta.get("text")
    if isinstance(text, str) and text:
        return text
    return None
