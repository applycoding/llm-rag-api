import json
from collections.abc import AsyncIterator, Sequence
from urllib.parse import quote

import httpx

from app.errors import ProviderError
from app.llm.http import read_error, request_failed
from app.llm.messages import split_system
from app.llm.types import ChatMessage


class GeminiProvider:
    name = "gemini"

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
        encoded = quote(model, safe="")
        root = base_url.rstrip("/")
        self._generate_url = f"{root}/models/{encoded}:generateContent"
        self._stream_url = f"{root}/models/{encoded}:streamGenerateContent?alt=sse"
        self._headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        try:
            response = await self._client.post(
                self._generate_url,
                headers=self._headers,
                json=self._body(messages),
            )
        except httpx.HTTPError as exc:
            raise request_failed(self.name, exc) from exc
        if response.status_code >= 400:
            raise read_error(self.name, response)
        return _text_from_payload(response.json())

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            async with self._client.stream(
                "POST",
                self._stream_url,
                headers=self._headers,
                json=self._body(messages),
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")[:300]
                    raise ProviderError(f"gemini returned HTTP {response.status_code}: {body}")
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

    def _body(self, messages: Sequence[ChatMessage]) -> dict[str, object]:
        system, rest = split_system(messages)
        contents = []
        for message in rest:
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": message.content}]})
        body: dict[str, object] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self._temperature,
                "maxOutputTokens": self._max_tokens,
            },
        }
        if system is not None:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body


def _text_from_payload(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ProviderError("gemini returned a response that is not an object.")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise ProviderError("gemini returned no candidates.")
    content = candidates[0].get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        raise ProviderError("gemini returned no text content.")
    texts = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    if not texts:
        raise ProviderError("gemini returned no text content.")
    return "".join(texts)


def _text_from_sse_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    try:
        parsed = json.loads(stripped[5:].strip())
    except json.JSONDecodeError:
        return None
    try:
        return _text_from_payload(parsed)
    except ProviderError:
        return None
