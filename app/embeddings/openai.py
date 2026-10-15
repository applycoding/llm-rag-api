from collections.abc import Sequence

import httpx

from app.errors import ProviderError
from app.llm.http import read_error, request_failed


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        base_url: str,
        timeout: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        ordered: list[list[float] | None] = [None] * len(texts)
        batch_size = 128
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            payload = await self._post(batch)
            for item in payload:
                index = int(item["index"])
                embedding = item.get("embedding")
                if not isinstance(embedding, list):
                    raise ProviderError("openai embeddings response is missing vectors.")
                ordered[start + index] = [float(value) for value in embedding]
        if any(vector is None for vector in ordered):
            raise ProviderError("openai embeddings response did not cover every input.")
        return [vector for vector in ordered if vector is not None]

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post(self, texts: list[str]) -> list[dict[str, object]]:
        url = f"{self._base_url}/embeddings"
        body = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
        }
        try:
            response = await self._client.post(url, headers=self._headers, json=body)
        except httpx.HTTPError as exc:
            raise request_failed("openai", exc) from exc
        if response.status_code >= 400:
            raise read_error("openai", response)
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list):
            raise ProviderError("openai embeddings response has no data list.")
        return data
