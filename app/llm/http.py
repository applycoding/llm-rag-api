"""Shared streaming POST used by the hosted adapters."""

from collections.abc import AsyncIterator

import httpx

from app.llm.base import USER_AGENT, ProviderError


async def post_lines(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
    timeout_s: float,
    transport: httpx.BaseTransport | None,
    provider_name: str,
) -> AsyncIterator[str]:
    request_headers = {"User-Agent": USER_AGENT, **headers}
    async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
        async with client.stream("POST", url, headers=request_headers, json=payload) as response:
            if response.status_code >= 400:
                raw = (await response.aread()).decode("utf-8", errors="replace")[:400]
                raise ProviderError(f"{provider_name} returned HTTP {response.status_code}: {raw}")
            async for line in response.aiter_lines():
                yield line
