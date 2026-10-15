import httpx

from app.errors import ProviderError


def request_failed(provider: str, exc: httpx.HTTPError) -> ProviderError:
    return ProviderError(f"{provider} request failed ({exc.__class__.__name__}).")


def read_error(provider: str, response: httpx.Response) -> ProviderError:
    body = response.text[:300].replace("\n", " ")
    return ProviderError(f"{provider} returned HTTP {response.status_code}: {body}")
