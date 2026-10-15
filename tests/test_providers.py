import json

import httpx
import pytest

from app.config import DEFAULT_LLM_MODELS, EmbeddingProviderName, LLMProviderName
from app.embeddings.factory import build_embedder
from app.embeddings.openai import OpenAIEmbeddingProvider
from app.errors import ProviderConfigError, ProviderError
from app.llm.anthropic import AnthropicProvider
from app.llm.factory import build_llm
from app.llm.gemini import GeminiProvider
from app.llm.ollama import OllamaProvider
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.llm.types import ChatMessage
from tests.helpers import make_settings

_KEYS = {
    "openai_api_key": "sk-openai",
    "anthropic_api_key": "sk-ant",
    "gemini_api_key": "gem-key",
    "mistral_api_key": "sk-mistral",
    "deepseek_api_key": "sk-deepseek",
}


def _user_message() -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content="Cite sources."),
        ChatMessage(role="user", content="Hello"),
    ]


@pytest.mark.asyncio
async def test_factory_builds_every_configured_provider() -> None:
    for name in LLMProviderName:
        provider = build_llm(make_settings(llm_provider=name, **_KEYS))
        assert provider.name == name.value
        assert provider.model == DEFAULT_LLM_MODELS[name]
        await provider.aclose()


@pytest.mark.parametrize(
    ("provider_name", "key_field"),
    [
        (LLMProviderName.OPENAI, "openai_api_key"),
        (LLMProviderName.ANTHROPIC, "anthropic_api_key"),
        (LLMProviderName.GEMINI, "gemini_api_key"),
        (LLMProviderName.MISTRAL, "mistral_api_key"),
        (LLMProviderName.DEEPSEEK, "deepseek_api_key"),
    ],
)
def test_cloud_providers_require_an_api_key(provider_name: LLMProviderName, key_field: str) -> None:
    settings = make_settings(llm_provider=provider_name, **{key_field: None})
    with pytest.raises(ProviderConfigError, match=provider_name.value):
        build_llm(settings)


def test_openai_embeddings_require_an_api_key() -> None:
    settings = make_settings(embedding_provider=EmbeddingProviderName.OPENAI, openai_api_key=None)
    with pytest.raises(ProviderConfigError, match="openai embeddings"):
        build_embedder(settings)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "base_url", "model"),
    [
        ("openai", "https://api.openai.com/v1", "gpt-4o-mini"),
        ("mistral", "https://api.mistral.ai/v1", "mistral-small-latest"),
        ("deepseek", "https://api.deepseek.com", "deepseek-chat"),
    ],
)
async def test_openai_compatible_complete_and_stream(name: str, base_url: str, model: str) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        if body["stream"]:
            payload = "\n".join(
                [
                    'data: {"choices":[{"delta":{"role":"assistant"}}]}',
                    'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                    'data: {"choices":[{"delta":{"content":" there"}}]}',
                    "data: [DONE]",
                ]
            )
            return httpx.Response(200, text=payload)
        assert body["model"] == model
        assert body["messages"][0]["role"] == "system"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Hello there"}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = OpenAICompatibleProvider(
            name=name,
            model=model,
            api_key="secret",
            base_url=base_url,
            temperature=0,
            timeout=5,
            http_client=http,
        )
        assert await provider.complete(_user_message()) == "Hello there"
        streamed = "".join([piece async for piece in provider.stream(_user_message())])

    assert streamed == "Hello there"
    assert seen[0].url == f"{base_url}/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer secret"
    assert json.loads(seen[0].content)["stream"] is False
    assert json.loads(seen[1].content)["stream"] is True


@pytest.mark.asyncio
async def test_openai_compatible_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = OpenAICompatibleProvider(
            name="openai",
            model="gpt-4o-mini",
            api_key="secret",
            base_url="https://api.openai.com/v1",
            temperature=0,
            timeout=5,
            http_client=http,
        )
        with pytest.raises(ProviderError, match="401"):
            await provider.complete(_user_message())


@pytest.mark.asyncio
async def test_anthropic_request_shape_and_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.headers["x-api-key"] == "sk-ant"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert request.url == "https://api.anthropic.com/v1/messages"
        assert body["system"] == "Cite sources."
        assert body["messages"] == [{"role": "user", "content": "Hello"}]
        assert body["max_tokens"] == 256
        if body["stream"]:
            hi = '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}'
            bang = '{"type":"content_block_delta","delta":{"type":"text_delta","text":"!"}}'
            payload = "\n".join(
                [
                    "event: content_block_delta",
                    f"data: {hi}",
                    "",
                    "event: content_block_delta",
                    f"data: {bang}",
                ]
            )
            return httpx.Response(200, text=payload)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Hi!"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = AnthropicProvider(
            model="claude-sonnet-4-5",
            api_key="sk-ant",
            base_url="https://api.anthropic.com",
            temperature=0,
            max_tokens=256,
            timeout=5,
            http_client=http,
        )
        assert await provider.complete(_user_message()) == "Hi!"
        streamed = "".join([piece async for piece in provider.stream(_user_message())])
    assert streamed == "Hi!"


@pytest.mark.asyncio
async def test_gemini_keeps_the_key_out_of_the_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "gem-key"
        assert "key=" not in str(request.url)
        body = json.loads(request.content)
        assert body["systemInstruction"]["parts"][0]["text"] == "Cite sources."
        assert body["contents"][0]["role"] == "user"
        if ":streamGenerateContent" in str(request.url):
            assert "alt=sse" in str(request.url)
            payload = 'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}\n'
            return httpx.Response(200, text=payload)
        assert str(request.url).endswith("/models/gemini-2.5-flash:generateContent")
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = GeminiProvider(
            model="gemini-2.5-flash",
            api_key="gem-key",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            temperature=0,
            max_tokens=128,
            timeout=5,
            http_client=http,
        )
        assert await provider.complete(_user_message()) == "Hi"
        streamed = "".join([piece async for piece in provider.stream(_user_message())])
    assert streamed == "Hi"


@pytest.mark.asyncio
async def test_ollama_parses_ndjson_chat() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://127.0.0.1:11434/api/chat"
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        assert body["model"] == "llama3.1"
        assert body["messages"][0]["role"] == "system"
        if body["stream"]:
            payload = "\n".join(
                [
                    '{"message":{"role":"assistant","content":"Hi"},"done":false}',
                    '{"message":{"role":"assistant","content":"!"},"done":true}',
                ]
            )
            return httpx.Response(200, text=payload)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Hi!"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = OllamaProvider(
            model="llama3.1",
            base_url="http://127.0.0.1:11434",
            temperature=0,
            timeout=5,
            http_client=http,
        )
        assert await provider.complete(_user_message()) == "Hi!"
        streamed = "".join([piece async for piece in provider.stream(_user_message())])
    assert streamed == "Hi!"


@pytest.mark.asyncio
async def test_openai_embeddings_request_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.openai.com/v1/embeddings"
        assert request.headers["authorization"] == "Bearer sk-openai"
        body = json.loads(request.content)
        assert body == {
            "model": "text-embedding-3-small",
            "input": ["alpha", "beta"],
            "dimensions": 384,
        }
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.2, 0.8]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = OpenAIEmbeddingProvider(
            api_key="sk-openai",
            model="text-embedding-3-small",
            dimensions=384,
            base_url="https://api.openai.com/v1",
            timeout=5,
            http_client=http,
        )
        vectors = await provider.embed(["alpha", "beta"])
    assert vectors == [[1.0, 0.0], [0.2, 0.8]]
