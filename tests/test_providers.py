import asyncio
import json
from pathlib import Path

import httpx
import pytest

from app.config import API_KEY_ENV, DEFAULT_MODELS, ProviderName, Settings
from app.llm.base import MissingCredentialsError
from app.llm.factory import build_provider
from app.llm.fake import FakeProvider
from app.llm.parse import (
    parse_anthropic_data_line,
    parse_gemini_data_line,
    parse_ollama_line,
    parse_openai_data_line,
)
from app.models import ChatMessage, RetrievedChunk
from app.prompts import build_messages, excerpts_from_messages


def _settings(tmp_path: Path, provider: ProviderName, **keys: str | None) -> Settings:
    return Settings(
        llm_provider=provider,
        data_dir=tmp_path / "data",
        openai_api_key=keys.get("openai", "test-openai"),
        anthropic_api_key=keys.get("anthropic", "test-anthropic"),
        gemini_api_key=keys.get("gemini", "test-gemini"),
        mistral_api_key=keys.get("mistral", "test-mistral"),
        deepseek_api_key=keys.get("deepseek", "test-deepseek"),
    )


def test_catalog_covers_every_provider() -> None:
    assert set(DEFAULT_MODELS) == set(ProviderName)
    assert set(API_KEY_ENV) == set(ProviderName)


@pytest.mark.parametrize("provider", list(ProviderName))
def test_factory_builds_each_adapter(tmp_path: Path, provider: ProviderName) -> None:
    built = build_provider(_settings(tmp_path, provider))
    assert built.name == provider.value
    assert built.model == DEFAULT_MODELS[provider]
    assert built.credentials_configured() is True


def test_missing_openai_key(tmp_path: Path) -> None:
    provider = build_provider(_settings(tmp_path, ProviderName.OPENAI, openai=None))
    assert provider.credentials_configured() is False
    with pytest.raises(MissingCredentialsError, match="RAG_OPENAI_API_KEY"):
        asyncio.run(provider.complete([ChatMessage(role="user", content="hello")]))


def test_fake_quotes_the_excerpt() -> None:
    messages = build_messages(
        "How long does Lumen Archive retain audit logs?",
        [
            RetrievedChunk(
                index=1,
                document_id="doc",
                chunk_id="doc:0",
                source="security.txt",
                title="Security",
                score=0.8,
                text="Lumen Archive retains audit logs for 180 days.",
            )
        ],
    )
    assert excerpts_from_messages(messages)[0][0] == 1
    answer = asyncio.run(FakeProvider().complete(messages))
    assert answer == "Lumen Archive retains audit logs for 180 days [1]."


def test_fake_ignores_a_weaker_heading() -> None:
    messages = build_messages(
        "How long does Lumen Archive retain audit logs?",
        [
            RetrievedChunk(
                index=1,
                document_id="doc",
                chunk_id="doc:0",
                source="security.txt",
                title="Security",
                score=0.8,
                text=(
                    "Lumen Archive security notes\n\n"
                    "Lumen Archive retains audit logs for 180 days.\n"
                ),
            )
        ],
    )
    answer = asyncio.run(FakeProvider().complete(messages))
    assert answer == "Lumen Archive retains audit logs for 180 days [1]."


def _openai_transport(seen: dict[str, object]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content.decode())
        seen["auth"] = request.headers.get("authorization")
        payload = 'data: {"choices":[{"delta":{"content":"Beacon"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, text=payload)

    return httpx.MockTransport(handler)


def test_openai_compatible_stream_shapes(tmp_path: Path) -> None:
    cases = (
        (
            ProviderName.OPENAI,
            "https://api.openai.com/v1/chat/completions",
            "max_completion_tokens",
        ),
        (
            ProviderName.MISTRAL,
            "https://api.mistral.ai/v1/chat/completions",
            "max_tokens",
        ),
        (
            ProviderName.DEEPSEEK,
            "https://api.deepseek.com/chat/completions",
            "max_tokens",
        ),
    )
    for provider_name, url, token_field in cases:
        seen: dict[str, object] = {}
        provider = build_provider(
            _settings(tmp_path, provider_name),
            transport=_openai_transport(seen),
        )
        text = asyncio.run(
            provider.complete([ChatMessage(role="user", content="Which plans include SSO?")])
        )
        assert text == "Beacon"
        assert seen["url"] == url
        assert seen["auth"] == f"Bearer test-{provider_name.value}"
        body = seen["body"]
        assert isinstance(body, dict)
        assert body["stream"] is True
        assert token_field in body


def test_anthropic_request_splits_the_system_prompt(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        seen["key"] = request.headers.get("x-api-key")
        seen["version"] = request.headers.get("anthropic-version")
        payload = (
            'event: content_block_delta\n'
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}\n\n'
        )
        return httpx.Response(200, text=payload)

    provider = build_provider(
        _settings(tmp_path, ProviderName.ANTHROPIC),
        transport=httpx.MockTransport(handler),
    )
    text = asyncio.run(
        provider.complete(
            [
                ChatMessage(role="system", content="Cite excerpts."),
                ChatMessage(role="user", content="Question"),
            ]
        )
    )
    assert text == "Hi"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["system"] == "Cite excerpts."
    assert body["messages"] == [{"role": "user", "content": "Question"}]
    assert body["stream"] is True
    assert seen["key"] == "test-anthropic"
    assert seen["version"] == "2023-06-01"


def test_gemini_uses_sse_and_api_key_header(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content.decode())
        payload = 'data: {"candidates":[{"content":{"parts":[{"text":"Iceland"}]}}]}\n\n'
        return httpx.Response(200, text=payload)

    provider = build_provider(
        _settings(tmp_path, ProviderName.GEMINI),
        transport=httpx.MockTransport(handler),
    )
    text = asyncio.run(provider.complete([ChatMessage(role="user", content="regions")]))
    assert text == "Iceland"
    assert seen["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.8-flash:streamGenerateContent?alt=sse"
    )
    assert seen["key"] == "test-gemini"
    body = seen["body"]
    assert isinstance(body, dict)
    assert "systemInstruction" not in body


def test_ollama_posts_ndjson_chat(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content.decode())
        seen["auth"] = request.headers.get("authorization")
        payload = '{"message":{"role":"assistant","content":"Harbor"},"done":false}\n'
        return httpx.Response(200, text=payload)

    provider = build_provider(
        _settings(tmp_path, ProviderName.OLLAMA),
        transport=httpx.MockTransport(handler),
    )
    text = asyncio.run(provider.complete([ChatMessage(role="user", content="who is on call?")]))
    assert text == "Harbor"
    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    assert seen["auth"] is None
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == "llama3.2"
    assert body["stream"] is True


def test_hosted_http_error_becomes_provider_error(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text='{"error":"bad key"}')

    provider = build_provider(
        _settings(tmp_path, ProviderName.MISTRAL),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(Exception, match="HTTP 401"):
        asyncio.run(provider.complete([ChatMessage(role="user", content="hi")]))


def test_stream_parsers_ignore_done_and_empty_deltas() -> None:
    assert parse_openai_data_line("data: [DONE]") is None
    assert parse_openai_data_line('data: {"choices":[{"delta":{"content":"Hi"}}]}') == "Hi"
    assert (
        parse_anthropic_data_line(
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Yo"}}'
        )
        == "Yo"
    )
    assert parse_anthropic_data_line('data: {"type":"message_stop"}') is None
    gemini_line = 'data: {"candidates":[{"content":{"parts":[{"text":"A"},{"text":"B"}]}}]}'
    assert parse_gemini_data_line(gemini_line) == "AB"
    assert parse_ollama_line('{"message":{"content":""},"done":true}') is None
    assert parse_ollama_line('{"message":{"content":"ok"},"done":false}') == "ok"
