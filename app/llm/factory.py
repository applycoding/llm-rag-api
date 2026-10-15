"""Select a chat adapter from settings."""

from typing import assert_never

import httpx

from app.config import ProviderName, Settings
from app.llm.anthropic import AnthropicProvider
from app.llm.base import LLMProvider
from app.llm.fake import FakeProvider
from app.llm.gemini import GeminiProvider
from app.llm.ollama import OllamaProvider
from app.llm.openai_compatible import OpenAICompatibleProvider


def build_provider(
    settings: Settings,
    transport: httpx.BaseTransport | None = None,
) -> LLMProvider:
    model = settings.resolved_model()
    timeout_s = settings.request_timeout_s
    max_tokens = settings.max_tokens
    provider = settings.llm_provider
    match provider:
        case ProviderName.FAKE:
            return FakeProvider(model)
        case ProviderName.OPENAI:
            return _openai_compatible(
                settings,
                name=provider.value,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                token_field="max_completion_tokens",
                transport=transport,
            )
        case ProviderName.MISTRAL:
            return _openai_compatible(
                settings,
                name=provider.value,
                api_key=settings.mistral_api_key,
                base_url=settings.mistral_base_url,
                token_field="max_tokens",
                transport=transport,
            )
        case ProviderName.DEEPSEEK:
            return _openai_compatible(
                settings,
                name=provider.value,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                token_field="max_tokens",
                transport=transport,
            )
        case ProviderName.ANTHROPIC:
            return AnthropicProvider(
                model=model,
                api_key=settings.anthropic_api_key,
                base_url=settings.anthropic_base_url,
                version=settings.anthropic_version,
                timeout_s=timeout_s,
                max_tokens=max_tokens,
                transport=transport,
            )
        case ProviderName.GEMINI:
            return GeminiProvider(
                model=model,
                api_key=settings.gemini_api_key,
                base_url=settings.gemini_base_url,
                timeout_s=timeout_s,
                max_tokens=max_tokens,
                transport=transport,
            )
        case ProviderName.OLLAMA:
            return OllamaProvider(
                model=model,
                base_url=settings.ollama_base_url,
                timeout_s=timeout_s,
                max_tokens=max_tokens,
                transport=transport,
            )
        case _ as unreachable:
            assert_never(unreachable)


def _openai_compatible(
    settings: Settings,
    *,
    name: str,
    api_key: str | None,
    base_url: str,
    token_field: str,
    transport: httpx.BaseTransport | None,
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name=name,
        provider_id=name,
        model=settings.resolved_model(),
        api_key=api_key,
        base_url=base_url,
        timeout_s=settings.request_timeout_s,
        max_tokens=settings.max_tokens,
        token_field=token_field,
        transport=transport,
    )
