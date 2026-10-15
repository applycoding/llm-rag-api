from typing import assert_never

import httpx

from app.config import LLMProviderName, Settings
from app.llm.anthropic import AnthropicProvider
from app.llm.fake import FakeProvider
from app.llm.gemini import GeminiProvider
from app.llm.ollama import OllamaProvider
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.llm.types import LLMProvider
from app.secrets import require_api_key


def build_llm(settings: Settings, *, http_client: httpx.AsyncClient | None = None) -> LLMProvider:
    model = settings.resolved_llm_model
    match settings.llm_provider:
        case LLMProviderName.FAKE:
            return FakeProvider(model=model)
        case LLMProviderName.OPENAI:
            return OpenAICompatibleProvider(
                name=LLMProviderName.OPENAI.value,
                model=model,
                api_key=require_api_key(settings.openai_api_key, "openai"),
                base_url=settings.openai_base_url,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_seconds,
                http_client=http_client,
            )
        case LLMProviderName.MISTRAL:
            return OpenAICompatibleProvider(
                name=LLMProviderName.MISTRAL.value,
                model=model,
                api_key=require_api_key(settings.mistral_api_key, "mistral"),
                base_url=settings.mistral_base_url,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_seconds,
                http_client=http_client,
            )
        case LLMProviderName.DEEPSEEK:
            return OpenAICompatibleProvider(
                name=LLMProviderName.DEEPSEEK.value,
                model=model,
                api_key=require_api_key(settings.deepseek_api_key, "deepseek"),
                base_url=settings.deepseek_base_url,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_seconds,
                http_client=http_client,
            )
        case LLMProviderName.ANTHROPIC:
            return AnthropicProvider(
                model=model,
                api_key=require_api_key(settings.anthropic_api_key, "anthropic"),
                base_url=settings.anthropic_base_url,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout_seconds,
                http_client=http_client,
            )
        case LLMProviderName.GEMINI:
            return GeminiProvider(
                model=model,
                api_key=require_api_key(settings.gemini_api_key, "gemini"),
                base_url=settings.gemini_base_url,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout_seconds,
                http_client=http_client,
            )
        case LLMProviderName.OLLAMA:
            return OllamaProvider(
                model=model,
                base_url=settings.ollama_base_url,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_seconds,
                http_client=http_client,
            )
        case _ as unreachable:
            assert_never(unreachable)
