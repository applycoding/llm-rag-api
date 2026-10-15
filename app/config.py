"""Runtime settings and the provider catalog."""

from enum import StrEnum
from pathlib import Path
from typing import Self, assert_never

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderName(StrEnum):
    """Chat backends the demo knows how to call."""

    FAKE = "fake"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"


# Defaults are starting points, not a promise that a vendor keeps that id forever.
DEFAULT_MODELS: dict[ProviderName, str] = {
    ProviderName.FAKE: "fake-extractive",
    ProviderName.OPENAI: "gpt-5.4-mini",
    ProviderName.ANTHROPIC: "claude-haiku-4-5",
    ProviderName.GEMINI: "gemini-3.8-flash",
    ProviderName.MISTRAL: "mistral-small-latest",
    ProviderName.DEEPSEEK: "deepseek-flash",
    ProviderName.OLLAMA: "llama3.2",
}

API_KEY_ENV: dict[ProviderName, str | None] = {
    ProviderName.FAKE: None,
    ProviderName.OPENAI: "RAG_OPENAI_API_KEY",
    ProviderName.ANTHROPIC: "RAG_ANTHROPIC_API_KEY",
    ProviderName.GEMINI: "RAG_GEMINI_API_KEY",
    ProviderName.MISTRAL: "RAG_MISTRAL_API_KEY",
    ProviderName.DEEPSEEK: "RAG_DEEPSEEK_API_KEY",
    ProviderName.OLLAMA: None,
}


class Settings(BaseSettings):
    """Environment-backed configuration. Fields use the RAG_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        extra="ignore",
    )

    llm_provider: ProviderName = ProviderName.FAKE
    llm_model: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str | None = None
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_version: str = "2023-06-01"
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    mistral_api_key: str | None = None
    mistral_base_url: str = "https://api.mistral.ai/v1"
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    ollama_base_url: str = "http://127.0.0.1:11434"
    data_dir: Path = Path("data")
    sample_dir: Path = Path("sample_docs")
    top_k: int = Field(default=4, ge=1, le=20)
    chunk_size: int = Field(default=800, ge=200, le=4_000)
    chunk_overlap: int = Field(default=120, ge=0, le=1_000)
    embedding_dim: int = Field(default=384, ge=32, le=4_096)
    min_score: float = Field(default=0.25, ge=0.0, le=1.0)
    max_tokens: int = Field(default=800, ge=32, le=8_192)
    request_timeout_s: float = Field(default=60.0, gt=0, le=300)
    auto_ingest: bool = False
    max_upload_bytes: int = Field(default=5_000_000, ge=1_000, le=20_000_000)

    @field_validator(
        "llm_model",
        "openai_api_key",
        "anthropic_api_key",
        "gemini_api_key",
        "mistral_api_key",
        "deepseek_api_key",
        mode="before",
    )
    @classmethod
    def blank_string_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def overlap_fits_in_chunk(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    def resolved_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return DEFAULT_MODELS[self.llm_provider]

    def api_key_for(self, provider: ProviderName) -> str | None:
        match provider:
            case ProviderName.FAKE | ProviderName.OLLAMA:
                return None
            case ProviderName.OPENAI:
                return self.openai_api_key
            case ProviderName.ANTHROPIC:
                return self.anthropic_api_key
            case ProviderName.GEMINI:
                return self.gemini_api_key
            case ProviderName.MISTRAL:
                return self.mistral_api_key
            case ProviderName.DEEPSEEK:
                return self.deepseek_api_key
            case _ as unreachable:
                assert_never(unreachable)


def missing_key_message(provider: ProviderName) -> str:
    env_name = API_KEY_ENV[provider]
    if env_name is None:
        return f"{provider.value} does not use an API key."
    return (
        f"{provider.value} API key is not configured. "
        f"Set {env_name}, or run with RAG_LLM_PROVIDER=fake."
    )
