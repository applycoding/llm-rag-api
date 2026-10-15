from enum import StrEnum
from functools import lru_cache
from typing import Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderName(StrEnum):
    FAKE = "fake"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"


class EmbeddingProviderName(StrEnum):
    LOCAL = "local"
    OPENAI = "openai"


DEFAULT_LLM_MODELS: dict[LLMProviderName, str] = {
    LLMProviderName.FAKE: "fake-extractive",
    LLMProviderName.OPENAI: "gpt-4o-mini",
    LLMProviderName.ANTHROPIC: "claude-sonnet-4-5",
    LLMProviderName.GEMINI: "gemini-2.5-flash",
    LLMProviderName.MISTRAL: "mistral-small-latest",
    LLMProviderName.DEEPSEEK: "deepseek-chat",
    LLMProviderName.OLLAMA: "llama3.1",
}


class Settings(BaseSettings):
    """Runtime configuration. Environment variables use the RAG_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "llm-rag-api"
    llm_provider: LLMProviderName = LLMProviderName.FAKE
    llm_model: str | None = None
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=1, le=8192)
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=300)

    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: SecretStr | None = None
    anthropic_base_url: str = "https://api.anthropic.com"
    gemini_api_key: SecretStr | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    mistral_api_key: SecretStr | None = None
    mistral_base_url: str = "https://api.mistral.ai/v1"
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    ollama_base_url: str = "http://127.0.0.1:11434"

    embedding_provider: EmbeddingProviderName = EmbeddingProviderName.LOCAL
    embedding_model: str | None = None
    embedding_dimensions: int = Field(default=384, ge=32, le=4096)

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = Field(
        default="documents",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$",
    )
    chroma_host: str | None = None
    chroma_port: int = Field(default=8000, ge=1, le=65535)

    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=120, ge=0)
    default_top_k: int = Field(default=4, ge=1, le=20)
    max_upload_bytes: int = Field(default=8_388_608, ge=1)
    max_chunks: int = Field(default=2000, ge=1, le=10_000)

    @field_validator(
        "openai_api_key",
        "anthropic_api_key",
        "gemini_api_key",
        "mistral_api_key",
        "deepseek_api_key",
        mode="before",
    )
    @classmethod
    def blank_secret_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("chroma_host", mode="before")
    @classmethod
    def blank_host_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("llm_model", "embedding_model", mode="before")
    @classmethod
    def blank_model_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def overlap_fits_in_chunk(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def resolved_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return DEFAULT_LLM_MODELS[self.llm_provider]

    @property
    def resolved_embedding_model(self) -> str:
        if self.embedding_model:
            return self.embedding_model
        if self.embedding_provider == EmbeddingProviderName.OPENAI:
            return "text-embedding-3-small"
        return "local-hashing-v1"

    @property
    def vector_store_mode(self) -> str:
        if self.chroma_host:
            return "http"
        return "embedded"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
