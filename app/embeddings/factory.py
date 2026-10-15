from typing import assert_never

from app.config import EmbeddingProviderName, Settings
from app.embeddings.local import LocalEmbeddingProvider
from app.embeddings.openai import OpenAIEmbeddingProvider
from app.secrets import require_api_key


def build_embedder(
    settings: Settings,
) -> LocalEmbeddingProvider | OpenAIEmbeddingProvider:
    match settings.embedding_provider:
        case EmbeddingProviderName.LOCAL:
            return LocalEmbeddingProvider(dimensions=settings.embedding_dimensions)
        case EmbeddingProviderName.OPENAI:
            return OpenAIEmbeddingProvider(
                api_key=require_api_key(settings.openai_api_key, "openai embeddings"),
                model=settings.resolved_embedding_model,
                dimensions=settings.embedding_dimensions,
                base_url=settings.openai_base_url,
                timeout=settings.llm_timeout_seconds,
            )
        case _ as unreachable:
            assert_never(unreachable)
