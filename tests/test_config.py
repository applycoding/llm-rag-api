import pytest
from pydantic import ValidationError

from app.config import LLMProviderName
from app.errors import ProviderConfigError
from app.llm.factory import build_llm
from tests.helpers import make_settings


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(llm_provider="azure")


def test_overlap_cannot_exceed_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        make_settings(chunk_size=100, chunk_overlap=100)


def test_blank_api_key_is_treated_as_missing() -> None:
    settings = make_settings(llm_provider=LLMProviderName.OPENAI, openai_api_key="  ")
    assert settings.openai_api_key is None
    with pytest.raises(ProviderConfigError, match="openai"):
        build_llm(settings)


def test_blank_chroma_host_uses_embedded_mode() -> None:
    settings = make_settings(chroma_host="  ")
    assert settings.chroma_host is None
    assert settings.vector_store_mode == "embedded"


def test_http_mode_is_selected_when_a_host_is_set() -> None:
    settings = make_settings(chroma_host="chroma")
    assert settings.vector_store_mode == "http"
