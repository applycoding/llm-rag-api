from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import EmbeddingProviderName, LLMProviderName
from app.main import create_app
from tests.helpers import make_settings


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    settings = make_settings(
        tmp_path,
        llm_provider=LLMProviderName.FAKE,
        embedding_provider=EmbeddingProviderName.LOCAL,
    )
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
