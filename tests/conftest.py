"""Shared settings for API tests. Each test gets its own index directory."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import ProviderName, Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_provider=ProviderName.FAKE,
        data_dir=tmp_path / "data",
        sample_dir=Path("sample_docs"),
        auto_ingest=False,
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
