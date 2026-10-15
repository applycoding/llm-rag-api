from app.store.chroma import build_chroma_client
from tests.helpers import make_settings


def test_chroma_client_uses_http_when_a_host_is_configured(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_http_client(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("app.store.chroma.chromadb.HttpClient", fake_http_client)
    build_chroma_client(make_settings(chroma_host="chroma", chroma_port=8001))
    assert captured["host"] == "chroma"
    assert captured["port"] == 8001
