import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import ProviderName, Settings
from app.documents import render_plaintext_pdf


def _events(body: str) -> list[tuple[str, dict]]:
    found: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data.append(line.split(":", 1)[1].strip())
        if data:
            found.append((event, json.loads("\n".join(data))))
    return found


def test_health_home_and_provider_list(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["provider"] == "fake"
    assert health.json()["documents"] == 0
    page = client.get("/")
    assert page.status_code == 200
    assert "Lumen Archive" in page.text
    assert "Load sample corpus" in page.text
    names = {row["name"] for row in client.get("/v1/providers").json()}
    assert names == {"fake", "openai", "anthropic", "gemini", "mistral", "deepseek", "ollama"}
    active = [row for row in client.get("/v1/providers").json() if row["active"]]
    assert active[0]["name"] == "fake"
    assert active[0]["credentials_configured"] is True


def test_paste_query_stream_and_delete(client: TestClient) -> None:
    created = client.post(
        "/v1/documents/text",
        json={
            "title": "Security",
            "source": "security.txt",
            "text": "Lumen Archive retains audit logs for 180 days.\n",
        },
    )
    assert created.status_code == 200
    document_id = created.json()["document_id"]
    assert created.json()["unchanged"] is False

    again = client.post(
        "/v1/documents/text",
        json={
            "title": "Security",
            "source": "security.txt",
            "text": "Lumen Archive retains audit logs for 180 days.\n",
        },
    )
    assert again.json()["unchanged"] is True
    assert again.json()["document_id"] == document_id

    unknown = client.post("/v1/query", json={"question": "What is the capital of France?"})
    assert unknown.status_code == 200
    assert unknown.json()["answer"] == "I do not know."
    assert unknown.json()["citations"] == []

    answer = client.post(
        "/v1/query",
        json={"question": "How long does Lumen Archive retain audit logs?"},
    )
    assert answer.status_code == 200
    body = answer.json()
    assert "180" in body["answer"]
    assert "[1]" in body["answer"]
    assert body["citations"][0]["source"] == "security.txt"
    assert body["provider"] == "fake"

    with client.stream(
        "POST",
        "/v1/query/stream",
        json={"question": "How long does Lumen Archive retain audit logs?"},
    ) as streamed:
        assert streamed.status_code == 200
        events = _events(streamed.read().decode())
    kinds = [event for event, _payload in events]
    assert kinds[0] == "citations"
    assert "token" in kinds
    assert kinds[-1] == "done"
    done = events[-1][1]
    assert "180" in done["answer"]

    listed = client.get("/v1/documents")
    assert listed.json()[0]["id"] == document_id
    assert client.delete(f"/v1/documents/{document_id}").status_code == 204
    assert client.delete(f"/v1/documents/{document_id}").status_code == 404
    assert client.get("/v1/documents").json() == []


def test_upload_pdf_markdown_and_reject_unknown(client: TestClient) -> None:
    pdf = render_plaintext_pdf(
        [
            "Lumen Archive export rules",
            "Each exported report watermark includes the requester email.",
        ]
    )
    uploaded = client.post(
        "/v1/documents",
        files={"file": ("retention.pdf", pdf, "application/pdf")},
    )
    assert uploaded.status_code == 200
    markdown = client.post(
        "/v1/documents",
        files={
            "file": (
                "overview.md",
                b"# Regions\n\nSupported regions are Iceland.\n",
                "text/markdown",
            )
        },
    )
    assert markdown.status_code == 200
    rejected = client.post(
        "/v1/documents",
        files={"file": ("notes.docx", b"nope", "application/octet-stream")},
    )
    assert rejected.status_code == 422
    reset = client.post("/v1/reset")
    assert reset.status_code == 200
    assert reset.json() == {"documents": 0, "chunks": 0}


def test_sample_corpus_and_blank_question(client: TestClient) -> None:
    loaded = client.post("/v1/documents/sample")
    assert loaded.status_code == 200
    sources = {item["source"] for item in loaded.json()["documents"]}
    assert sources == {
        "overview.md",
        "pricing.md",
        "security.txt",
        "operations.md",
        "retention.pdf",
    }
    blank = client.post("/v1/query", json={"question": "   "})
    assert blank.status_code == 422


def test_missing_provider_key_returns_503(tmp_path: Path) -> None:
    settings = Settings(
        llm_provider=ProviderName.OPENAI,
        openai_api_key=None,
        data_dir=tmp_path / "data",
        auto_ingest=False,
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/query", json={"question": "Hello"})
        assert response.status_code == 503
        assert "RAG_OPENAI_API_KEY" in response.json()["detail"]
        streamed = client.post("/v1/query/stream", json={"question": "Hello"})
        assert streamed.status_code == 503


def test_auto_ingest_on_startup(tmp_path: Path) -> None:
    settings = Settings(
        llm_provider=ProviderName.FAKE,
        data_dir=tmp_path / "data",
        sample_dir=Path("sample_docs"),
        auto_ingest=True,
    )
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
    assert health["documents"] == 5
    assert health["chunks"] >= 5
