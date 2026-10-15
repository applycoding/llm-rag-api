from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.ingestion.loaders import load_document
from app.rag.service import NO_SOURCES_ANSWER
from tests.helpers import minimal_pdf, parse_sse

STORAGE = "Embeddings are stored in Chroma on disk."
STREAMING = "The query API streams tokens and returns citations."


def _upload(client: TestClient, filename: str, content: str) -> dict[str, object]:
    response = client.post(
        "/documents",
        files={"file": (filename, content.encode(), "text/plain")},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert isinstance(body, dict)
    return body


def test_health_reports_the_offline_demo(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["demo"] is True
    assert body["llm_provider"] == "fake"
    assert body["llm_model"] == "fake-extractive"
    assert body["embedding_provider"] == "local"
    assert body["vector_store"] == "chroma"
    assert body["vector_store_mode"] == "embedded"
    assert "Failed to send telemetry" not in caplog.text


def test_empty_corpus_does_not_invent_an_answer(client: TestClient) -> None:
    response = client.post("/query", json={"question": "Where are embeddings stored?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == NO_SOURCES_ANSWER
    assert body["citations"] == []
    assert body["provider"] == "fake"


def test_upload_query_and_stream_cite_the_matching_document(client: TestClient) -> None:
    stored = _upload(client, "storage.md", STORAGE)
    _upload(client, "streaming.txt", STREAMING)
    listed = client.get("/documents")
    assert listed.status_code == 200
    assert len(listed.json()["documents"]) == 2
    assert stored["chunk_count"] == 1

    response = client.post("/query", json={"question": "Where are embeddings stored?", "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["citations"][0]["filename"] == "storage.md"
    assert body["citations"][0]["page"] is None
    assert "Chroma" in body["answer"]
    assert "[1]" in body["answer"]

    streamed = client.post(
        "/query/stream",
        json={"question": "Where are embeddings stored?", "top_k": 2},
    )
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(streamed.text)
    assert events[0][0] == "citations"
    assert events[-1][0] == "done"
    token_text = "".join(str(payload["text"]) for name, payload in events if name == "token")
    assert token_text == events[-1][1]["answer"]
    assert token_text == body["answer"]
    assert len([name for name, _payload in events if name == "token"]) > 1
    citations = events[0][1]["citations"]
    assert isinstance(citations, list)
    assert citations[0]["filename"] == "storage.md"


def test_pdf_upload_keeps_the_page_number(client: TestClient) -> None:
    payload = minimal_pdf(STORAGE)
    assert "Chroma" in load_document("notes.pdf", payload)[0].text
    response = client.post("/documents", files={"file": ("notes.pdf", payload, "application/pdf")})
    assert response.status_code == 201, response.text
    queried = client.post("/query", json={"question": "Where are embeddings stored?"})
    citation = queried.json()["citations"][0]
    assert citation["filename"] == "notes.pdf"
    assert citation["page"] == 1


def test_rejects_unsupported_empty_and_unreadable_files(client: TestClient) -> None:
    unsupported = client.post(
        "/documents", files={"file": ("notes.exe", b"hello", "application/octet-stream")}
    )
    assert unsupported.status_code == 415

    empty = client.post("/documents", files={"file": ("notes.txt", b"  \n", "text/plain")})
    assert empty.status_code == 400

    corrupt = client.post(
        "/documents", files={"file": ("notes.pdf", b"not a pdf", "application/pdf")}
    )
    assert corrupt.status_code == 400


def test_delete_removes_a_document_from_later_answers(client: TestClient) -> None:
    uploaded = _upload(client, "storage.txt", STORAGE)
    document_id = str(uploaded["id"])
    assert client.delete(f"/documents/{document_id}").status_code == 204
    missing = client.delete(f"/documents/{document_id}")
    assert missing.status_code == 404
    assert client.delete("/documents/not-an-id").status_code == 404

    response = client.post("/query", json={"question": "Where are embeddings stored?"})
    assert response.json()["citations"] == []
    assert response.json()["answer"] == NO_SOURCES_ANSWER


def test_sample_note_matches_the_documented_answer(client: TestClient) -> None:
    sample = Path(__file__).parents[1] / "examples" / "retrieval-notes.md"
    uploaded = client.post(
        "/documents",
        files={"file": ("retrieval-notes.md", sample.read_bytes(), "text/markdown")},
    )
    assert uploaded.status_code == 201, uploaded.text

    body = client.post("/query", json={"question": "Where are embeddings stored?"}).json()
    assert body["answer"] == (
        "Embeddings are stored on disk for similarity search. [1] "
        "This demo stores embeddings in Chroma. [1]"
    )
    assert body["citations"][0]["filename"] == "retrieval-notes.md"
    assert body["citations"][0]["score"] == 0.325


def test_blank_question_is_rejected(client: TestClient) -> None:
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 422
