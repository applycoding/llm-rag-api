from pathlib import Path

from app.embeddings import HashingEmbedder
from app.store import IndexedChunk, LocalVectorIndex


def _chunk(embedder: HashingEmbedder, text: str) -> IndexedChunk:
    return IndexedChunk(text=text, start=0, end=len(text), embedding=embedder.embed(text))


def test_search_prefers_the_overlapping_chunk(tmp_path: Path) -> None:
    embedder = HashingEmbedder(dim=128)
    index = LocalVectorIndex(tmp_path / "index.json", embedder.dim)
    index.upsert(
        source="security.txt",
        title="Security",
        sha256="a",
        chunks=[_chunk(embedder, "Lumen Archive retains audit logs for 180 days.")],
    )
    index.upsert(
        source="pricing.md",
        title="Pricing",
        sha256="b",
        chunks=[_chunk(embedder, "The Lantern plan is free and includes one seat.")],
    )
    hits = index.search(embedder.embed("audit logs"), "How are audit logs retained?", top_k=2)
    assert hits[0].chunk.source == "security.txt"
    assert hits[0].score > hits[1].score if len(hits) > 1 else True


def test_unchanged_upsert_keeps_the_document_id(tmp_path: Path) -> None:
    embedder = HashingEmbedder(dim=32)
    index = LocalVectorIndex(tmp_path / "index.json", embedder.dim)
    chunk = _chunk(embedder, "Field teams reach the on-call rotation named Harbor.")
    first, unchanged = index.upsert(
        source="overview.md",
        title="Overview",
        sha256="same",
        chunks=[chunk],
    )
    assert unchanged is False
    second, unchanged = index.upsert(
        source="overview.md",
        title="Overview",
        sha256="same",
        chunks=[chunk],
    )
    assert unchanged is True
    assert second.id == first.id
    assert index.counts().documents == 1


def test_index_reloads_from_disk(tmp_path: Path) -> None:
    embedder = HashingEmbedder(dim=32)
    path = tmp_path / "index.json"
    text = "Supported regions are Iceland, Portugal, and Japan."
    LocalVectorIndex(path, embedder.dim).upsert(
        source="overview.md",
        title="Overview",
        sha256="hash",
        chunks=[_chunk(embedder, text)],
    )
    reloaded = LocalVectorIndex(path, embedder.dim)
    assert reloaded.counts().chunks == 1
    hits = reloaded.search(embedder.embed("regions"), "Which regions are supported?", top_k=1)
    assert hits[0].chunk.text == text


def test_corrupt_index_starts_empty(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    path.write_text("not-json", encoding="utf-8")
    index = LocalVectorIndex(path, dim=32)
    assert index.counts().documents == 0


def test_delete_and_clear(tmp_path: Path) -> None:
    embedder = HashingEmbedder(dim=32)
    index = LocalVectorIndex(tmp_path / "index.json", embedder.dim)
    record, _ = index.upsert(
        source="notes.txt",
        title="Notes",
        sha256="x",
        chunks=[_chunk(embedder, "Backups run nightly at 02:15 UTC.")],
    )
    assert index.delete(record.id) is True
    assert index.delete(record.id) is False
    index.upsert(
        source="notes.txt",
        title="Notes",
        sha256="y",
        chunks=[_chunk(embedder, "Backups run nightly at 02:15 UTC.")],
    )
    index.clear()
    assert index.counts().chunks == 0
