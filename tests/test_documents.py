import pytest

from app.documents import IngestError, infer_title, load_bytes, render_plaintext_pdf


def test_markdown_title_comes_from_the_heading() -> None:
    loaded = load_bytes("notes.md", b"# Harbor notes\n\nThe rotation is named Harbor.\n")
    assert loaded.title == "Harbor notes"
    assert "Harbor" in loaded.text


def test_text_and_markdown_extensions_are_accepted() -> None:
    loaded = load_bytes("policy.txt", b"Retention is 180 days.")
    assert loaded.source == "policy.txt"


def test_unknown_extension_is_rejected() -> None:
    with pytest.raises(IngestError, match="Unsupported"):
        load_bytes("notes.docx", b"hello")


def test_empty_document_is_rejected() -> None:
    with pytest.raises(IngestError, match="No text"):
        load_bytes("empty.txt", b"   \n")


def test_pdf_round_trip() -> None:
    raw = render_plaintext_pdf(
        [
            "Lumen Archive export rules",
            "Each exported report watermark includes the requester email.",
        ]
    )
    loaded = load_bytes("retention.pdf", raw)
    assert "requester email" in loaded.text
    assert loaded.title.startswith("Lumen Archive")


def test_infer_title_uses_filename_when_body_is_blank_lines_only() -> None:
    assert infer_title("   ", "field-notes.txt") == "field-notes"
