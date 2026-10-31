from pathlib import Path

import pytest

from teleagent.rag.document_loader import DocumentLoader


def test_load_txt(tmp_path):

    file = tmp_path / "sample.txt"

    file.write_text(
        "Hello Telecom",
        encoding="utf-8",
    )

    loader = DocumentLoader()

    doc = loader.load(file)

    assert doc.source == "sample.txt"
    assert "Hello Telecom" in doc.content


def test_load_markdown(tmp_path):

    file = tmp_path / "guide.md"

    file.write_text(
        "# Registration",
        encoding="utf-8",
    )

    loader = DocumentLoader()

    doc = loader.load(file)

    assert doc.source == "guide.md"
    assert "Registration" in doc.content


def test_missing_file():

    loader = DocumentLoader()

    with pytest.raises(FileNotFoundError):
        loader.load("missing.txt")


def test_invalid_extension(tmp_path):

    file = tmp_path / "sample.docx"

    file.write_text("dummy")

    loader = DocumentLoader()

    with pytest.raises(ValueError):
        loader.load(file)


def test_load_directory(tmp_path):

    (tmp_path / "a.txt").write_text("AAA")
    (tmp_path / "b.md").write_text("BBB")
    (tmp_path / "ignore.doc").write_text("CCC")

    loader = DocumentLoader()

    docs = loader.load_directory(tmp_path)

    assert len(docs) == 2

    names = sorted(d.source for d in docs)

    assert names == [
        "a.txt",
        "b.md",
    ]

