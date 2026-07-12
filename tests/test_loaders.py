import pytest

from rag.loaders import read_document


def test_reads_txt(tmp_path):
    file = tmp_path / "note.txt"
    file.write_text("plain text content", encoding="utf-8")
    assert read_document(str(file)) == "plain text content"


def test_reads_markdown(tmp_path):
    file = tmp_path / "readme.md"
    file.write_text("# Title\n\nBody.", encoding="utf-8")
    assert "Body." in read_document(str(file))


def test_extension_is_case_insensitive(tmp_path):
    file = tmp_path / "NOTE.TXT"
    file.write_text("upper case extension", encoding="utf-8")
    assert read_document(str(file)) == "upper case extension"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_document("does/not/exist.txt")


def test_unsupported_format_raises(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_document(str(file))
