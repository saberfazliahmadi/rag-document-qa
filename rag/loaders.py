"""Document loaders for plain text, PDF, and Word files."""

import os

import docx
import PyPDF2

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")


def read_text_file(file_path: str) -> str:
    """Return the content of a plain text or Markdown file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_pdf_file(file_path: str) -> str:
    """Extract text from every page of a PDF file."""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_docx_file(file_path: str) -> str:
    """Extract text from a Word (.docx) document."""
    document = docx.Document(file_path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def read_document(file_path: str) -> str:
    """Load a document, dispatching to the right reader by file extension.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the file extension is not supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = os.path.splitext(file_path)[1].lower()
    if extension in (".txt", ".md"):
        return read_text_file(file_path)
    if extension == ".pdf":
        return read_pdf_file(file_path)
    if extension == ".docx":
        return read_docx_file(file_path)

    raise ValueError(
        f"Unsupported file format: '{extension}'. "
        f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
    )
