"""
PDF text extraction and identifier (DOI / arXiv) regex matching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from .config import DOI_REGEX, ARXIV_REGEX


def extract_text(pdf_path: Path, max_pages: int = 3) -> Optional[str]:
    """
    Pull the text out of the first few pages of a PDF.
    Raises RuntimeError if the file can't be opened or has no text
    (e.g. scanned image PDFs).
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Cannot open PDF '{pdf_path.name}': {exc}") from exc

    pages_text: list[str] = []
    try:
        for page_num in range(min(max_pages, len(doc))):
            page = doc.load_page(page_num)
            pages_text.append(page.get_text())
    except Exception as exc:
        doc.close()
        raise RuntimeError(
            f"Error reading page(s) of '{pdf_path.name}': {exc}"
        ) from exc
    finally:
        doc.close()

    full_text = "\n".join(pages_text).strip()
    if not full_text:
        raise RuntimeError(
            f"No extractable text in '{pdf_path.name}' (scanned PDF?)."
        )

    return full_text


def _clean_doi(raw: str) -> Optional[str]:
    """Strip trailing junk and validate a raw DOI match."""
    doi = raw.rstrip(".,;:\"')}]>\n\r\t ")

    # balance parentheses — DOIs can contain (), but trailing ) is usually junk
    open_count = doi.count("(")
    close_count = doi.count(")")
    while close_count > open_count and doi.endswith(")"):
        doi = doi[:-1]
        close_count -= 1

    # must still look like a DOI after cleanup
    if len(doi) < 10 or "/" not in doi:
        return None

    return doi


def find_doi(text: str) -> Optional[str]:
    """
    Try to grab a DOI from the text using regex.
    Tries all matches (not just the first) since early matches are
    sometimes broken partial DOIs from headers or watermarks.
    Returns the best cleaned match, or None.
    """
    for match in DOI_REGEX.finditer(text):
        cleaned = _clean_doi(match.group(0))
        if cleaned:
            return cleaned
    return None


def find_arxiv_id(text: str, filename: str) -> Optional[str]:
    """
    Look for an arXiv ID in the extracted text first, then fall back to the
    filename. Returns the bare ID (e.g. '2402.10688v2') or None.
    """
    match = ARXIV_REGEX.search(text)
    if match:
        return match.group(1)

    # sometimes the filename itself is the arXiv ID
    match = ARXIV_REGEX.search(filename)
    if match:
        return match.group(1)

    return None
