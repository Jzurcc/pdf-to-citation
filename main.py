#!/usr/bin/env python3
"""
PDF-to-Citation CLI Tool
========================
Scans a directory of research paper PDFs, extracts DOIs (and arXiv IDs) via
regex, resolves published DOIs through Semantic Scholar, fetches CSL-JSON
from doi.org, and renders deterministic APA 7th edition HTML citations
using citeproc-py.

Pipeline:
    1. Extract text from the first 3 pages of each PDF (PyMuPDF).
    2. Regex search for a standard DOI or arXiv ID (text + filename).
    3. If arXiv ID found -> Semantic Scholar -> published DOI.
    4. If nothing found -> LLM fallback (local Ollama llama3.2:3b) -> title ->
       Semantic Scholar title search -> published DOI.
    5. Fetch CSL-JSON from doi.org -> render APA HTML with citeproc-py.
    6. Write citations to bibliography.html, failures to errors.log.

Usage:
    python main.py --email you@example.com
    python main.py --email you@example.com --test
    python main.py --email you@example.com --dir ./my_papers --output refs.html
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

# Suppress harmless citeproc-py warnings (unsupported fields & schema validation)
warnings.filterwarnings("ignore", category=UserWarning, module=r"citeproc\.source")
warnings.filterwarnings("ignore", category=UserWarning, module=r"citeproc\.frontend")

import fitz  # PyMuPDF
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOI_REGEX = re.compile(r"10\.\d{4,9}/[^\s]+", re.IGNORECASE)

# Matches arXiv IDs like: arXiv:2402.10688, arXiv:2402.10688v2,
# 2402.10688v2, arxiv.org/abs/2402.10688
ARXIV_REGEX = re.compile(
    r"(?:arXiv[:\s]*)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE
)

DOI_URL = "https://doi.org/{doi}"

SEMANTIC_SCHOLAR_PAPER_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}"
)
SEMANTIC_SCHOLAR_SEARCH_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)

CSL_JSON_HEADERS = {"Accept": "application/vnd.citationstyles.csl+json"}

APA_CSL_URL = (
    "https://raw.githubusercontent.com/"
    "citation-style-language/styles/master/apa.csl"
)
APA_CSL_FILE = "apa.csl"

REQUEST_DELAY = 0.5  # seconds between requests (polite pool)

LLM_MAX_CHARS = 3000  # max chars of extracted text sent to the LLM

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>APA 7th Edition Bibliography</title>
  <style>
    body {{
      font-family: "Times New Roman", Times, serif;
      font-size: 12pt;
      line-height: 2;
      max-width: 8.5in;
      margin: 1in auto;
    }}
    h1 {{
      text-align: center;
      font-size: 12pt;
      font-weight: bold;
    }}
    .citation {{
      padding-left: 0.5in;
      text-indent: -0.5in;
      margin-bottom: 1em;
    }}
  </style>
</head>
<body>
  <h1>References</h1>
{entries}
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------


def _setup_loggers(output_dir: Path) -> logging.Logger:
    """Create and return an error logger that writes to errors.log."""
    error_logger = logging.getLogger("pdf_citation.errors")
    error_logger.setLevel(logging.WARNING)

    # Avoid duplicate handlers on re-import
    if not error_logger.handlers:
        fh = logging.FileHandler(
            output_dir / "errors.log", mode="w", encoding="utf-8"
        )
        fh.setLevel(logging.WARNING)
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        error_logger.addHandler(fh)

    return error_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _polite_headers(email: str) -> dict:
    """Return a User-Agent header for polite-pool compliance."""
    return {"User-Agent": f"PDFtoCitation/1.0 (mailto:{email})"}


# ---------------------------------------------------------------------------
# Step 1 — Extract text from PDF
# ---------------------------------------------------------------------------


def extract_text(pdf_path: Path, max_pages: int = 3) -> Optional[str]:
    """
    Open a PDF with PyMuPDF and return the concatenated text of the first
    *max_pages* pages.  Raises RuntimeError on any read error (malformed,
    encrypted, password-protected, scanned, etc.).
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


# ---------------------------------------------------------------------------
# Step 2a — DOI Regex
# ---------------------------------------------------------------------------


def find_doi(text: str) -> Optional[str]:
    """
    Search *text* for a DOI using a broad regex pattern.
    Returns the first match (cleaned of trailing punctuation) or None.
    """
    match = DOI_REGEX.search(text)
    if match:
        doi = match.group(0)
        # Strip common trailing punctuation that is not part of the DOI
        doi = doi.rstrip(".,;:\"')}]>")
        return doi
    return None


# ---------------------------------------------------------------------------
# Step 2b — ArXiv ID Regex (text + filename)
# ---------------------------------------------------------------------------


def find_arxiv_id(text: str, filename: str) -> Optional[str]:
    """
    Search both the extracted *text* and the PDF *filename* for an arXiv ID.
    Returns the bare ID (e.g. '2402.10688v2') or None.
    """
    # Try the body text first (more likely to have the full ID)
    match = ARXIV_REGEX.search(text)
    if match:
        return match.group(1)

    # Fallback: check the filename itself
    match = ARXIV_REGEX.search(filename)
    if match:
        return match.group(1)

    return None


# ---------------------------------------------------------------------------
# Step 3a — Semantic Scholar: ArXiv ID -> published DOI
# ---------------------------------------------------------------------------


def get_semantic_scholar_doi_by_arxiv(
    arxiv_id: str, email: str
) -> Optional[str]:
    """
    Query the Semantic Scholar API with an arXiv ID and return the published
    DOI from ``externalIds.DOI``, or None if the paper has no published DOI
    (i.e. it is strictly a preprint).
    """
    # Strip version suffix for the API query (e.g. 2402.10688v2 -> 2402.10688)
    clean_id = re.sub(r"v\d+$", "", arxiv_id)

    url = SEMANTIC_SCHOLAR_PAPER_URL.format(arxiv_id=clean_id)
    headers = _polite_headers(email)
    params = {"fields": "externalIds,title"}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        doi = data.get("externalIds", {}).get("DOI")
        return doi  # may be None if paper is a preprint only
    except requests.exceptions.HTTPError:
        return None
    except requests.exceptions.RequestException:
        return None


# ---------------------------------------------------------------------------
# Step 3b — Semantic Scholar: title search -> published DOI
# ---------------------------------------------------------------------------


def get_semantic_scholar_doi_by_title(
    title: str, email: str
) -> Optional[str]:
    """
    Search Semantic Scholar by *title* and return the published DOI of the
    top result, or None.
    """
    headers = _polite_headers(email)
    params = {
        "query": title,
        "limit": 1,
        "fields": "externalIds,title",
    }

    try:
        resp = requests.get(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        papers = data.get("data", [])
        if papers:
            doi = papers[0].get("externalIds", {}).get("DOI")
            return doi
    except requests.exceptions.HTTPError:
        pass
    except requests.exceptions.RequestException:
        pass

    return None


# ---------------------------------------------------------------------------
# Step 3c — LLM Fallback (local Ollama) for title extraction
# ---------------------------------------------------------------------------


def ollama_fallback(text: str) -> Optional[dict]:
    """
    Send the first *LLM_MAX_CHARS* characters of extracted PDF text to a
    local Ollama instance running llama3.2:3b and ask it to return a JSON object
    with ``title`` and ``authors`` keys.

    Uses Ollama's ``format: "json"`` mode for strict JSON output.

    Returns a dict like ``{"title": "...", "authors": "..."}``, or None if
    the Ollama server is unreachable or the response is unusable.
    """
    prompt = f"You are a research-paper metadata extractor. Given the following text extracted from the first few pages of a research paper, identify the paper's title and authors.\n\nReturn ONLY a valid JSON object with exactly two keys:\n  \"title\": \"<full paper title>\"\n  \"authors\": \"<author names, comma-separated>\"\n\nDo NOT add any other text, explanation, or markdown formatting.\n\n--- BEGIN EXTRACTED TEXT ---\n{text[:LLM_MAX_CHARS]}\n--- END EXTRACTED TEXT ---"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        data = json.loads(raw)
        if "title" in data:
            return data
    except requests.exceptions.ConnectionError:
        print("  ⚠ Ollama server not reachable at"
              f" {OLLAMA_URL} — skipping LLM fallback")
    except requests.exceptions.RequestException as exc:
        print(f"  ⚠ Ollama request failed: {exc}")
    except (json.JSONDecodeError, KeyError):
        print("  ⚠ Ollama returned invalid JSON — skipping")

    return None


# ---------------------------------------------------------------------------
# Step 4a — Auto-download APA CSL style file
# ---------------------------------------------------------------------------


def ensure_apa_csl(script_dir: Path) -> Path:
    """
    Check if ``apa.csl`` exists next to the script.  If not, download it
    from the official CSL GitHub repository.

    Returns the resolved Path to the CSL file.
    Raises RuntimeError if the download fails.
    """
    csl_path = script_dir / APA_CSL_FILE
    if csl_path.is_file():
        return csl_path

    print(f"  … Downloading {APA_CSL_FILE} from GitHub …")
    try:
        resp = requests.get(APA_CSL_URL, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Failed to download {APA_CSL_FILE}: {exc}"
        ) from exc

    csl_path.write_text(resp.text, encoding="utf-8")
    print(f"  ✓ Saved {APA_CSL_FILE}")
    return csl_path


# ---------------------------------------------------------------------------
# Step 4b — Fetch CSL-JSON from doi.org
# ---------------------------------------------------------------------------


def fetch_csl_json(doi: str, email: str) -> dict:
    """
    Request the CSL-JSON metadata for *doi* from ``https://doi.org/{doi}``.

    Raises RuntimeError on HTTP or parsing errors.
    """
    url = DOI_URL.format(doi=doi)
    headers = {
        **CSL_JSON_HEADERS,
        **_polite_headers(email),
    }

    try:
        resp = requests.get(
            url, headers=headers, timeout=15, allow_redirects=True
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"doi.org returned HTTP {resp.status_code} for DOI '{doi}': {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Network error fetching CSL-JSON for DOI '{doi}': {exc}"
        ) from exc

    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"doi.org returned invalid JSON for DOI '{doi}': {exc}"
        ) from exc

    return data


# ---------------------------------------------------------------------------
# Step 4c — Render APA HTML citation via citeproc-py
# ---------------------------------------------------------------------------


def _clean_citation_html(html: str) -> str:
    """
    Post-process an HTML citation string to fix known citeproc-py rendering
    quirks:

    1. Remove trailing ``(as <i>[]</i>)`` artifact.
    2. Fix missing space before ``&amp;`` in author lists.
    3. Collapse double periods (``..`` -> ``.``).
    4. Ensure a space before DOI/URL links that got glued to preceding text.
    5. Collapse multiple consecutive spaces into one.
    """
    # 1. Remove "(as <i>[]</i>)" or similar empty-bracket artifacts
    html = re.sub(r"\s*\(as\s*<i>\[.*?\]</i>\)", "", html)

    # 2. Fix "P. A.&amp;" -> "P. A., &amp;"  (missing comma+space before &)
    #    and  "P. A. &amp;" is fine, but "A.&amp;" needs a space
    html = re.sub(r"(\w\.)\s*(&amp;)", r"\1, \2", html)

    # 3. Collapse double periods
    html = html.replace("..", ".")

    # 4. Add space before bare DOI URLs glued to preceding text
    #    e.g. "Elsevier BV.https://doi.org" -> "Elsevier BV. https://doi.org"
    html = re.sub(r"([.)])(\s*)(https?://)", r"\1 \3", html)

    # 5. Collapse multiple spaces
    html = re.sub(r"  +", " ", html)

    return html.strip()


def generate_html_citation(csl_json_data: dict, csl_path: Path) -> str:
    """
    Take a single CSL-JSON record (as returned by doi.org) and render it
    into an APA 7th-edition HTML string using citeproc-py, with
    post-processing to fix known rendering quirks.

    Returns an HTML fragment (e.g. containing ``<i>`` tags for journal
    titles and volume numbers).
    """
    from citeproc import (
        CitationStylesBibliography,
        CitationStylesStyle,
        Citation,
        CitationItem,
    )
    from citeproc.source.json import CiteProcJSON
    from citeproc import formatter

    # Ensure the record has an 'id' field (citeproc-py requires it)
    if "id" not in csl_json_data:
        csl_json_data["id"] = csl_json_data.get("DOI", "item1")

    source = CiteProcJSON([csl_json_data])
    style = CitationStylesStyle(str(csl_path))
    bib = CitationStylesBibliography(style, source, formatter.html)

    citation = Citation([CitationItem(csl_json_data["id"])])
    bib.register(citation)

    entries = bib.bibliography()
    if entries:
        return _clean_citation_html(str(entries[0]))

    raise RuntimeError("citeproc-py produced an empty bibliography entry.")


# ---------------------------------------------------------------------------
# Step 5 — Per-file orchestrator
# ---------------------------------------------------------------------------


def process_pdf(
    pdf_path: Path,
    email: str,
    csl_path: Path,
    error_logger: logging.Logger,
    *,
    verbose: bool = False,
) -> Optional[str]:
    """
    End-to-end processing of a single PDF:

    1. Extract text (first 3 pages).
    2. Regex: look for a standard DOI *and* an arXiv ID (text + filename).
    3. If arXiv ID found (and no DOI) -> Semantic Scholar -> published DOI.
    4. If still no DOI -> local Ollama LLM title extraction -> Semantic
       Scholar title search -> published DOI.
    5. Fetch APA citation from doi.org using content negotiation.

    Returns the APA citation string, or None (with the error logged).
    """
    filename = pdf_path.name

    # -- Extract text -------------------------------------------------------
    try:
        text = extract_text(pdf_path)
    except RuntimeError as exc:
        error_logger.warning("TEXT_EXTRACTION_FAILED | %s | %s", filename, exc)
        if verbose:
            print(f"  ✗ Text extraction failed: {exc}")
        return None

    # -- Regex: DOI ---------------------------------------------------------
    doi = find_doi(text)
    if doi:
        if verbose:
            print(f"  ✓ DOI found via regex: {doi}")

    # -- Regex: arXiv ID (text + filename) ----------------------------------
    arxiv_id = find_arxiv_id(text, filename)
    if arxiv_id and verbose:
        print(f"  ✓ ArXiv ID found: {arxiv_id}")

    # -- Route 1: arXiv -> Semantic Scholar -> published DOI ------------------
    if not doi and arxiv_id:
        if verbose:
            print(f"  … Querying Semantic Scholar for arXiv:{arxiv_id}")
        doi = get_semantic_scholar_doi_by_arxiv(arxiv_id, email)
        if doi:
            if verbose:
                print(f"  ✓ Published DOI from Semantic Scholar: {doi}")
        else:
            if verbose:
                print(
                    "  ⚠ Semantic Scholar has no published DOI for this arXiv"
                    " paper (preprint only?)"
                )
        time.sleep(REQUEST_DELAY)

    # -- Route 2: Ollama LLM fallback -> title -> Semantic Scholar search -----
    if not doi:
        if verbose:
            print("  … No DOI yet — trying Ollama LLM fallback for title")

        metadata = ollama_fallback(text)
        if metadata and metadata.get("title"):
            title = metadata["title"]
            if verbose:
                print(f"  … LLM extracted title: {title}")
            doi = get_semantic_scholar_doi_by_title(title, email)
            if doi:
                if verbose:
                    print(
                        f"  ✓ DOI found via Semantic Scholar title"
                        f" search: {doi}"
                    )
            else:
                if verbose:
                    print(
                        "  ✗ Semantic Scholar title search returned no DOI"
                    )
            time.sleep(REQUEST_DELAY)
        else:
            if verbose:
                print("  ✗ Ollama fallback returned no usable metadata")

    # -- Final check: do we have a DOI? ------------------------------------
    if not doi:
        error_logger.warning("DOI_NOT_FOUND | %s", filename)
        if verbose:
            print(f"  ✗ Could not determine DOI for '{filename}'")
        return None

    # -- Fetch CSL-JSON & render HTML citation -------------------------------
    try:
        csl_data = fetch_csl_json(doi, email)
    except RuntimeError as exc:
        error_logger.warning(
            "CSL_JSON_FETCH_FAILED | %s | DOI=%s | %s", filename, doi, exc
        )
        if verbose:
            print(f"  ✗ CSL-JSON fetch failed: {exc}")
        return None

    try:
        citation_html = generate_html_citation(csl_data, csl_path)
    except Exception as exc:
        error_logger.warning(
            "CITATION_RENDER_FAILED | %s | DOI=%s | %s", filename, doi, exc
        )
        if verbose:
            print(f"  ✗ Citation render failed: {exc}")
        return None

    return citation_html


# ---------------------------------------------------------------------------
# Step 6 — CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process a directory of research paper PDFs and generate a "
            "deterministic APA 7th edition bibliography via doi.org content "
            "negotiation, with arXiv -> Semantic Scholar DOI resolution."
        ),
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("./papers"),
        help="Directory containing PDF files (default: ./papers)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bibliography.html"),
        help="Output file for the bibliography (default: bibliography.html)",
    )
    parser.add_argument(
        "--email",
        type=str,
        required=True,
        help=(
            "Your email address — included in User-Agent headers for polite "
            "pool compliance with external APIs.  Required."
        ),
    )
    parser.add_argument(
        "--test",
        "--dry-run",
        action="store_true",
        dest="test",
        help="Process only the first PDF found (for quick verification).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress details to stdout.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    papers_dir: Path = args.dir.resolve()
    output_file: Path = args.output.resolve()
    email: str = args.email
    test_mode: bool = args.test
    verbose: bool = args.verbose

    # -- Validate input directory -------------------------------------------
    if not papers_dir.is_dir():
        print(
            f"Error: directory '{papers_dir}' does not exist.", file=sys.stderr
        )
        return 1

    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No .pdf files found in '{papers_dir}'.", file=sys.stderr)
        return 1

    if test_mode:
        pdf_files = pdf_files[:1]
        print(f"[TEST MODE] Processing only: {pdf_files[0].name}\n")

    # -- Setup --------------------------------------------------------------
    error_logger = _setup_loggers(output_file.parent)
    script_dir = Path(__file__).resolve().parent

    try:
        csl_path = ensure_apa_csl(script_dir)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    citations: list[str] = []
    total = len(pdf_files)

    print(f"Processing {total} PDF(s) from '{papers_dir}' …\n")

    # -- Main loop ----------------------------------------------------------
    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{idx}/{total}] {pdf_path.name}")

        citation = process_pdf(
            pdf_path,
            email,
            csl_path,
            error_logger,
            verbose=verbose,
        )

        if citation:
            citations.append(citation)
            if verbose:
                # Strip HTML tags for the terminal preview
                import re as _re
                preview_text = _re.sub(r"<[^>]+>", "", citation)[:120]
                ellipsis = "…" if len(preview_text) >= 120 else ""
                print(f"  -> {preview_text}{ellipsis}")

        # Polite-pool delay between iterations
        if idx < total:
            time.sleep(REQUEST_DELAY)

        print()

    # -- Write HTML output --------------------------------------------------
    if citations:
        # Sort citations alphabetically (standard for APA bibliographies)
        citations.sort(key=lambda c: re.sub(r"<[^>]+>", "", c).casefold())

        entries = "\n".join(
            f'  <p class="citation">{c}</p>' for c in citations
        )
        html = HTML_TEMPLATE.format(entries=entries)

        with open(output_file, "w", encoding="utf-8") as fh:
            fh.write(html)

        print(f"✓ {len(citations)} citation(s) written to '{output_file}'.")
    else:
        print("⚠ No citations were successfully retrieved.")

    failed = total - len(citations)
    if failed:
        log_path = output_file.parent / "errors.log"
        print(f"✗ {failed} PDF(s) failed — see '{log_path}' for details.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
