"""
Per-PDF processing pipeline — ties together extraction, resolution,
LLM fallback, and citation rendering into one function.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from .config import REQUEST_DELAY
from .extractors import extract_text, find_doi, find_arxiv_id
from .resolvers import get_doi_by_arxiv, get_doi_by_title, get_doi_by_title_crossref
from .llm import together_fallback, ollama_fallback
from .citation import fetch_csl_json, render_citation


def process_pdf(
    pdf_path: Path,
    email: str,
    csl_path: Path,
    error_logger: logging.Logger,
    *,
    verbose: bool = False,
) -> tuple[Optional[str], dict]:
    """
    Run the full pipeline on a single PDF:
      1. Extract text from the first few pages
      2. Regex for DOI / arXiv ID
      3. arXiv -> Semantic Scholar -> published DOI
      4. If still nothing -> LLM title extraction -> S2 title search -> DOI
      5. Fetch CSL-JSON and render APA citation

    Returns (citation_html_or_None, stats_dict).
    The stats dict always has keys: found, title, link.
    """
    filename = pdf_path.name
    stats = {"found": False, "title": None, "link": None, "reason": None}

    # -- extract text --------------------------------------------------------
    try:
        text = extract_text(pdf_path)
    except RuntimeError as exc:
        error_logger.warning("TEXT_EXTRACTION_FAILED | %s | %s", filename, exc)
        if verbose:
            print(f"  ✗ Text extraction failed: {exc}")
        stats["reason"] = "text_extraction_failed"
        return None, stats

    # -- regex: DOI ----------------------------------------------------------
    doi = find_doi(text)
    if doi and verbose:
        print(f"  ✓ DOI found via regex: {doi}")

    # -- regex: arXiv ID -----------------------------------------------------
    arxiv_id = find_arxiv_id(text, filename)
    if arxiv_id and verbose:
        print(f"  ✓ arXiv ID found: {arxiv_id}")

    # -- route 1: arXiv -> Semantic Scholar ----------------------------------
    if not doi and arxiv_id:
        if verbose:
            print(f"  … Querying Semantic Scholar for arXiv:{arxiv_id}")
        doi = get_doi_by_arxiv(arxiv_id, email)
        if doi:
            if verbose:
                print(f"  ✓ Published DOI from Semantic Scholar: {doi}")
        elif verbose:
            print("  ⚠ No published DOI found (preprint only?)")
        time.sleep(REQUEST_DELAY)

    # -- route 2: LLM fallback -> title search ------------------------------
    if not doi:
        use_together = bool(os.environ.get("TOGETHER_API_KEY"))
        provider = "Together AI" if use_together else "Ollama"

        if verbose:
            print(f"  … No DOI yet — trying {provider} for title extraction")

        metadata = together_fallback(text) if use_together else ollama_fallback(text)

        if metadata and metadata.get("title"):
            title = metadata["title"]
            if verbose:
                print(f"  … LLM extracted title: {title}")

            # try Semantic Scholar first
            doi = get_doi_by_title(title, email)
            if doi:
                if verbose:
                    print(f"  ✓ DOI via Semantic Scholar title search: {doi}")
            else:
                # fall back to Crossref
                if verbose:
                    print("  … S2 title search missed — trying Crossref")
                doi = get_doi_by_title_crossref(title, email)
                if doi:
                    if verbose:
                        print(f"  ✓ DOI via Crossref title search: {doi}")
                elif verbose:
                    print("  ✗ Both S2 and Crossref title search came up empty")

            time.sleep(REQUEST_DELAY)
        elif verbose:
            print(f"  ✗ {provider} returned no usable metadata")

    # -- final check ---------------------------------------------------------
    if not doi:
        error_logger.warning("DOI_NOT_FOUND | %s", filename)
        if verbose:
            print(f"  ✗ Could not determine DOI for '{filename}'")
        stats["reason"] = "doi_not_found"
        return None, stats

    # -- fetch CSL-JSON and render -------------------------------------------
    try:
        csl_data = fetch_csl_json(doi, email)
        stats["found"] = True
        stats["title"] = csl_data.get("title", "Unknown Title")
        stats["link"] = f"https://doi.org/{doi}"
    except RuntimeError as exc:
        error_logger.warning(
            "CSL_JSON_FETCH_FAILED | %s | DOI=%s | %s", filename, doi, exc
        )
        if verbose:
            print(f"  ✗ CSL-JSON fetch failed: {exc}")
        stats["reason"] = "csl_fetch_failed"
        return None, stats

    # -- arXiv published version upgrade check -------------------------------
    # If Semantic Scholar couldn't link the arXiv ID directly, we are stuck with an arXiv DOI.
    # But now that we have the exact title from the arXiv CSL-JSON, we can ask Crossref!
    if "arxiv" in str(doi).lower() or "10.48550" in str(doi):
        title = csl_data.get("title")
        if title:
            if verbose:
                print(f"  … Checking if arXiv preprint '{title}' was published in a journal...")
            
            from .resolvers import get_doi_by_title_crossref
            published_doi = get_doi_by_title_crossref(title, email)
            
            # Make sure we actually got a different DOI and not just the arXiv one again
            if published_doi and "arxiv" not in published_doi.lower() and "10.48550" not in published_doi:
                if verbose:
                    print(f"  ✔️ Found published version of preprint! Upgrading DOI to: {published_doi}")
                try:
                    pub_csl = fetch_csl_json(published_doi, email)
                    doi = published_doi
                    csl_data = pub_csl
                    stats["title"] = csl_data.get("title", "Unknown Title")
                    stats["link"] = f"https://doi.org/{doi}"
                except RuntimeError as exc:
                    if verbose:
                        print(f"  ⚠ Failed to fetch CSL-JSON for published version, keeping arXiv. Error: {exc}")

    try:
        citation_html = render_citation(csl_data, csl_path)
    except Exception as exc:
        error_logger.warning(
            "CITATION_RENDER_FAILED | %s | DOI=%s | %s", filename, doi, exc
        )
        if verbose:
            print(f"  ✗ Citation render failed: {exc}")
        stats["reason"] = "render_failed"
        return None, stats

    return citation_html, stats
