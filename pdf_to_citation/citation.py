"""
Everything related to turning a DOI into a formatted APA citation:
downloading the CSL style file, fetching CSL-JSON from doi.org,
cleaning up citeproc-py's output quirks, and rendering the final HTML.
"""

from __future__ import annotations

import re
from pathlib import Path

import requests

from .config import (
    APA_CSL_FILE,
    APA_CSL_URL,
    CSL_JSON_HEADERS,
    DOI_URL,
    polite_headers,
)


def ensure_apa_csl(script_dir: Path) -> Path:
    """
    Make sure apa.csl exists next to the script. Downloads it from
    the official CSL GitHub repo if missing.
    """
    csl_path = script_dir / APA_CSL_FILE
    if csl_path.is_file():
        return csl_path

    print(f"  … Downloading {APA_CSL_FILE} from GitHub …")
    try:
        resp = requests.get(APA_CSL_URL, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Failed to download {APA_CSL_FILE}: {exc}") from exc

    csl_path.write_text(resp.text, encoding="utf-8")
    print(f"  ✓ Saved {APA_CSL_FILE}")
    return csl_path


def fetch_csl_json(doi: str, email: str) -> dict:
    """
    Grab the CSL-JSON metadata for a DOI from doi.org via content negotiation.
    Raises RuntimeError on HTTP or JSON parsing failures.
    """
    url = DOI_URL.format(doi=doi)
    headers = {**CSL_JSON_HEADERS, **polite_headers(email)}

    try:
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"doi.org returned HTTP {resp.status_code} for '{doi}': {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Network error fetching CSL-JSON for '{doi}': {exc}"
        ) from exc

    try:
        return resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"doi.org returned invalid JSON for '{doi}': {exc}"
        ) from exc


def _clean_citation_html(html: str) -> str:
    """
    Fix a few known citeproc-py rendering quirks:
    - stray "(as <i>[]</i>)" artifacts
    - missing space before & in author lists
    - double periods
    - DOI URLs glued to preceding text
    - multiple consecutive spaces
    """
    html = re.sub(r"\s*\(as\s*<i>\[.*?\]</i>\)", "", html)
    html = re.sub(r"(\w\.)\s*(&amp;)", r"\1, \2", html)
    html = html.replace("..", ".")
    html = re.sub(r"([.)])(\s*)(https?://)", r"\1 \3", html)
    
    # Strip "In " from journal names
    html = html.replace("In <i>", "<i>")
    
    # Fix the non-standard journal format (Vol. X, Number Y, pp. Z) or (X, Y, Z)
    def _fix_journal(match):
        journal = match.group(1)
        meta = match.group(2).strip()
        
        vol = None
        num = None
        pages = None
        
        vol_m = re.search(r"Vol\.\s*([a-zA-Z0-9]+)", meta, re.I)
        num_m = re.search(r"(?:Number|No\.)\s*([a-zA-Z0-9\-]+)", meta, re.I)
        pages_m = re.search(r"pp?\.\s*(.+)", meta, re.I)
        
        if vol_m or num_m or pages_m:
            if vol_m: vol = vol_m.group(1)
            if num_m: num = num_m.group(1)
            if pages_m: pages = pages_m.group(1).strip()
            
            # Catch "3, 1, p. 100184" where vol and num are bare numbers
            if not vol and not num and pages_m:
                remainder = meta[:pages_m.start()].strip().strip(',')
                bare_match = re.match(r"^([a-zA-Z0-9]+)(?:,\s*([a-zA-Z0-9\-]+))?$", remainder)
                if bare_match:
                    vol = bare_match.group(1)
                    num = bare_match.group(2)
        else:
            # Catch fully bare "37, 33, 28191-28267" or "37, 33" or "37"
            bare_match = re.match(r"^([a-zA-Z0-9]+)(?:,\s*([a-zA-Z0-9\-]+))?(?:,\s*(.+))?$", meta)
            if bare_match:
                vol = bare_match.group(1)
                num = bare_match.group(2)
                pages = bare_match.group(3)
            else:
                return match.group(0)

        res = f"<i>{journal}"
        if vol:
            res += f", {vol}"
        res += "</i>"
            
        if num:
            res += f"({num})"
            
        if pages:
            res += f", {pages}"
            
        return res

    html = re.sub(r"<i>(.*?)</i>\s*\((.*?)\)", _fix_journal, html)

    html = re.sub(r"  +", " ", html)
    return html.strip()


def render_citation(csl_json_data: dict, csl_path: Path) -> str:
    """
    Take a CSL-JSON record and render it as an APA 7th-edition HTML string
    using citeproc-py. Applies post-processing cleanup.
    """
    from citeproc import (
        CitationStylesBibliography,
        CitationStylesStyle,
        Citation,
        CitationItem,
    )
    from citeproc.source.json import CiteProcJSON
    from citeproc import formatter

    if "id" not in csl_json_data:
        csl_json_data["id"] = csl_json_data.get("DOI", "item1")

    # Clean up redundant metadata before handing it off to citeproc-py
    # If `event` or `collection-title` repeats the `container-title`, drop them
    # to avoid printing a non-italicized duplicate string!
    ct = csl_json_data.get("container-title", "")
    if isinstance(ct, str) and ct.strip():
        ct_words = set(re.findall(r"\w+", ct.lower()))
        if ct_words:
            for field in ["event", "collection-title"]:
                val = csl_json_data.get(field)
                if isinstance(val, str):
                    val_words = set(re.findall(r"\w+", val.lower()))
                    if val_words:
                        overlap = len(val_words.intersection(ct_words)) / len(val_words)
                        if overlap >= 0.5:
                            csl_json_data.pop(field, None)

    source = CiteProcJSON([csl_json_data])
    style = CitationStylesStyle(str(csl_path))
    bib = CitationStylesBibliography(style, source, formatter.html)

    citation = Citation([CitationItem(csl_json_data["id"])])
    bib.register(citation)

    entries = bib.bibliography()
    if entries:
        return _clean_citation_html(str(entries[0]))

    raise RuntimeError("citeproc-py produced an empty bibliography entry.")
