"""
Semantic Scholar API lookups — resolve arXiv IDs and titles to published DOIs.
Includes exponential backoff for rate-limited responses.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import requests

from .config import (
    SEMANTIC_SCHOLAR_PAPER_URL,
    SEMANTIC_SCHOLAR_SEARCH_URL,
    polite_headers,
)

# S2 rate-limits at 100 requests per 5 minutes on the free tier
_MAX_RETRIES = 10
_INITIAL_BACKOFF = 1  # seconds


def _request_with_backoff(method, url, **kwargs):
    """
    Thin wrapper around requests that retries on 429 / 5xx with
    exponential backoff. Returns the response or raises on final failure.
    """
    backoff = _INITIAL_BACKOFF
    for attempt in range(_MAX_RETRIES):
        resp = method(url, **kwargs)
        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt < _MAX_RETRIES - 1:
                print(
                    f"  ⚠ Semantic Scholar Error {resp.status_code} — "
                    f"retrying in {backoff}s..."
                )
                time.sleep(backoff)
                backoff *= 2
                continue
        return resp
    return resp  # last attempt's response


def get_doi_by_arxiv(arxiv_id: str, email: str) -> Optional[str]:
    """
    Given an arXiv ID, ask Semantic Scholar for the published DOI.
    Returns None if the paper is preprint-only or the lookup fails.
    """
    # strip version suffix (2402.10688v2 -> 2402.10688) for the API
    clean_id = re.sub(r"v\d+$", "", arxiv_id)

    url = SEMANTIC_SCHOLAR_PAPER_URL.format(arxiv_id=clean_id)
    headers = polite_headers(email)
    params = {"fields": "externalIds,title"}

    try:
        resp = _request_with_backoff(
            requests.get, url, headers=headers, params=params, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("externalIds", {}).get("DOI")
    except requests.exceptions.RequestException:
        return None


def get_doi_by_title(title: str, email: str) -> Optional[str]:
    """
    Search Semantic Scholar by title and return the top result's DOI,
    or None if nothing comes back.
    """
    headers = polite_headers(email)
    params = {"query": title, "limit": 1, "fields": "externalIds,title"}

    try:
        resp = _request_with_backoff(
            requests.get,
            SEMANTIC_SCHOLAR_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        papers = data.get("data", [])
        if papers:
            return papers[0].get("externalIds", {}).get("DOI")
    except requests.exceptions.RequestException:
        pass

    return None


def get_doi_by_title_crossref(title: str, email: str) -> Optional[str]:
    """
    Search Crossref by title as a second fallback when Semantic Scholar
    comes up empty. Returns the top result's DOI, or None.
    """
    url = "https://api.crossref.org/works"
    headers = polite_headers(email)
    params = {"query.title": title, "rows": 1, "select": "DOI,title"}

    try:
        resp = _request_with_backoff(
            requests.get, url, headers=headers, params=params, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("message", {}).get("items", [])
        if items:
            return items[0].get("DOI")
    except requests.exceptions.RequestException:
        pass

    return None
