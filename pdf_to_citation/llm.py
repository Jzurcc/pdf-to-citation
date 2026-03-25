"""
LLM-based title extraction fallbacks.

When regex can't find a DOI or arXiv ID in the PDF text, we send the first
chunk of text to an LLM and ask it to pull out the paper title + authors.
Supports Together AI (cloud, default) and Ollama (local).
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import requests

from .config import (
    LLM_MAX_CHARS,
    LLM_PROMPT_TEMPLATE,
    OLLAMA_URL,
    OLLAMA_MODEL,
    TOGETHER_API_URL,
    TOGETHER_MODEL,
)


def _build_prompt(text: str) -> str:
    """Format the shared prompt template with the (truncated) PDF text."""
    return LLM_PROMPT_TEMPLATE.format(text=text[:LLM_MAX_CHARS])


def ollama_fallback(text: str) -> Optional[dict]:
    """
    Try extracting title/authors via a local Ollama server.
    Retries up to 3 times with backoff for transient failures.
    Returns {"title": "...", "authors": "..."} or None.
    """
    prompt = _build_prompt(text)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    max_attempts = 3
    backoff = 1

    for attempt in range(max_attempts):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=60)

            if resp.status_code in (500, 502, 503, 504):
                if attempt < max_attempts - 1:
                    print(
                        f"  ⚠ Ollama {resp.status_code} — "
                        f"retrying in {backoff}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            data = json.loads(raw)
            if "title" in data:
                return data

        except requests.exceptions.ConnectionError:
            if attempt < max_attempts - 1:
                print(
                    f"  ⚠ Ollama not reachable — retrying in {backoff}s..."
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            print(f"  ⚠ Ollama not reachable at {OLLAMA_URL} — giving up")
            break

        except requests.exceptions.RequestException as exc:
            print(f"  ⚠ Ollama request failed: {exc}")
            break

        except (json.JSONDecodeError, KeyError):
            print("  ⚠ Ollama returned bad JSON — skipping")
            break

    return None


def together_fallback(text: str) -> Optional[dict]:
    """
    Try extracting title/authors via Together AI's API.
    Uses exponential backoff (1s, 2s, 4s, ...) on rate limits, up to 10 retries.
    Returns {"title": "...", "authors": "..."} or None.
    """
    api_key = os.environ.get("TOGETHER_API_KEY", "").strip()
    if not api_key:
        return None

    prompt = _build_prompt(text)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": TOGETHER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }

    max_attempts = 10
    backoff = 1

    for attempt in range(max_attempts):
        try:
            resp = requests.post(
                TOGETHER_API_URL, headers=headers, json=payload, timeout=60
            )

            # retry on rate limit or server hiccup
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < max_attempts - 1:
                    print(
                        f"  ⚠ Together AI {resp.status_code} — "
                        f"retrying in {backoff}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

            resp.raise_for_status()
            raw = resp.json()
            content = raw["choices"][0]["message"]["content"].strip()
            data = json.loads(content)
            if "title" in data:
                return data

        except requests.exceptions.HTTPError as exc:
            # log the response body so we can see what Together AI is complaining about
            try:
                body = resp.text[:300]
            except Exception:
                body = "(no body)"
            print(f"  ⚠ Together AI HTTP {resp.status_code}: {body}")
            break

        except requests.exceptions.RequestException as exc:
            if attempt < max_attempts - 1:
                print(
                    f"  ⚠ Together AI error ({exc}) — retrying in {backoff}s..."
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            print(f"  ⚠ Together AI gave up after {max_attempts} attempts: {exc}")
            break

        except (json.JSONDecodeError, KeyError, IndexError):
            print("  ⚠ Together AI returned bad JSON — skipping")
            break

    return None
