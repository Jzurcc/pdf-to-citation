"""
Shared constants, regex patterns, API URLs, and small helpers
used across the rest of the package.
"""

from __future__ import annotations

import os
import re
import warnings

# Silence harmless citeproc-py warnings about unknown fields
warnings.filterwarnings("ignore", category=UserWarning, module=r"citeproc\.source")
warnings.filterwarnings("ignore", category=UserWarning, module=r"citeproc\.frontend")

# Load .env if python-dotenv is installed (optional but recommended)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# -- Regex patterns ----------------------------------------------------------

DOI_REGEX = re.compile(r"10\.\d{4,9}/[^\s]+", re.IGNORECASE)

# Matches arXiv IDs like 2402.10688, 2402.10688v2, arXiv:2402.10688, etc.
ARXIV_REGEX = re.compile(
    r"(?:arXiv[:\s]*)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE
)


# -- API URLs ----------------------------------------------------------------

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


# -- LLM config -------------------------------------------------------------

LLM_MAX_CHARS = 3000  # how much extracted text we send to the LLM

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# The prompt both Ollama and Together AI share
LLM_PROMPT_TEMPLATE = (
    "You are a research-paper metadata extractor. Given the following text "
    "extracted from the first few pages of a research paper, identify the "
    "paper's title and authors.\n\n"
    "Return ONLY a valid JSON object with exactly two keys:\n"
    '  "title": "<full paper title>"\n'
    '  "authors": "<author names, comma-separated>"\n\n'
    "Do NOT add any other text, explanation, or markdown formatting.\n\n"
    "--- BEGIN EXTRACTED TEXT ---\n{text}\n--- END EXTRACTED TEXT ---"
)


# -- Misc --------------------------------------------------------------------

REQUEST_DELAY = 1.5  # polite-pool delay between API calls (seconds)

HTML_TEMPLATE = """\
<!DOCTYPE html>
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


def polite_headers(email: str) -> dict:
    """Build a User-Agent header so APIs put us in their polite pool."""
    return {"User-Agent": f"PDFtoCitation/1.0 (mailto:{email})"}
