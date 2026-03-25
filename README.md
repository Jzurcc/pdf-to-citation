# Auto APA Generator from PDFs

I couldn't find a tool that did this locally, so I made one.

A Python CLI tool that turns a folder of research paper PDFs into a properly formatted APA 7th edition bibliography.

Unlike Zotero or Citation Machine, this tool talks directly to Crossref and Semantic Scholar to resolve citations, and automatically upgrades arXiv preprints to their formally published journal versions when they exist. This means you can point it at a folder full of freshly downloaded papers and get a clean, accurate bibliography in one click.

Every citation is rendered deterministically from doi.org metadata via citeproc-py — no LLM ever writes the citation itself, so there are zero hallucinations in the output.

## How It Works

1. Drop your PDFs into the `./papers` folder (or specify a directory)
2. Double-click `run.bat`
3. Open `bibliography.html` in your browser and copy-paste into Word

Under the hood:

```
PDF -> extract text (first 3 pages)
    -> regex for DOI or arXiv ID
    -> if arXiv ID -> Semantic Scholar -> published DOI
    -> if still no DOI -> Together AI extracts the title
       -> Semantic Scholar title search -> DOI
       -> Crossref title search (fallback) -> DOI
    -> fetch CSL-JSON from doi.org
    -> if DOI is arXiv -> Crossref title search for published version
       -> upgrade to published DOI if found
    -> citeproc-py renders APA 7th edition HTML
    -> bibliography.html
```

## Quick Start

The easiest way is to just double-click **`run.bat`**. It handles everything automatically: checks dependencies, sets up your API key and email, and runs the pipeline.

If you prefer the CLI:

```bash
# Install dependencies
pip install -r requirements.txt

# Run on all PDFs in ./papers
python -m pdf_to_citation --email you@example.com

# Test on a single PDF first
python -m pdf_to_citation --email you@example.com --test -v

# Custom input/output
python -m pdf_to_citation --email you@example.com --dir ./my_papers --output refs.html
```

The `--email` flag is required unless you set `EMAIL` in your `.env` file — Crossref and doi.org use it for their polite pool (you get faster responses).

## Options

| Flag | Description |
|---|---|
| `--email` | Your email for API polite pool (or set `EMAIL` in `.env`) |
| `--dir` | PDF directory (default: `./papers`) |
| `--output` | Output file (default: `bibliography.html`) |
| `--test` / `--dry-run` | Process only the first PDF |
| `--fresh` | Ignore `processed.log` and re-process everything |
| `-v` / `--verbose` | Show detailed progress |

## LLM Fallback

If a PDF has no DOI or arXiv ID in its text, the script uses an LLM to extract the paper's title, then searches by that title on Semantic Scholar and Crossref.

**Together AI (default)** — uses `Llama-3.3-70B-Instruct-Turbo` via the Together AI API. Get a free key at [together.ai](https://together.ai) and set it in your `.env` file:

```
TOGETHER_API_KEY=your_key_here
```

`run.bat` will prompt you for this on first run and save it automatically.

**Local Ollama (fallback)** — if no Together AI key is set, the tool falls back to a local [Ollama](https://ollama.com/) instance running `llama3.2:3b`:

```bash
ollama pull llama3.2:3b
ollama serve
```

If neither is configured, PDFs without a detectable DOI are logged to `errors.log` and skipped.

## Resume Support

If the script is interrupted, just re-run it — it reads `processed.log` and skips PDFs that already succeeded. Use `--fresh` to force re-processing everything from scratch.

## Output

- **`bibliography.html`** — your formatted bibliography (open in browser, copy into Word)
- **`processed.log`** — per-PDF status: found/not found, title, DOI link, or failure reason
- **`errors.log`** — detailed error info for PDFs that couldn't be processed

## Configuration (`.env`)

All user settings live in a single `.env` file:

```
TOGETHER_API_KEY=your_key_here
EMAIL=you@example.com
```

Both are optional if you pass `--email` on the command line and don't need Together AI.

## Requirements

- Python 3.10+
- `PyMuPDF` — PDF text extraction
- `requests` — HTTP calls
- `citeproc-py` — deterministic APA citation rendering
- `python-dotenv` — `.env` file loading
- `tqdm` — progress bar
- A free [Together AI](https://together.ai) API key (recommended) or a local [Ollama](https://ollama.com/) instance as fallback

## License

[MIT](LICENSE)
