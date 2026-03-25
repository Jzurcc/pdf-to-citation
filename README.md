# Auto APA Generator from PDFs

I couldn't find a tool that did this locally, so I made one.

A Python CLI tool that turns a folder of research paper PDFs into a properly formatted APA 7th edition bibliography.

Unlike Zotero, this tool uses the official Crossref citation and Semantic Scholar APIs to automatically convert arXiv preprint references to their published versions (if they ever have one). This means lazy people (a.k.a me) who work with newly-written tech papers can now quickly make updated bibliographies in a single click. 

It extracts DOIs from your PDFs using regex, resolves arXiv preprints through Semantic Scholar, and fetches mathematically precise citations via doi.org. The output is an HTML file with proper italics and hanging indents that you can copy-paste directly into Microsoft Word. No LLM ever writes the citation itself — every citation comes deterministically from doi.org + citeproc-py, so there are zero hallucinations.

## How It Works

1. Drop your PDFs into the `./papers` folder (or specify a directory)
2. Double-click `run.bat` (or run `python main.py` manually)
3. Open `bibliography.html` in your browser
4. Copy-paste into Word

Under the hood:

```
PDF -> extract text (first 3 pages)
    -> regex for DOI or arXiv ID
    -> if arXiv -> Semantic Scholar -> published DOI
    -> if nothing found -> LLM extracts the title
       -> Semantic Scholar title search -> DOI
       -> Crossref title search (fallback) -> DOI
    -> doi.org CSL-JSON -> citeproc-py renders APA HTML
    -> bibliography.html
```

## Quick Start

The easiest way is to just double-click **`run.bat`**. It handles everything automatically: checks dependencies, sets up your API key and email, and runs the pipeline.

If you prefer the CLI:

```bash
# Install dependencies
pip install -r requirements.txt

# Run on all PDFs in ./papers
python main.py --email you@example.com

# Or use as a Python module
python -m pdf_to_citation --email you@example.com

# Test on a single PDF first
python main.py --email you@example.com --test -v

# Custom input/output
python main.py --email you@example.com --dir ./my_papers --output refs.html
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

If a PDF has no DOI or arXiv ID in the text, the script tries to extract the title using an LLM, then searches Semantic Scholar + Crossref by title.

**Together AI (recommended, default)** — set your API key in `.env`:

```
TOGETHER_API_KEY=your_key_here
```

`run.bat` will prompt you for this on first run and save it automatically.

**Local Ollama (alternative)** — if no Together AI key is set, falls back to a local [Ollama](https://ollama.com/) instance running `llama3.2:3b`:

```bash
ollama pull llama3.2:3b
ollama serve
```

If neither is available, those PDFs just get logged to `errors.log`.

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
- `tqdm` — progress bar (optional, auto-detected)
- [Ollama](https://ollama.com/) with `llama3.2:3b` (optional, for local LLM fallback)

## License

[MIT](LICENSE)
