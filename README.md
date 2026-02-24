# PDF-to-citation

I couldn't find a tool that did this locally, so I made one. :))

A local Python CLI tool that turns a folder of research paper PDFs into a properly formatted APA 7th edition bibliography.

It extracts DOIs from your PDFs using regex, resolves arXiv preprints through Semantic Scholar, and fetches mathematically precise citations via doi.org. The output is an HTML file with proper italics and hanging indents that you can copy-paste directly into Microsoft Word.

## How It Works

1. Drop your PDFs into the `./papers` folder (or specify a directory)
2. Run the script
3. Open `bibliography.html` in your browser
4. Copy-paste into Word

Under the hood, the pipeline looks like this:

```
PDF -> extract text (first 3 pages)
    -> regex for DOI or arXiv ID
    -> if arXiv -> Semantic Scholar -> published DOI
    -> if nothing found -> Ollama (llama3) extracts the title -> Semantic Scholar search -> DOI
    -> doi.org CSL-JSON -> citeproc-py renders APA HTML
    -> bibliography.html
```

No LLM ever writes the citation itself. Every citation comes deterministically from doi.org + citeproc-py, so there are zero hallucinations.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on all PDFs in ./papers
python main.py --email you@example.com

# Test on a single PDF first
python main.py --email you@example.com --test -v

# Custom input/output
python main.py --email you@example.com --dir ./my_papers --output refs.html
```

The `--email` flag is required — Crossref and doi.org use it for their polite pool (you get faster responses).

## Options

| Flag | Description |
|---|---|
| `--email` | Your email for API polite pool (required) |
| `--dir` | PDF directory (default: `./papers`) |
| `--output` | Output file (default: `bibliography.html`) |
| `--test` / `--dry-run` | Process only the first PDF |
| `-v` / `--verbose` | Show detailed progress |

## LLM Fallback (Optional)

If a PDF has no DOI or arXiv ID in the text, the script tries to extract the title using a local [Ollama](https://ollama.com/) instance running `llama3`. This is optional — if Ollama isn't running, those PDFs just get logged to `errors.log`.

To set it up:

```bash
ollama pull llama3
ollama serve
```

## Output

- **`bibliography.html`** — your formatted bibliography (open in browser, copy into Word)
- **`errors.log`** — any PDFs that couldn't be processed (no DOI found, API errors, etc.)

## Requirements

- Python 3.10+
- `PyMuPDF` — PDF text extraction
- `requests` — HTTP calls
- `citeproc-py` — deterministic APA citation rendering
- [Ollama](https://ollama.com/) with `llama3` (optional, for fallback)

## License

[MIT](LICENSE)