"""
CLI entry point and main processing loop.

Run with:
    python -m pdf_to_citation --email you@example.com
    python main.py --email you@example.com
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from .config import HTML_TEMPLATE, REQUEST_DELAY
from .citation import ensure_apa_csl
from .pipeline import process_pdf


def _setup_error_logger(output_dir: Path) -> logging.Logger:
    """Set up an error logger that writes to errors.log."""
    logger = logging.getLogger("pdf_citation.errors")
    logger.setLevel(logging.WARNING)

    if not logger.handlers:
        fh = logging.FileHandler(
            output_dir / "errors.log", mode="w", encoding="utf-8"
        )
        fh.setLevel(logging.WARNING)
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(fh)

    return logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Turn a folder of research paper PDFs into a formatted "
            "APA 7th edition bibliography."
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
        help="Output file (default: bibliography.html)",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help=(
            "Your email for API polite-pool headers. "
            "Falls back to EMAIL in .env if not provided."
        ),
    )
    parser.add_argument(
        "--test", "--dry-run",
        action="store_true",
        dest="test",
        help="Process only the first PDF (quick sanity check).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore processed.log and re-process all PDFs from scratch.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress in the terminal.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    papers_dir: Path = args.dir.resolve()
    output_file: Path = args.output.resolve()
    test_mode: bool = args.test
    verbose: bool = args.verbose
    fresh: bool = args.fresh

    # resolve email: CLI flag > .env > error
    email = args.email or os.environ.get("EMAIL")
    if not email:
        print(
            "Error: --email is required (or set EMAIL in your .env file).",
            file=sys.stderr,
        )
        return 1

    # validate input directory
    if not papers_dir.is_dir():
        print(f"Error: directory '{papers_dir}' does not exist.", file=sys.stderr)
        return 1

    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No .pdf files found in '{papers_dir}'.", file=sys.stderr)
        return 1

    if test_mode:
        pdf_files = pdf_files[:1]
        print(f"[TEST MODE] Processing only: {pdf_files[0].name}\n")

    # setup
    error_logger = _setup_error_logger(output_file.parent)
    script_dir = Path(__file__).resolve().parent.parent  # project root

    try:
        csl_path = ensure_apa_csl(script_dir)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # optional progress bar (disabled when verbose — they fight over the terminal)
    progress = None
    if not verbose:
        try:
            from tqdm import tqdm
            progress = tqdm(pdf_files, desc="Processing PDFs", unit="pdf")
        except ImportError:
            pass

    citations: list[str] = []
    seen_dois: set[str] = set()  # dedup
    total = len(pdf_files)
    processed_log_path = output_file.parent / "processed.log"

    # resume support — read previously succeeded PDFs from processed.log
    already_done: set[str] = set()
    if not fresh and processed_log_path.is_file():
        with open(processed_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if "is found? yes" in line:
                    name = line.split(" - ")[0].strip()
                    already_done.add(name)
        if already_done:
            print(
                f"Resuming — skipping {len(already_done)} already-processed "
                f"PDF(s). Use --fresh to re-process all.\n"
            )

    print(f"Processing {total} PDF(s) from '{papers_dir}' …\n")

    # append mode when resuming so we don't lose old entries
    log_mode = "w" if fresh or not already_done else "a"

    # statistics
    num_success = 0
    num_duplicates = 0
    num_skipped = 0
    num_errors = 0

    with open(processed_log_path, log_mode, encoding="utf-8") as plog:
        iterable = progress if progress is not None else pdf_files
        for idx, pdf_path in enumerate(iterable, start=1):

            # skip if already done from a previous run
            if pdf_path.name in already_done:
                if verbose:
                    print(f"[{idx}/{total}] {pdf_path.name} — skipped (already processed)")
                num_skipped += 1
                continue

            if progress is None:
                print(f"[{idx}/{total}] {pdf_path.name}")

            citation, stats = process_pdf(
                pdf_path, email, csl_path, error_logger, verbose=verbose
            )

            # write processed.log line with failure reason if applicable
            found_str = "yes" if stats["found"] else "no"
            log_line = f"{pdf_path.name} - is found? {found_str}"
            if stats["found"]:
                log_line += f" - {stats['title']}, {stats['link']}"
            elif stats.get("reason"):
                log_line += f" - reason: {stats['reason']}"
            plog.write(log_line + "\n")
            plog.flush()

            # dedup: skip if we already have a citation for this DOI
            if citation and stats.get("link"):
                doi_key = stats["link"]
                if doi_key in seen_dois:
                    if verbose:
                        print(f"  ⚠ Duplicate DOI, skipping: {doi_key}")
                    citation = None
                    num_duplicates += 1
                else:
                    seen_dois.add(doi_key)

            if citation:
                citations.append(citation)
                num_success += 1
                if verbose:
                    preview = re.sub(r"<[^>]+>", "", citation)[:120]
                    ellipsis = "…" if len(preview) >= 120 else ""
                    print(f"  -> {preview}{ellipsis}")
            elif not stats["found"]:
                num_errors += 1

            if progress is None and idx < total:
                time.sleep(REQUEST_DELAY)

            if progress is None:
                print()

    # write HTML output
    if citations:
        citations.sort(key=lambda c: re.sub(r"<[^>]+>", "", c).casefold())

        entries = "\n".join(
            f'  <p class="citation">{c}</p>' for c in citations
        )
        html = HTML_TEMPLATE.format(entries=entries)

        with open(output_file, "w", encoding="utf-8") as fh:
            fh.write(html)

        print(f"✓ {len(citations)} citation(s) written to '{output_file}'.")
    else:
        print("⚠ No new citations were generated in this run.")

    if num_duplicates > 0:
        print(f"ℹ {num_duplicates} duplicate PDF(s) ignored.")
    
    if num_skipped > 0:
        print(f"ℹ {num_skipped} previously processed PDF(s) skipped (use --fresh to rebuild full HTML).")

    if num_errors > 0:
        log_path = output_file.parent / "errors.log"
        print(f"✗ {num_errors} PDF(s) failed — see '{log_path}' for details.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
