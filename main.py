#!/usr/bin/env python3
"""
Backward-compatible entry point.
The actual code now lives in the pdf_to_citation package.

Usage:
    python main.py --email you@example.com
    python -m pdf_to_citation --email you@example.com
"""

from pdf_to_citation.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
