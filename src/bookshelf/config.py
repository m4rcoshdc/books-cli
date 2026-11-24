"""Paths. Override with BOOKS_LIBRARY / BOOKS_DATA environment variables."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Calibre library (read-only input).
LIBRARY = Path(os.environ.get("BOOKS_LIBRARY", PROJECT_ROOT / "books"))

# Hand-curated metadata fixes (tracked in git; the library is a static copy).
OVERRIDES_FILE = PROJECT_ROOT / "metadata_overrides.json"

# What the user has read (personal state, not derived: outside data/, gitignored).
READING_FILE = PROJECT_ROOT / "reading.json"

# Everything we derive lives here and can be rebuilt from LIBRARY.
DATA = Path(os.environ.get("BOOKS_DATA", PROJECT_ROOT / "data"))
CATALOG_FILE = DATA / "catalog.json"
BOOKS_DIR = DATA / "books"
CACHE_DIR = DATA / "cache"
