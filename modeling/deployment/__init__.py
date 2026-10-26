"""Portable, read-only serving without installing the corpus ingestion package."""
from pathlib import Path
import sys

# The GitHub checkout retains its conventional src/ layout. Only cpss.core and
# the pure export helpers are imported by serving; no database client is needed.
_source = Path(__file__).resolve().parents[2] / "src"
if _source.is_dir() and str(_source) not in sys.path:
    sys.path.insert(0, str(_source))
