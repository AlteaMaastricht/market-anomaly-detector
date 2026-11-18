"""Test package: makes ``src/`` importable when run with
``python -m unittest discover -s tests -t .`` from the repository root."""
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
