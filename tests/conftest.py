"""Shared pytest fixtures and path setup."""
import sys
from pathlib import Path

# Ensure src/ is importable when running tests without installing the package.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
