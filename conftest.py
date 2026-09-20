"""Ensures the repo root is on sys.path so `from src...` and
`from scripts...` imports resolve during test collection, without
requiring the package to be installed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
