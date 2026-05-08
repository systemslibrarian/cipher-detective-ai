"""pytest configuration for Cipher Detective AI tests.

Ensures the repo root is on sys.path so `import core` works regardless of how
pytest is invoked (e.g. `pytest`, `python -m pytest`, or via pyproject.toml
``pythonpath`` setting).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
