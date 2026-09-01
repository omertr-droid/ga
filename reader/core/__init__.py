"""reader.core — pure domain logic (no web deps).

Importing this package makes the existing pipeline modules under <repo>/src importable by their
bare names (``import m2_bm``, ``import m3_projections``, ``import bm``, ``import qcviz`` …), exactly
as the pipeline scripts expect (they import each other as siblings on ``sys.path``). We add ``src``
to ``sys.path`` here so every core module can reuse that code without copying it.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

REPO_ROOT = _REPO_ROOT
SRC_DIR = _SRC
