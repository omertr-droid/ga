"""clinic.core — pure domain logic (no web dependencies).

Importing this package imports ``reader.core`` for its side effect: ``reader/core/__init__.py`` puts
the repo's ``src/`` directory on ``sys.path``, so the bare-named pipeline modules (``bm``, ``bm_dl``,
``m2_bm``, ``m3_projections``, ``qcviz`` …) — and ``paths`` — import everywhere below without each
module re-deriving the path. Every clinic core module can therefore simply ``import reader.core`` /
``from reader.core import …`` / ``import bm_dl`` and trust the path is set up.
"""
import reader.core  # noqa: F401  (side effect: adds repo src/ to sys.path)
