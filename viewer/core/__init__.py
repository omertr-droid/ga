"""viewer.core — pure domain logic for the doctor viewer (no web deps).

Importing this package imports `reader.core`, which puts `<repo>/src` on `sys.path`, so the pipeline
modules (`m3_projections`, `qcviz`, `register_qc`, …) and `reader.core.*` are importable from here.
"""
import reader.core as _rc   # noqa: F401  (side effect: adds <repo>/src to sys.path)

REPO_ROOT = _rc.REPO_ROOT
