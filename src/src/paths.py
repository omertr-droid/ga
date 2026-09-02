"""Central path anchor for the oct_ga repo.

Every script resolves data, outputs, results, and the cohort relative to the repo
root via the constants here, instead of each file computing its own
``os.path.dirname(__file__)``. The repo root is derived from *this* file's location
(``src/paths.py`` -> parent of ``src/``), so it resolves correctly no matter which
script imports it, including scripts under ``src/exploration/``.

Scripts are run from the repo root, e.g.::

    oct_env\\Scripts\\python.exe src\\m2_bm.py

so ``sys.path[0]`` is ``src/`` and ``import paths`` (and ``import bm`` etc.) resolve
as siblings with no per-file ``sys.path`` juggling.

Layout::

    <repo>/
      src/            code (this package) + exploration/ archive
      data/           raw E2E / PLEX scans + source zips
      outputs/        regenerable figures, per-stage dumps, feature cache
      results/        built CSV indexes + metrics
      cohort/         curated per-eye working dataset (stays at repo root)
      cohort_masks/   validated GA label masks (referenced by the pairing CSV)
"""
import os

# repo root = parent directory of src/ (this file lives at <repo>/src/paths.py)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
COHORT_DIR = os.path.join(REPO_ROOT, "cohort")
MASKS_DIR = os.path.join(REPO_ROOT, "cohort_masks")
