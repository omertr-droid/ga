"""Cohort listing for the BM-validation Library: the device-seeded scans to validate, each with its
reader identity (eid) and per-scan validation progress read from the layer store. Pure CSV + JSON
reads (no E2E decode) so the Library lists instantly; the worklist itself (which eyes carry device
BM on the 6x6) is precomputed offline by src/build_bm_worklist.py -> results/bm_worklist.csv.
"""
import csv
import os

import paths

from . import ids

WORKLIST = os.path.join(paths.RESULTS_DIR, "bm_worklist.csv")


def _abspath(e2e_file):
    return os.path.join(paths.DATA_DIR, *str(e2e_file).split("/"))


def _state(validated, n, missing):
    """Library row colour: green = ALL B-scans validated AND none still missing BM; orange = partial
    (some validated, or gaps remain); red = nothing done. A previously-'validated' eye that still has
    missing columns (validated before the 100%-coverage bar) reads orange — NOT green. `missing` may be
    None (not computed) -> ignored. No-device eyes are NOT greyed (seed BM with the DL model)."""
    if missing and missing > 0:
        return "orange" if validated and validated > 0 else "red"
    if not n or validated <= 0:
        return "red"
    return "green" if validated >= n else "orange"


def list_scans(layer_store=None):
    """One entry per cohort eye from results/bm_worklist.csv, device-first then by subject/eye.
    Each entry carries eid + abspath (the row click opens that E2E) and validation progress."""
    if not os.path.exists(WORKLIST):
        return {"ok": False, "scans": [],
                "error": "results/bm_worklist.csv missing — run src/build_bm_worklist.py"}
    out = []
    with open(WORKLIST, newline="") as f:
        for r in csv.DictReader(f):
            eye = (r.get("eye") or "").strip().upper()
            e2e_file = (r.get("e2e_file") or "").strip()
            abspath = _abspath(e2e_file)
            eid = ids.e2e_id(abspath)
            n = int(r.get("n_bscans") or 0)
            has_dev = str(r.get("has_device_bm")).strip().lower() in ("true", "1", "yes")
            validated = len(layer_store.bm_validated(eid, eye)) if layer_store is not None else 0
            missing = layer_store.get_missing_count(eid, eye) if layer_store is not None else None
            if missing is None:                       # not cached -> show the precomputed initial count
                init = r.get("n_missing_initial")     # (raw device gaps; whole eye when no device BM)
                missing = int(init) if str(init).strip() not in ("", "None") else None
            out.append({
                "subject": r.get("subject", ""),
                "visit": r.get("visit", ""),
                "eye": eye,
                "e2e_file": e2e_file,
                "abspath": abspath,
                "eid": eid,
                "n_bscans": n,
                "has_device_bm": has_dev,
                "validated": validated,
                "missing_bscans": missing,
                "state": _state(validated, n, missing),
                "advRPE_area_mm2": r.get("advRPE_area_mm2", ""),
            })
    out.sort(key=lambda s: (not s["has_device_bm"], s["subject"], s["eye"]))
    return {"ok": True, "scans": out}
