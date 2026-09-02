"""Precompute the BM-validation worklist: which cohort eyes carry a DEVICE BM contour on their
native 6x6 (97-line) volume — the device-seeded scans the reader's Library lists for BM validation.

Reads results/spectralis_ga_pairing.csv (qc_status==ok), opens each E2E once (one file holds both
eyes), finds the 6x6 volume per eye, and checks device BM cheaply from the contour metadata (no
pixel load). Writes results/bm_worklist.csv and prints the device-BM count on the 6x6 — the number
behind the 6x6-vs-30deg scope decision.

Run from repo root:  oct_env\\Scripts\\python.exe src\\build_bm_worklist.py
"""
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

import paths                                  # noqa: E402
from reader.core import e2e_source            # noqa: E402

PAIRING = os.path.join(paths.RESULTS_DIR, "spectralis_ga_pairing.csv")
OUT = os.path.join(paths.RESULTS_DIR, "bm_worklist.csv")
COLS = ["subject", "visit", "eye", "e2e_file", "n_bscans", "has_device_bm",
        "n_missing_initial", "advRPE_area_mm2"]


def _six_ref(raw, eye):
    """The 6x6 VolumeRef for an eye (most B-scans), or None."""
    six = [r for r in raw.refs if r.eye == eye and r.is_6x6]
    return max(six, key=lambda r: r.n_bscans) if six else None


def main():
    with open(PAIRING, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("qc_status") or "").strip() == "ok"]

    opened = {}                                # abspath -> RawE2E (cache: both eyes share a file)
    out = []
    n_dev = n_six = 0
    for r in rows:
        eye = r["eye"].strip().upper()
        rel = r["e2e_file"].strip()
        abspath = os.path.join(paths.DATA_DIR, *rel.split("/"))
        rec = {"subject": r["subject"], "visit": r["visit"], "eye": eye, "e2e_file": rel,
               "n_bscans": 0, "has_device_bm": False, "n_missing_initial": 0,
               "advRPE_area_mm2": r.get("advRPE_area_mm2", "")}
        try:
            raw = opened.get(abspath)
            if raw is None:
                raw = e2e_source.open_e2e(abspath)
                opened[abspath] = raw
            ref = _six_ref(raw, eye)
            if ref is not None:
                n_six += 1
                rec["n_bscans"] = ref.n_bscans
                v = raw.vols[ref.index]
                _, bm_raw = e2e_source._device_layers(v, (ref.n_bscans, ref.W))
                rec["has_device_bm"] = bm_raw is not None
                if rec["has_device_bm"]:
                    n_dev += 1
                    # initial Library "Missing" count: B-scans whose DEVICE BM has a gap (any column with no
                    # finite value). Cheap (contour only, no pixels); the exact count refines on open. Device
                    # eyes typically fill the saturated band, so band columns are not gaps here.
                    bm = np.asarray(bm_raw, float)
                    has = np.isfinite(bm) & (bm > 0)
                    rec["n_missing_initial"] = int((~has.all(axis=1)).sum())
                else:
                    rec["n_missing_initial"] = ref.n_bscans   # no device seed -> whole eye starts missing
        except Exception as e:                 # noqa: BLE001 — record + continue per row
            rec["error"] = repr(e)
            print(f"  !! {r['subject']} {eye}: {e}")
        out.append(rec)
        flag = "device" if rec["has_device_bm"] else ("self" if rec["n_bscans"] else "NO-6x6")
        print(f"  {r['subject']:>18} {eye}  6x6 n={rec['n_bscans']:>3}  {flag}")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)

    print(f"\n{len(rows)} qc=ok eyes · {n_six} have a 6x6 · {n_dev} have DEVICE BM on the 6x6 "
          f"({n_six - n_dev} 6x6 self-seg, {len(rows) - n_six} no 6x6).")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
