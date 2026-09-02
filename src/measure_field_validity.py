#!/usr/bin/env python
"""Per-eye saturated-edge-band (machine-fill) QC -> results/field_validity.csv.

A regenerable SIDECAR (the mask is deterministic from the volume, so nothing else needs persisting).
For each qc_status==ok cohort eye it opens the E2E, loads the 6x6 volume, and rolls up the
field-validity mask (reader/core/fieldmask) into the band extent. The band is MASKED everywhere in the
pipeline (so the GA area is a documented lower bound when it overlaps a lesion) -- it never auto-excludes
an eye. apply_qc / a reader badge can left-join this for triage.

Run from the repo root:
  oct_env\\Scripts\\python.exe src\\measure_field_validity.py            # all ok eyes
  oct_env\\Scripts\\python.exe src\\measure_field_validity.py --only 008  # substring filter on subject/eye
"""
import argparse
import csv
import os
import sys

from paths import DATA_DIR, RESULTS_DIR, REPO_ROOT

sys.path.insert(0, REPO_ROOT)                          # so `reader.core` resolves from a src/ script
from reader.core import e2e_source, fieldmask          # noqa: E402

PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
OUT = os.path.join(RESULTS_DIR, "field_validity.csv")
FIELDS = ["subject", "visit", "eye", "n_bscans_with_band", "frac_bscans_with_band",
          "max_band_width_px", "max_band_width_mm", "total_invalid_frac", "edge"]


def _edge_summary(inv, fov_mm):
    """Aggregate which frame edge(s) carry the band across the eye ('L'/'R'/'LR'/'')."""
    edges = set()
    for rec in fieldmask.summarize(inv, fov_mm):
        for ch in rec["edge"]:
            edges.add(ch)
    return "".join(sorted(edges))


def measure_eye(row):
    e2e_path = os.path.join(DATA_DIR, row["e2e_file"])
    if not os.path.exists(e2e_path):
        print(f"  SKIP {row['subject']} {row['eye']}: E2E missing ({e2e_path})")
        return None
    raw = e2e_source.open_e2e(e2e_path)
    idx = e2e_source.default_volume_index(raw, row["eye"])
    ov = e2e_source.load_volume(raw, idx)
    inv = ov.field_invalid
    m = fieldmask.eye_metrics(inv, ov.fov_mm)
    m.update({"subject": row["subject"], "visit": row.get("visit", ""), "eye": row["eye"],
              "edge": _edge_summary(inv, ov.fov_mm)})
    flag = "" if m["n_bscans_with_band"] == 0 else \
        f"  <- band on {m['n_bscans_with_band']} B-scans, max {m['max_band_width_mm']}mm ({m['edge']})"
    print(f"  {row['subject']} {row['visit']} {row['eye']}: "
          f"invalid_frac={m['total_invalid_frac']}{flag}")
    return {k: m.get(k) for k in FIELDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter on subject/eye (e.g. 008)")
    args = ap.parse_args()

    with open(PAIRING, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("qc_status") or "").strip() == "ok"]

    out_rows = []
    for r in rows:
        tag = f"{r['subject']} {r['eye']}"
        if args.only and args.only.lower() not in tag.lower():
            continue
        try:
            rec = measure_eye(r)
        except Exception as e:                                     # noqa: BLE001
            print(f"  FAILED {tag}: {e!r}")
            continue
        if rec is not None:
            out_rows.append(rec)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    banded = sum(1 for r in out_rows if r["n_bscans_with_band"])
    print(f"\nwrote {len(out_rows)} eyes ({banded} with a saturated band) -> {OUT}")


if __name__ == "__main__":
    main()
