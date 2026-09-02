#!/usr/bin/env python
"""Build the Spectralis<->GA pairing manifest.

Joins the per-eye advRPE GA reference table (ga_cohort_manifest.csv) to the extracted
Spectralis E2E folders and the recovered GA masks (cohort_masks/), keyed on
NHAMD-003-<p>-V<v> + eye. No image processing -- just a keyed join that writes down,
explicitly, which E2E pairs with which GA annotation so the validation is runnable.

Run: oct_env\\Scripts\\python.exe build_pairing.py
Output: spectralis_ga_pairing.csv
"""
import csv
import glob
import os

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
MANIFEST = os.path.join(RESULTS_DIR, "ga_cohort_manifest.csv")
MASKS_DIR = os.path.join(ROOT, "cohort_masks")
OUT = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")

COLUMNS = [
    "subject", "visit", "eye", "date",
    "e2e_file", "ga_mask",
    "advRPE_area_mm2", "mask_area_mm2",
    "e2e_exists", "ga_mask_exists",
]


def find_e2e(subject):
    """Return path (relative to ROOT) of the single .E2E in the subject's SPECTRALIS folder, or ''."""
    folder = os.path.join(DATA_DIR, f"{subject}-SPECTRALIS")
    hits = sorted(glob.glob(os.path.join(folder, "*.E2E")) + glob.glob(os.path.join(folder, "*.e2e")))
    if not hits:
        return ""
    return os.path.relpath(hits[0], DATA_DIR).replace("\\", "/")


def main():
    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    flagged = []
    for r in rows:
        subject, eye = r["subject"], r["eye"]
        e2e = find_e2e(subject)
        mask_rel = f"cohort_masks/{subject}_{eye}_GAmask.png"
        mask_abs = os.path.join(ROOT, mask_rel)
        e2e_exists = bool(e2e)
        mask_exists = os.path.isfile(mask_abs)
        out = {
            "subject": subject,
            "visit": r["visit"],
            "eye": eye,
            "date": r.get("date", ""),
            "e2e_file": e2e,
            "ga_mask": mask_rel if mask_exists else "",
            "advRPE_area_mm2": r.get("advRPE_area_mm2", ""),
            "mask_area_mm2": r.get("mask_area_mm2", ""),
            "e2e_exists": e2e_exists,
            "ga_mask_exists": mask_exists,
        }
        out_rows.append(out)
        if not (e2e_exists and mask_exists):
            flagged.append(out)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out_rows)

    n = len(out_rows)
    paired = sum(1 for o in out_rows if o["e2e_exists"] and o["ga_mask_exists"])
    print(f"Wrote {OUT}")
    print(f"  rows (eyes): {n}")
    print(f"  fully paired (E2E + GA mask): {paired}/{n}")
    print(f"  distinct subjects: {len({o['subject'] for o in out_rows})}")
    if flagged:
        print(f"  FLAGGED {len(flagged)} unpaired row(s):")
        for o in flagged:
            print(f"    {o['subject']} {o['eye']}  e2e_exists={o['e2e_exists']}  ga_mask_exists={o['ga_mask_exists']}")
    else:
        print("  OK: every eye maps to both an E2E and a GA mask.")


if __name__ == "__main__":
    main()
