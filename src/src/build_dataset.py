#!/usr/bin/env python
"""Assemble the light, per-eye WORKING dataset under cohort/.

Philosophy: keep heavy sources as authoritative ARCHIVES (the 300 MB E2Es stay in their
*-SPECTRALIS folders; the 7.5 GB GA zip stays zipped; full B-scans are read on demand from
the E2E by the pipeline). Here we materialize only the small, reusable artifacts you actually
look at and register, gathered together per subject-visit / eye:

  cohort/<NHAMD-003-p-Vv>/
    meta.json                       subject: patient_id, e2e path, per-eye acq/FOV/qc/area
    <OD|OS>/
      spectralis_ir.png             NIR localizer  (registration SOURCE)        <- E2E
      spectralis_baf.png            BAF en-face    (dev/QC reference only)       <- E2E
      ga_mask.png                   validated binary GA mask (the reference)     <- cohort_masks
      advrpe_subrpe_enface.png      hypertransmission substrate (reg TARGET)     <- GA zip
      advrpe_pseudocolor.png        PLEX en-face (reg TARGET)                    <- GA zip
      advrpe_ga_vals.csv            area, mmX/mmY, date, patientID               <- GA zip

Master index stays spectralis_ga_pairing.csv (paths + qc_status). Excluded eyes (007 timepoint,
013 identity) still get a folder so they can be inspected, but meta.json carries the exclusion.
Run: oct_env\\Scripts\\python.exe build_dataset.py
"""
import csv
import io
import json
import os
import re
import shutil
import zipfile

import numpy as np
from PIL import Image
from oct_converter.readers import E2E

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
COH = os.path.join(ROOT, "cohort")
MASKS = os.path.join(ROOT, "cohort_masks")
ZIP = os.path.join(DATA_DIR, "Zeiss GA Algorithm Run PLEX 6x6 only.zip")

GA_BY_NAME = {  # zip suffix (lower) -> output filename
    "subrpe_enface.png": "advrpe_subrpe_enface.png",
    "pseudocolor_enface.png": "advrpe_pseudocolor.png",
    "ga_seg_outline.png": "advrpe_ga_outline.png",
}
# GA_vals CSV is matched BY CONTENT (header has 'Total_GA'), since some subjects store files under
# truncated 8.3 names (168202~3.CSV) where the suffix is lost.
GA_OUTS = list(GA_BY_NAME.values()) + ["advrpe_ga_vals.csv"]


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def norm8(a):
    a = np.asarray(a, float)
    lo, hi = np.nanpercentile(a, 1), np.nanpercentile(a, 99)
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1).__mul__(255).astype(np.uint8)


def eye_of(lat):
    return "OD" if str(lat).strip().upper() in ("R", "OD", "RIGHT") else "OS"


def main():
    pair = {(r["subject"], r["eye"]): r for r in load(os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv"))}
    val = {(r["subject"], r["eye"]): r for r in load(os.path.join(RESULTS_DIR, "spectralis_validation.csv"))}
    subjects = sorted({s for s, _ in pair})

    # index the needed GA-zip entries in one pass (no full extraction of the 7.5 GB archive).
    # eye is taken from the _6x6_(OD|OS)_ directory in the full path, so mangled 8.3 names still map.
    ga_entry = {}  # (subject, eye, outname) -> zip entry name
    z = zipfile.ZipFile(ZIP)
    rx = re.compile(r"(NHAMD-\d+-\d+-V\d+).*?_6x6_(O[DS])_\d{8}_\d{6}")
    for n in z.namelist():
        m = rx.search(n)
        if not m:
            continue
        subj, eye, low = m.group(1), m.group(2), n.lower()
        hit = next((out for suf, out in GA_BY_NAME.items() if low.endswith(suf)), None)
        if hit:
            ga_entry[(subj, eye, hit)] = n
        elif low.endswith(".csv"):
            try:
                head = z.read(n)[:120].decode("utf-8", "replace").lower()
            except Exception:
                continue
            if "total_ga" in head:
                ga_entry[(subj, eye, "advrpe_ga_vals.csv")] = n

    os.makedirs(COH, exist_ok=True)
    for subject in subjects:
        sdir = os.path.join(COH, subject)
        os.makedirs(sdir, exist_ok=True)
        e2e_rel = pair[(subject, next(e for s, e in pair if s == subject))]["e2e_file"]
        print(f"[{subject}] {e2e_rel}", flush=True)

        e = E2E(os.path.join(DATA_DIR, e2e_rel))
        md = e.read_all_metadata()
        emap = md.get("enface_modality", {})
        ir, baf = {}, {}
        for f in e.read_fundus_image():
            sid = str(getattr(f, "image_id", ""))
            eye = eye_of(getattr(f, "laterality", ""))
            mod = emap.get(sid)
            if mod == "IR" and eye not in ir:
                ir[eye] = norm8(f.image)
            elif mod == "BAF" and eye not in baf:
                baf[eye] = norm8(f.image)
        patient_id = (md.get("patient_data") or [{}])[0].get("patient_id")

        meta = {"subject": subject, "e2e_file": e2e_rel, "e2e_patient_id": patient_id, "eyes": {}}
        for eye in ("OD", "OS"):
            if (subject, eye) not in pair:
                continue
            ed = os.path.join(sdir, eye)
            os.makedirs(ed, exist_ok=True)
            if eye in ir:
                Image.fromarray(ir[eye]).save(os.path.join(ed, "spectralis_ir.png"))
            if eye in baf:
                Image.fromarray(baf[eye]).save(os.path.join(ed, "spectralis_baf.png"))
            msk = os.path.join(MASKS, f"{subject}_{eye}_GAmask.png")
            if os.path.isfile(msk):
                shutil.copyfile(msk, os.path.join(ed, "ga_mask.png"))
            for out in GA_OUTS:
                nm = ga_entry.get((subject, eye, out))
                if nm:
                    with open(os.path.join(ed, out), "wb") as fh:
                        fh.write(z.read(nm))
            p, v = pair[(subject, eye)], val.get((subject, eye), {})
            meta["eyes"][eye] = {
                "qc_status": p.get("qc_status"), "qc_reason": p.get("qc_reason"),
                "advRPE_area_mm2": p.get("advRPE_area_mm2"),
                "acq_date": v.get("e2e_acq_date"), "ga_visit_date": v.get("ga_visit_date"),
                "fov_mm": [v.get("fov_H_mm"), v.get("fov_V_mm")], "fov_volume": v.get("fov_volume"),
            }
        with open(os.path.join(sdir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
    z.close()

    with open(os.path.join(COH, "_README.md"), "w") as fh:
        fh.write(README)
    print(f"\nDONE -> {COH}  ({len(subjects)} subjects)")


README = """# cohort/ — working dataset (one folder per subject-visit, per-eye artifacts)

Light, inspectable files gathered from the heavy archives. **Master index:** `../spectralis_ga_pairing.csv`
(has `qc_status`/`qc_reason`; filter to `qc_status == ok`).

Per eye:
- `spectralis_ir.png`  — Spectralis NIR localizer; registration SOURCE (from the E2E).
- `spectralis_baf.png` — BAF en-face; dev/QC reference only, never in the production number.
- `ga_mask.png`        — validated binary advRPE GA mask; the reference standard.
- `advrpe_subrpe_enface.png` / `advrpe_pseudocolor.png` — PLEX 6x6 en-face; registration TARGETS.
- `advrpe_ga_outline.png` — advRPE GA as the translucent yellow fill (provenance of ga_mask).
- `advrpe_ga_vals.csv` — advRPE area (mm2), mmX/mmY, date, patientID.

Some subjects store advRPE files under truncated 8.3 names; ga_vals is matched by content
(Total_GA header) and pseudocolor may be absent for those (subrpe covers the en-face target).

NOT copied here (read on demand): the 300 MB E2E (path in `meta.json`), full B-scan stacks,
PLEX cubes, GA-zip extras (cube TIFs, layer maps). Calibration: Spectralis is angular; mm uses the
24 mm model eye, 0.2924 mm/deg.

Excluded eyes (see qc_status): 003-007 (Spectralis ~7 mo off its GA visit), 003-013 (E2E is
vessel-confirmed patient 012, cross-patient vs GA-013).
"""

if __name__ == "__main__":
    main()
