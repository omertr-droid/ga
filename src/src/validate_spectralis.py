#!/usr/bin/env python
"""Validate the extracted Spectralis cohort against two requirements:

1. METADATA INTEGRITY -- the E2E's embedded patient_id / first_name must match the
   patient number encoded in the folder name (not trust the folder name alone), and the
   E2E acquisition date must match the paired advRPE/PLEX visit date (confirms the visit).
2. 6x6 COVERAGE -- the advRPE GA reference is a PLEX 6x6 mm field, so each cohort eye must
   have a Spectralis OCT volume whose field of view covers >= 6 x 6 mm. Spectralis geometry
   is angular; we convert with the 24 mm model-eye factor (0.2924 mm/deg, per CLAUDE.md),
   the same assumption HEYEX uses when no biometry was entered.

Reads which (subject, eye) pairs to check from spectralis_ga_pairing.csv.
Run: oct_env\\Scripts\\python.exe validate_spectralis.py
Output: spectralis_validation.csv  (one row per cohort eye) + console summary.
"""
import csv
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from oct_converter.readers import E2E

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
OUT = os.path.join(RESULTS_DIR, "spectralis_validation.csv")

MM_PER_DEG = 0.2924   # 24 mm model eye; reproduces the HEYEX display (CLAUDE.md Calibration)
MIN_FOV_MM = 6.0      # advRPE reference field is 6 x 6 mm; scan must CONTAIN this box (then crop)

COLUMNS = [
    "subject", "visit", "eye", "e2e_file",
    "e2e_patient_id", "folder_pnum", "id_number_match",
    "e2e_first_name", "name_note",
    "e2e_acq_date", "ga_visit_date", "date_match",
    "fov_volume", "fov_H_mm", "fov_V_mm", "covers_6x6",
    "eye_volumes",
]


def eye_tag(lat):
    s = str(lat).strip().upper()
    if s in ("R", "OD", "RIGHT"):
        return "OD"
    if s in ("L", "OS", "LEFT"):
        return "OS"
    return "U"


def parse_date(s):
    s = str(s).strip()
    if not s or s.lower() == "none":
        return None
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    try:
        return datetime.strptime(s.split()[0], "%d-%b-%Y").date()
    except ValueError:
        return None


def class_fov_mm(bscan_data):
    """Map each volume class (numImages, width) -> (H_mm, V_mm), measured directly from the
    per-B-scan angular endpoints (posX span = horizontal field; centrePosY span = vertical).
    Coverage is judged by FIELD EXTENT, not B-scan count: a 61-line 30x25 volume covers the
    same 8.8x7.3 mm as the 121-line, just with coarser slow-axis sampling."""
    groups = defaultdict(list)
    for b in bscan_data:
        groups[(b.get("numImages"), b.get("imgSizeX"))].append(b)
    out = {}
    for key, grp in groups.items():
        hs = [b["posX2"] - b["posX1"] for b in grp
              if b.get("posX1") is not None and b.get("posX2") is not None]
        cys = [b["centrePosY"] for b in grp if b.get("centrePosY") is not None]
        if not hs:
            continue
        h = float(np.median(hs)) * MM_PER_DEG
        v = (float(max(cys) - min(cys)) * MM_PER_DEG) if cys else 0.0
        out[key] = (h, v)
    return out


def extract_pnum(*strings):
    """Pull the 3-digit patient number from an E2E id/name field, tolerating prefixes and
    formatting variants ('003-012', 'NHAMD-003-016', 'NHAMD003-001')."""
    for s in strings:
        m = re.search(r"003-?0*(\d{1,3})", str(s))
        if m:
            return m.group(1).zfill(3)
    return None


def main():
    rows = list(csv.DictReader(open(PAIRING, newline="")))
    by_sub = defaultdict(list)
    for r in rows:
        by_sub[r["subject"]].append(r)

    out_rows = []
    flags = []
    for subject in sorted(by_sub):
        eyes = by_sub[subject]
        m = re.match(r"NHAMD-003-(\d+)-V(\d+)", subject)
        pnum, visit = m.group(1).zfill(3), "V" + m.group(2)
        e2e_rel = eyes[0]["e2e_file"]
        e2e_path = os.path.join(DATA_DIR, e2e_rel)
        print(f"[{subject}] reading {e2e_rel} ...", flush=True)

        try:
            e = E2E(e2e_path)
            md = e.read_all_metadata()
            pd = (md.get("patient_data") or [{}])[0]
            pid = str(pd.get("patient_id", "")).strip()
            fname = str(pd.get("first_name", "")).strip()
            cfov = class_fov_mm(md.get("bscan_data") or [])

            # per-eye volume inventory (laterality authoritative via read_oct_volume) + acq date
            vols = e.read_oct_volume()
            acq = None
            eye_vols = defaultdict(list)   # eye -> [(nb, width), ...]
            for v in vols:
                et = eye_tag(getattr(v, "laterality", None))
                nb = len(v.volume)
                w = int(np.asarray(v.volume[0]).shape[1]) if nb else 0
                eye_vols[et].append((nb, w))
                if acq is None:
                    acq = getattr(v, "acquisition_date", None)
            acq_date = acq.date() if hasattr(acq, "date") else parse_date(acq)
        except Exception as ex:
            for r in eyes:
                bad = {c: "" for c in COLUMNS}
                bad.update(subject=subject, visit=visit, eye=r["eye"],
                           e2e_file=e2e_rel, covers_6x6="READ_ERROR")
                out_rows.append(bad)
            flags.append(f"{subject}: E2E READ ERROR: {ex!r}")
            print(f"   !! READ ERROR: {ex!r}", flush=True)
            continue

        # identity by patient NUMBER (tolerant of prefix/typo variants); name is a soft note
        e2e_pnum = extract_pnum(pid, fname)
        id_ok = (e2e_pnum == pnum)
        name_note = "" if fname == f"NHAMD-003-{pnum}" else f"name='{fname}'"
        if not id_ok:
            flags.append(f"{subject}: E2E patient is '{pid}' (#{e2e_pnum}) but folder is #{pnum}")

        for r in eyes:
            eye = r["eye"]
            ga_date = parse_date(r.get("date"))
            date_ok = (acq_date is not None and ga_date is not None and acq_date == ga_date)

            # best-covering volume for this eye: maximize min(H,V) across its volumes' class FOVs
            best = (0.0, 0.0, None)  # (H_mm, V_mm, "<nb>line/<w>px")
            for nb, w in eye_vols.get(eye, []):
                hv = cfov.get((nb, w))
                if hv and min(hv) > min(best[0], best[1]):
                    best = (hv[0], hv[1], f"{nb}line/{w}px")
            H_mm, V_mm, fov_vol = best
            covers = bool(H_mm >= MIN_FOV_MM and V_mm >= MIN_FOV_MM)

            if eye in eye_vols and not date_ok and ga_date is not None:
                flags.append(f"{subject} {eye}: E2E acq {acq_date} != GA visit {ga_date}")
            if eye in eye_vols and not covers:
                flags.append(f"{subject} {eye}: does NOT cover 6x6 (best {H_mm:.2f}x{V_mm:.2f}mm)")
            out_rows.append({
                "subject": subject, "visit": visit, "eye": eye, "e2e_file": e2e_rel,
                "e2e_patient_id": pid, "folder_pnum": pnum, "id_number_match": id_ok,
                "e2e_first_name": fname, "name_note": name_note,
                "e2e_acq_date": str(acq_date), "ga_visit_date": str(ga_date), "date_match": date_ok,
                "fov_volume": fov_vol or "", "fov_H_mm": round(H_mm, 2), "fov_V_mm": round(V_mm, 2),
                "covers_6x6": covers,
                "eye_volumes": ";".join(f"{nb}line/{w}px" for nb, w in sorted(eye_vols.get(eye, []))),
            })
        print(f"   pid={pid}(#{e2e_pnum} {'ok' if id_ok else 'MISMATCH'}) acq={acq_date} "
              f"ga={parse_date(eyes[0].get('date'))} covers6x6="
              f"{[(r['eye']) for r in eyes]}", flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out_rows)

    n = len(out_rows)
    id_ok = sum(1 for o in out_rows if o["id_number_match"] is True)
    date_ok = sum(1 for o in out_rows if o["date_match"] is True)
    cov_ok = sum(1 for o in out_rows if o["covers_6x6"] is True)
    print("\n" + "=" * 60)
    print(f"Wrote {OUT}  ({n} eye rows, {len({o['subject'] for o in out_rows})} subjects)")
    print(f"  patient NUMBER matches folder : {id_ok}/{n}")
    print(f"  acq date matches GA visit     : {date_ok}/{n}")
    print(f"  eye covers 6x6 mm             : {cov_ok}/{n}")
    if flags:
        print(f"\n  {len(flags)} FLAG(S):")
        for fl in flags:
            print("   - " + fl)
    else:
        print("\n  ALL CHECKS PASS.")


if __name__ == "__main__":
    main()
