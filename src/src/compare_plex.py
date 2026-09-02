#!/usr/bin/env python
"""OCT-only GA area vs the PLEX (advRPE) reference, across every QC-ok cohort eye, under two BM regimes.

For each eye (qc_status==ok in spectralis_ga_pairing.csv) it opens the native 6x6 (97-line) volume the
reader uses, then computes the OAC GA area (reader.core.oac_ga, the SAME detector the reader/viewer run)
under two Bruch's-membrane surfaces:

  * EFFECTIVE BM  = the hand-VALIDATED BM where the eye was validated in the reader
                    (reader/data_store/corrections/<eid>_<eye>/bm_status.json non-empty),
                    otherwise the device/self-seg BM (effective_surfaces).
  * DL BM         = the trained DL Bruch's-membrane model (src/bm_dl.segment_volume), every eye.

From those it reports the two scenarios the analysis asks for:
  Scenario A (HYBRID)  = validated BM where validated, DL BM otherwise.
  Scenario B (ALL-DL)  = DL BM for every eye.

Each area is computed at BOTH baseline orders (quadratic = the reader/CLI default; linear = the doctor
viewer default) so the baseline choice is transparent. Output -> results/plex_compare.csv (+ console
summary). No file is overwritten in place beyond that CSV.

Run (repo root):
  oct_env\\Scripts\\python.exe src\\compare_plex.py            # all qc-ok eyes
  oct_env\\Scripts\\python.exe src\\compare_plex.py --only 005 # substring filter (quick test)
"""
import argparse
import csv
import glob
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import bm_dl  # noqa: E402
from paths import DATA_DIR, RESULTS_DIR  # noqa: E402
from reader.core import e2e_source, layers as core_layers, oac_ga  # noqa: E402
from reader.core.layer_store import JsonSidecarLayerStore  # noqa: E402

CORR_DIR = os.path.join(_REPO, "reader", "data_store", "corrections")
PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "plex_compare.csv")


def validated_eyes():
    """{(eid, eye)} for every corrections folder with a non-empty bm_status.json — i.e. >=1 B-scan
    explicitly validated in the reader (the SAME definition the doctor-viewer baker uses)."""
    out = set()
    for d in glob.glob(os.path.join(CORR_DIR, "*_*")):
        bs = os.path.join(d, "bm_status.json")
        if not os.path.exists(bs):
            continue
        try:
            st = json.load(open(bs))
        except (OSError, json.JSONDecodeError):
            continue
        if not st:
            continue
        name = os.path.basename(d)
        eid, _, eye = name.rpartition("_")
        if eid and eye:
            out.add((eid, eye))
    return out


def area(ov, bm, order):
    """OAC GA area (mm^2) for this volume + BM at the given healthy-baseline polynomial order."""
    return oac_ga.detect(ov, bm, trend_order=order)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter on subject (quick test)")
    args = ap.parse_args()

    val = validated_eyes()
    print(f"validated eyes (non-empty bm_status.json): {len(val)}", flush=True)
    print(f"DL BM model: {bm_dl.model_path()}  backend={bm_dl.backend()}", flush=True)
    store = JsonSidecarLayerStore(CORR_DIR)

    with open(PAIRING, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("qc_status") or "").strip() == "ok"]
    if args.only:
        rows = [r for r in rows if args.only.lower() in r["subject"].lower()]

    raw_cache = {}
    out_rows = []
    for r in rows:
        subj, visit, eye = r["subject"], r["visit"], r["eye"].upper()
        e2e_path = os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
        if not os.path.exists(e2e_path):
            print(f"  SKIP {subj} {eye}: E2E missing", flush=True)
            continue
        try:
            adv = float(r["advRPE_area_mm2"])
        except (TypeError, ValueError):
            adv = float("nan")
        if e2e_path not in raw_cache:
            raw_cache.clear()
            raw_cache[e2e_path] = e2e_source.open_e2e(e2e_path)
        raw = raw_cache[e2e_path]
        idx = e2e_source.default_volume_index(raw, eye)
        ov = e2e_source.load_volume(raw, idx)
        is_val = (ov.eid, eye) in val

        ilm, bm_eff = core_layers.effective_surfaces(ov, store)         # validated where validated
        bm_dl_surf = bm_dl.segment_volume(ov.vol)                       # DL everywhere

        a_eff_q, a_eff_l = area(ov, bm_eff, 2), area(ov, bm_eff, 1)
        a_dl_q, a_dl_l = area(ov, bm_dl_surf, 2), area(ov, bm_dl_surf, 1)

        rec = {
            "subject": subj, "visit": visit, "eye": eye,
            "plex_mm2": round(adv, 4),
            "is_validated": int(is_val), "bm_src": ov.bm_src, "n_bscans": ov.n_bscans,
            "sat_band": int(ov.field_invalid is not None and bool(np.asarray(ov.field_invalid).any())),
            # effective-BM (validated where validated)
            "eff_quad": round(a_eff_q, 4), "eff_lin": round(a_eff_l, 4),
            # DL-BM (every eye)
            "dl_quad": round(a_dl_q, 4), "dl_lin": round(a_dl_l, 4),
            # Scenario A (hybrid) = validated where validated else DL
            "A_quad": round(a_eff_q if is_val else a_dl_q, 4),
            "A_lin": round(a_eff_l if is_val else a_dl_l, 4),
            # Scenario B (all DL)
            "B_quad": round(a_dl_q, 4), "B_lin": round(a_dl_l, 4),
        }
        out_rows.append(rec)
        print(f"  {subj} {eye:2}  val={int(is_val)} bm_src={ov.bm_src:6}  PLEX={adv:6.2f}  "
              f"A(q/l)={rec['A_quad']:6.2f}/{rec['A_lin']:6.2f}  "
              f"B-DL(q/l)={rec['B_quad']:6.2f}/{rec['B_lin']:6.2f}", flush=True)

    cols = ["subject", "visit", "eye", "plex_mm2", "is_validated", "bm_src", "n_bscans", "sat_band",
            "eff_quad", "eff_lin", "dl_quad", "dl_lin", "A_quad", "A_lin", "B_quad", "B_lin"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {OUT_CSV}  ({len(out_rows)} eyes)", flush=True)


if __name__ == "__main__":
    main()
