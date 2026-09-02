#!/usr/bin/env python
"""Hypertransmission-locator bake-off: candidate sub-BM hyper en-faces vs the advRPE reference.

The production hyper channel (m3_slab.hyper_enface) is `slab(BM+130..250um) / median(rpe(BM-40..+10um))`
-- a per-EYE SCALAR normalisation, so spatially it is just the deep-choroid slab (keeps vessel shadows;
can't tell bright-sub-BM-because-RPE-gone (GA) from bright-sub-BM-with-RPE-present (016 OD FP)). Its own
docstring describes a better, never-wired PER-A-SCAN LOCAL contrast. This script renders, per eye, the
candidates side by side with the advRPE SubRPE en-face (THE reference hypertransmission substrate) + the
advRPE GA outline, so we can judge VISUALLY which locator best matches the reference and best discriminates
RPE-present-but-bright (016 OD) from true GA -- while holding the gold (005 OD).

Variants (all native n x W, then destriped + to_enface):
  v0_scalar   slab / median(rpe)                       (PRODUCTION)
  v1_local    slab / (rpe_local + eps)                 (per-A-scan ratio)
  v2_mich     (slab - rpe_local) / (slab + rpe_local)  (per-A-scan Michelson, bounded [-1,1])

Run (repo root):
  oct_env\\Scripts\\python.exe src\\hyper_locator.py                 # default discriminative set
  oct_env\\Scripts\\python.exe src\\hyper_locator.py NHAMD-003-016 V2 OD
Output -> outputs/hyper_locator/<subject>_<eye>_hyper.png
"""
import argparse
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("OCT_BM_DL", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import bm_dl  # noqa: E402
import m3_projections as mp  # noqa: E402
import qcviz as qv  # noqa: E402
from paths import DATA_DIR, OUT_DIR, RESULTS_DIR  # noqa: E402
from reader.core import e2e_source  # noqa: E402
from reader.core import projection as proj  # noqa: E402

PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
COHORT = os.path.join(_REPO, "cohort")
OUT = os.path.join(OUT_DIR, "hyper_locator")

SLAB = (130.0, 250.0)        # deep-choroid slab (um below BM)
RPEB = (-40.0, 10.0)         # RPE reference band (um around BM)


def variants(vol, bm):
    """Return dict name -> native (n,W) hyper map for each candidate."""
    slab = mp.band(vol.astype(np.float32), bm, *SLAB, "mean")
    rpe = mp.band(vol.astype(np.float32), bm, *RPEB, "mean")
    eps = 0.02 * float(np.median(rpe) + 1e-6)        # scale eps to the eye's RPE level
    return {
        "v0_scalar": slab / (float(np.median(rpe)) + 0.02),
        "v1_local": slab / (rpe + eps),
        "v2_mich": (slab - rpe) / (slab + rpe + eps),
    }


def to6(nat):
    return mp.destripe2d(nat, signed=True) if nat.min() < 0 else mp.destripe2d(nat, signed=False)


def disp(enf, p=99):
    """uint8 display: robust percentile stretch (signed maps centered)."""
    a = np.nan_to_num(np.asarray(enf, np.float32))
    if a.min() < 0:
        m = np.percentile(np.abs(a), p) + 1e-9
        return qv.norm8(np.clip(a, -m, m))
    return qv.norm8(np.clip(a, 0, np.percentile(a, p) + 1e-9))


def ref_tile(subject, eye, name, shape):
    p = os.path.join(COHORT, subject, eye, name)
    if not os.path.exists(p):
        return np.zeros(shape, np.uint8)
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (shape[1], shape[0]))


def row_for(subject, visit, eye):
    with open(PAIRING, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("qc_status") or "").strip() == "ok" and subject in r["subject"] \
                    and r["visit"] == visit and r["eye"].upper() == eye:
                return r
    raise SystemExit(f"{subject} {visit} {eye} not qc_ok in pairing")


def render(subject, visit, eye):
    r = row_for(subject, visit, eye)
    subj = r["subject"]
    try:
        adv = float(r["advRPE_area_mm2"])
    except (TypeError, ValueError):
        adv = float("nan")
    raw = e2e_source.open_e2e(os.path.join(DATA_DIR, *r["e2e_file"].split("/")))
    ov = e2e_source.load_volume(raw, e2e_source.default_volume_index(raw, eye))
    bm = bm_dl.segment_volume(ov.vol)
    vs = variants(ov.vol, bm)
    e6 = {k: proj.to_enface(to6(v), ov.fov_mm) for k, v in vs.items()}
    shape = next(iter(e6.values())).shape

    tiles = [ref_tile(subj, eye, "advrpe_subrpe_enface.png", shape),
             ref_tile(subj, eye, "advrpe_ga_outline.png", shape),
             disp(e6["v0_scalar"]), disp(e6["v1_local"]), disp(e6["v2_mich"])]
    titles = ["advRPE SubRPE (reference)", f"advRPE GA outline ({adv:.2f} mm²)",
              "v0 scalar (PRODUCTION)", "v1 local ratio", "v2 Michelson"]
    os.makedirs(OUT, exist_ok=True)
    panel = qv.panel([qv.ensure_rgb(t) for t in tiles], titles,
                     header=f"{subj} {eye}  hypertransmission locators vs advRPE  (PLEX {adv:.2f} mm²)  "
                            f"slab BM+{SLAB[0]:.0f}..{SLAB[1]:.0f}µm / rpe BM{RPEB[0]:.0f}..+{RPEB[1]:.0f}µm",
                     mm_per_px=proj.ENFACE_MMPP)
    out = os.path.join(OUT, f"{subj}_{eye}_hyper.png")
    qv.save_rgb(out, panel)
    print(f"  wrote {out}  (PLEX {adv:.2f})", flush=True)


DEFAULT = [("NHAMD-003-005", "V3", "OD"),    # gold focal GA — must light up correctly
           ("NHAMD-003-008", "V1", "OS"),    # large confluent GA
           ("NHAMD-003-016", "V2", "OD"),    # FP control — RPE present, bright sub-BM (must NOT light up)
           ("NHAMD-003-006", "V3", "OS")]    # clean control (PLEX 0) — must be dark


def main():
    a = sys.argv[1:]
    eyes = DEFAULT if not a else [(a[0] if a[0].startswith("NHAMD") else "NHAMD-003-" + a[0], a[1], a[2])]
    print(f"DL BM: {bm_dl.model_path()} ({bm_dl.backend()})", flush=True)
    for subj, visit, eye in eyes:
        try:
            render(subj, visit, eye.upper())
        except Exception as e:                       # noqa: BLE001
            print(f"  FAILED {subj} {eye}: {e!r}", flush=True)


if __name__ == "__main__":
    main()
