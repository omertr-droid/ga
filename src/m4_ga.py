#!/usr/bin/env python
"""M4: GA map + area. Hypertransmission en-face (M3) -> resample into the fovea-centred 6x6 mm
frame -> threshold -> cRORA >=250 um morphology -> area (mm2), vs the advRPE reference.

Runs each eye with BOTH BM sources (DEVICE BM and our SELF-seg) so we can see which segmentation
the result needs. The cRORA step (drop anything <250 um across) should also remove the thin
per-B-scan stripes. Per-eye QC panel + an areas CSV.

Run: oct_env\\Scripts\\python.exe m4_ga.py [all | SUBJECT EYE ...]   (default: a spread of eyes)
"""
import csv
import json
import os
import sys

import cv2
import numpy as np
from skimage import measure, morphology

import bm as bmseg
import m2_bm
import m3_slab
import qcviz as qv
import register_qc as reg

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
COH = os.path.join(ROOT, "cohort")
OUT = os.path.join(OUT_DIR, "m4_ga_out")
MMPP = reg.ADV_MMPP                       # 6/512 mm/px in the 6x6 frame
MIN_DIAM_PX = 0.250 / MMPP                 # 250 um cRORA min diameter ~ 21.3 px
THR = float(os.environ.get("THR", "0.30"))  # ABSOLUTE threshold on the deep-choroid/RPE ratio

# a spread spanning advRPE area: controls -> small -> mid -> large (+ one no-device eye, 003-010 OS)
DEFAULT = [("NHAMD-003-009-V2", "OS"), ("NHAMD-003-002-V2", "OD"), ("NHAMD-003-012-V3", "OD"),
           ("NHAMD-003-010-V1", "OS"), ("NHAMD-003-014-V1", "OS"), ("NHAMD-003-017-V3", "OD"),
           ("NHAMD-003-006-V3", "OD"), ("NHAMD-003-005-V3", "OD"), ("NHAMD-003-001-V1", "OD"),
           ("NHAMD-003-003-V3", "OS"), ("NHAMD-003-004-V1", "OD"), ("NHAMD-003-008-V1", "OD")]


def fov_of(subject, eye):
    try:
        with open(os.path.join(COH, subject, "meta.json")) as f:
            fov = json.load(f)["eyes"][eye]["fov_mm"]
        return float(fov[0]), float(fov[1])
    except Exception:
        return 8.77, 7.31


def to_6mm(enface_native, fov):
    """Resample a native (n x W) en-face into the fovea-centred 6x6 mm 512 frame.
    Flip rows: the OCT B-scan order is vertically mirrored vs the IR/advRPE (PLEX) frame."""
    return reg.resample(enface_native[::-1].astype(np.float32),
                        (fov[0] / enface_native.shape[1], fov[1] / enface_native.shape[0]))


def crora(enf):
    """Absolute threshold on the hypertransmission ratio, then keep only components >=250 um across."""
    binimg = enf.astype(float) > THR
    binimg = morphology.remove_small_holes(binimg, area_threshold=int(MIN_DIAM_PX ** 2))
    lbl = measure.label(binimg)
    keep = np.zeros_like(binimg, bool)
    for r in measure.regionprops(lbl):
        if r.axis_major_length >= MIN_DIAM_PX:
            keep[lbl == r.label] = True
    return keep


def dice(a, b):
    a, b = a > 0, b > 0
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else (1.0 if not s else 0.0)


def ga_from_bm(vol, bmrow, fov):
    hyper6 = to_6mm(m3_slab.hyper_enface(vol, bmrow), fov)
    mask = crora(hyper6)
    disp = qv.norm8(np.clip(hyper6, 0.1, 0.55))   # fixed display scale (deep/RPE ratio)
    return hyper6, disp, mask, float(mask.sum()) * MMPP ** 2


def process(subject, eye, vol, dev_bm):
    os.makedirs(OUT, exist_ok=True)
    fov = fov_of(subject, eye)
    adv = cv2.imread(os.path.join(COH, subject, eye, "advrpe_subrpe_enface.png"), cv2.IMREAD_GRAYSCALE)
    advm = cv2.imread(os.path.join(COH, subject, eye, "ga_mask.png"), cv2.IMREAD_GRAYSCALE)
    advm = (advm > 127) if advm is not None else np.zeros((512, 512), bool)
    with open(os.path.join(COH, subject, "meta.json")) as f:
        adv_area = float(json.load(f)["eyes"][eye].get("advRPE_area_mm2") or 0.0)

    out = {"subject": subject, "eye": eye, "advRPE_mm2": round(adv_area, 3)}
    tiles, titles = [], []

    # reference tiles: SubRPE en-face + GA contour, AND the advRPE GA annotation map (yellow fill)
    ref_t = qv.draw_contour(adv if adv is not None else np.zeros((512, 512), np.uint8), advm,
                            color=(0, 220, 255), thick=2)
    tiles.append(ref_t); titles.append(f"advRPE SubRPE + GA  ({adv_area:.2f} mm2)")
    ga_ann = cv2.imread(os.path.join(COH, subject, eye, "advrpe_ga_outline.png"), cv2.IMREAD_COLOR)
    if ga_ann is not None:
        ga_ann = cv2.cvtColor(cv2.resize(ga_ann, (512, 512)), cv2.COLOR_BGR2RGB)
    else:                                   # fall back to the binary mask if no outline image
        ga_ann = qv.ensure_rgb((advm * 255).astype(np.uint8))
    tiles.append(ga_ann); titles.append("advRPE GA annotation map")

    p99 = ""
    for label, bmrow in (("DEVICE BM", m2_bm.fill_bm(dev_bm) if dev_bm is not None else None),
                         ("SELF-seg BM", bmseg.segment_volume(vol))):
        if bmrow is None:
            tiles.append(np.zeros((512, 512, 3), np.uint8)); titles.append("no DEVICE BM")
            out["device_mm2"] = ""
            continue
        hyper6, disp, mask, area = ga_from_bm(vol, bmrow, fov)
        t = qv.draw_contour(disp, mask, color=(255, 255, 0), thick=2)         # our GA yellow
        t = qv.draw_contour(t, advm, color=(0, 220, 255), thick=1)           # advRPE GA cyan
        tiles.append(t)
        titles.append(f"GA via {label}  ({area:.2f} mm2, Dice {dice(mask, advm):.2f})")
        out["device_mm2" if "DEVICE" in label else "self_mm2"] = round(area, 3)
        if "DEVICE" in label:
            p99 = f"hyperP[50/90/99]={np.percentile(hyper6,50):.2f}/{np.percentile(hyper6,90):.2f}/{np.percentile(hyper6,99):.2f}"

    hdr = (f"{subject} {eye}   advRPE={adv_area:.2f}  device={out.get('device_mm2','-')}  "
           f"self={out.get('self_mm2','-')} mm2   (yellow=ours, cyan=advRPE; THR={THR})")
    qv.save_rgb(os.path.join(OUT, f"{subject}_{eye}_m4.png"),
                qv.panel(tiles, titles, header=hdr))
    print(f"[{subject} {eye}] advRPE={adv_area:.2f} device={out.get('device_mm2','-')} "
          f"self={out.get('self_mm2','-')} mm2   {p99}", flush=True)
    return out


def main():
    args = sys.argv[1:]
    if args and args[0].lower() == "all":
        with open(os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv"), newline="") as f:
            pairs = [(r["subject"], r["eye"]) for r in csv.DictReader(f) if r.get("qc_status") == "ok"]
    elif args:
        pairs = [((args[i] if args[i].startswith("NHAMD") else "NHAMD-003-" + args[i]), args[i + 1])
                 for i in range(0, len(args) - 1, 2)]
    else:
        pairs = DEFAULT

    by_sub = {}
    for s, e in pairs:
        by_sub.setdefault(s, []).append(e)
    results = []
    for subject in sorted(by_sub):
        try:
            loaded = m2_bm.load_subject(subject)
        except Exception as ex:
            print(f"[{subject}] LOAD ERROR {type(ex).__name__}", flush=True); continue
        for eye in by_sub[subject]:
            if eye in loaded:
                results.append(process(subject, eye, *loaded[eye]))

    cols = ["subject", "eye", "advRPE_mm2", "device_mm2", "self_mm2"]
    with open(os.path.join(RESULTS_DIR, "m4_areas.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nwrote {len(results)} eyes -> m4_ga_out/  + m4_areas.csv  (THR={THR})")


if __name__ == "__main__":
    main()
