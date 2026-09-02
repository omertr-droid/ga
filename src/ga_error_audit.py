#!/usr/bin/env python
"""GA SPATIAL error audit -- decompose OCT-vs-PLEX disagreement IN SPACE, both directions, so false
positives and false negatives can NEVER cancel into a deceptively-good area number (workflow
ga-error-experiments).

Runs the exact reader/viewer detector (reader.core.oac_ga) with DL BM (the library default), so numbers
match the baked `oac_area_dl_mm2`. Per eye it produces:

  SPATIAL DECOMPOSITION (pixels, restricted to the in-field `core`), reported SEPARATELY WITHOUT
  pretending PLEX is ground truth:
     overlap    = ours & PLEX
     ours_only  = ours & ~PLEX
     plex_only  = ~ours & PLEX
     spatial Dice, and net = ours_only - plex_only (== the area diff that cancellation hides).

  E1 GATE ATTRIBUTION -- mm2 removed/added by each detector gate (which gate ate a real lesion).

  EVIDENCE B-SCANS through EACH disagreement region, both directions, with the PLEX span (red), our GA
  (lime) and our raw candidate (yellow) marked -- the images to ADJUDICATE per the labeling protocol
  (judge the RPE band: complete loss + hypertransmission = GA; drusen with intact RPE = GA-free).

PLEX outline comes from the baked bundle (`meta.plex_polygons`, already registered into our en-face
frame). NOTE the registration is GA-driven + imperfect (small euclidean, up to ~12% scale on int-orb
eyes), so treat small spatial offsets with care; GROSS cancellation (large FP AND large FN in different
regions) is the robust signal.

Run (repo root):
  oct_env\\Scripts\\python.exe src\\ga_error_audit.py NHAMD-003-014 V1 OD
  oct_env\\Scripts\\python.exe src\\ga_error_audit.py --set        # standard disagreement set
  oct_env\\Scripts\\python.exe src\\ga_error_audit.py --cohort     # ALL 25 library eyes (spatial table)
Panels -> outputs/ga_audit/.  Spatial table -> results/ga_spatial_decomp.csv ; gates -> results/ga_gate_attribution.csv.
"""
import argparse
import csv
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OCT_BM_DL", "1")            # the library default is DL BM; match it

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bm_dl
from paths import DATA_DIR, OUT_DIR, RESULTS_DIR
from reader.core import e2e_source, layers, oac_ga
from reader.core.layer_store import JsonSidecarLayerStore
from viewer.core import ga_native

CORR_DIR = os.path.join(_REPO, "reader", "data_store", "corrections")
LIB_DIR = os.path.join(_REPO, "viewer", "data_store", "library")
OUT = os.path.join(OUT_DIR, "ga_audit")
MMPP2 = oac_ga.MMPP2

STD_SET = [
    ("NHAMD-003-014", "V1", "OD"), ("NHAMD-003-001", "V1", "OD"), ("NHAMD-003-001", "V1", "OS"),
    ("NHAMD-003-010", "V1", "OD"), ("NHAMD-003-006", "V3", "OD"), ("NHAMD-003-026", "V3", "OD"),
    ("NHAMD-003-026", "V3", "OS"), ("NHAMD-003-008", "V1", "OS"), ("NHAMD-003-011", "V3", "OD"),
    ("NHAMD-003-011", "V3", "OS"), ("NHAMD-003-015", "V3", "OD"),
    ("NHAMD-003-003", "V3", "OD"), ("NHAMD-003-003", "V3", "OS"),
    ("NHAMD-003-005", "V3", "OD"), ("NHAMD-003-005", "V3", "OS"),
]


def load_adjudication():
    p = os.path.join(RESULTS_DIR, "plex_adjudication.csv")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8-sig") as f:
        rows = csv.DictReader([ln for ln in f if not ln.lstrip().startswith("#")])
        return {(r["subject"], r["eye"].upper()): r for r in rows}


ADJUDICATION = load_adjudication()


def resolve(subject, visit, eye):
    want = subject if subject.endswith(f"-{visit}") else f"{subject}-{visit}"
    eye = eye.upper()
    with open(os.path.join(RESULTS_DIR, "bm_worklist.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["subject"] == want and r["eye"].upper() == eye:
                return r, os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
    return None, None


def bundle_slug(subject, visit, eye):
    base = subject if subject.endswith(f"-{visit}") else f"{subject}-{visit}"
    return f"{base}_{eye.upper()}"


def subject_visit(subject, visit):
    return subject if subject.endswith(f"-{visit}") else f"{subject}-{visit}"


def plex_label_enface(slug, out):
    mp = os.path.join(LIB_DIR, slug, "meta.json")
    if not os.path.exists(mp):
        return None, None
    meta = json.load(open(mp))
    lab = np.zeros((out, out), np.uint8)
    for poly in (meta.get("plex_polygons") or []):
        cv2.fillPoly(lab, [np.array(poly, np.int32).reshape(-1, 1, 2)], 1)
    return lab.astype(bool), meta


def norm8(bs):
    lo, hi = np.percentile(bs, 1), np.percentile(bs, 99.5)
    return np.clip((bs - lo) / (hi - lo + 1e-6), 0, 1)


def pick_bscans(nat, k=4):
    per = nat.sum(axis=1)
    return sorted(int(i) for i in np.argsort(-per)[:k] if per[i] > 0)


def draw_spans(ax, W, H, bm_row, plex_row, our_row, cand_row):
    ax.plot(np.arange(W), bm_row, color="yellow", lw=0.7)
    for x0, x1 in ga_native.intervals(cand_row):
        ax.plot([x0, x1], [H - 24, H - 24], color="yellow", lw=3)
    for x0, x1 in ga_native.intervals(our_row):
        ax.plot([x0, x1], [H - 16, H - 16], color="lime", lw=4)
    for x0, x1 in ga_native.intervals(plex_row):
        ax.plot([x0, x1], [H - 7, H - 7], color="red", lw=4)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def audit_eye(subject, visit, eye, sp_writer=None, gate_writer=None, render=True):
    row, e2e = resolve(subject, visit, eye)
    if row is None or not os.path.exists(e2e):
        print(f"  skip {subject} {visit} {eye}: no E2E"); return None
    adv = float(row["advRPE_area_mm2"])
    raw = e2e_source.open_e2e(e2e)
    idx = e2e_source.default_volume_index(raw, eye.upper())
    ov = e2e_source.load_volume(raw, idx)
    bp = os.path.join(LIB_DIR, bundle_slug(subject, visit, eye), "bundle.npz")
    bm = None
    if os.path.exists(bp):
        try:
            with np.load(bp, allow_pickle=False) as z:
                baked = np.asarray(z["bm_dl"], np.float32) if "bm_dl" in z.files else None
            if baked is not None and baked.shape == (ov.n_bscans, ov.W):
                bm = baked; bm_src = "baked-DL"
        except (OSError, ValueError, KeyError):
            bm = None
    if bm is None and bm_dl.available():
        bm = bm_dl.segment_volume(ov.vol).astype(np.float32); bm_src = "DL"
    elif bm is None:
        _, bm = layers.effective_surfaces(ov, JsonSidecarLayerStore(CORR_DIR)); bm_src = ov.bm_src

    P = oac_ga.prep(ov, bm, baseline="radial2")
    S = oac_ga.footprint_stages(P)
    core = P["core"]
    out = P["rpe6"].shape[0]
    plex6, _ = plex_label_enface(bundle_slug(subject, visit, eye), out)
    if plex6 is None:
        plex6 = np.zeros((out, out), bool)

    ours = S["final"] & core
    plex = plex6 & core
    overlap = ours & plex; ours_only = ours & ~plex; plex_only = plex & ~ours
    A = lambda m: float(np.asarray(m, bool).sum()) * MMPP2
    overlap_a, ours_only_a, plex_only_a = A(overlap), A(ours_only), A(plex_only)
    dice = 2 * overlap_a / (2 * overlap_a + ours_only_a + plex_only_a + 1e-9)
    # gate attribution
    cand = A(S["rpe_candidate"]); after_hyper = A(S["hyper_kept"] & S["rpe_candidate"])
    after_fill = A(S["filled"]); after_size = A(S["sized"]); final = A(S["final"])

    adj = ADJUDICATION.get((bundle_slug(subject, visit, eye).rsplit("_", 1)[0], eye.upper()), {})
    verdict = adj.get("verdict", "")
    sv = subject_visit(subject, visit)
    print(f"\n=== {sv} {eye}  BM={bm_src}  PLEX {adv:.2f} | ours {final:.2f} mm2"
          f" | adjudication={verdict or 'none'} ===")
    print(f"  SPATIAL: overlap {overlap_a:.2f} | ours-only {ours_only_a:.2f} | PLEX-only {plex_only_a:.2f}"
          f" | Dice {dice:.2f} | net {ours_only_a-plex_only_a:+.2f}")
    print(f"  E1 gates: candidate {cand:.2f} -> require-hyper {after_hyper:.2f} -> +holes {after_fill:.2f}"
          f" -> cRORA-size {after_size:.2f} -> complete-loss {final:.2f}")
    if sp_writer:
        sp_writer.writerow([sv, eye.upper(), verdict, round(adv, 3), round(final, 3),
                            round(overlap_a, 3), round(ours_only_a, 3), round(plex_only_a, 3), round(dice, 3),
                            round(ours_only_a - plex_only_a, 3), round(ours_only_a + plex_only_a, 3)])
    if gate_writer:
        gate_writer.writerow([sv, eye.upper(), round(adv, 3), round(cand, 3),
                              round(after_hyper, 3), round(after_fill, 3), round(after_size, 3), round(final, 3)])
    if not render:
        return dict(overlap=overlap_a, ours_only=ours_only_a, plex_only=plex_only_a, dice=dice)

    flip = getattr(ov, "enface_flip", True)
    our_nat = ga_native.enface_to_native(ours, ov.fov_mm, ov.n_bscans, ov.W, flip)
    cand_nat = ga_native.enface_to_native(S["rpe_candidate"], ov.fov_mm, ov.n_bscans, ov.W, flip)
    plex_nat = ga_native.enface_to_native(plex, ov.fov_mm, ov.n_bscans, ov.W, flip)
    plex_only_nat = ga_native.enface_to_native(plex_only, ov.fov_mm, ov.n_bscans, ov.W, flip)
    ours_only_nat = ga_native.enface_to_native(ours_only, ov.fov_mm, ov.n_bscans, ov.W, flip)
    plex_only_bs = pick_bscans(plex_only_nat) or pick_bscans(plex_nat)
    ours_only_bs = pick_bscans(ours_only_nat)

    fig, axs = plt.subplots(3, 4, figsize=(18, 13))
    rpe_rgb = np.dstack([norm8(P["rpe6"])] * 3)
    axs[0, 0].imshow(rpe_rgb); axs[0, 0].set_title("RPE-loss en-face (dark = GA)", fontsize=10)
    plex_tint = rpe_rgb.copy(); plex_tint[plex] = [0.9, 0.1, 0.1]
    axs[0, 1].imshow(plex_tint); axs[0, 1].set_title(f"PLEX GA (red) = {adv:.2f} mm2", fontsize=10)
    comp = rpe_rgb.copy() * 0.5
    comp[overlap] = [0, 0.9, 0]; comp[ours_only] = [1, 0.85, 0]; comp[plex_only] = [0.2, 0.4, 1]
    axs[0, 2].imshow(comp)
    axs[0, 2].set_title(f"green=overlap  YELLOW=ours-only {ours_only_a:.2f}  "
                        f"BLUE=PLEX-only {plex_only_a:.2f}", fontsize=10)
    axs[0, 3].imshow(np.dstack([norm8(P["hyper6"])] * 3)); axs[0, 3].set_title("hypertransmission", fontsize=10)
    for j in range(4):
        axs[0, j].axis("off")

    H = ov.H
    for k in range(4):
        if k < len(plex_only_bs):
            i = plex_only_bs[k]
            axs[1, k].imshow(norm8(ov.vol[i].astype(np.float32)), cmap="gray", aspect="auto")
            draw_spans(axs[1, k], ov.W, H, bm[i], plex_nat[i], our_nat[i], cand_nat[i])
            axs[1, k].set_title(f"PLEX-only B-scan {i}", fontsize=9)
        else:
            axs[1, k].axis("off")
        if k < len(ours_only_bs):
            i = ours_only_bs[k]
            axs[2, k].imshow(norm8(ov.vol[i].astype(np.float32)), cmap="gray", aspect="auto")
            draw_spans(axs[2, k], ov.W, H, bm[i], plex_nat[i], our_nat[i], cand_nat[i])
            axs[2, k].set_title(f"ours-only B-scan {i}", fontsize=9)
        else:
            axs[2, k].axis("off")
    axs[1, 0].set_ylabel("PLEX-only", fontsize=11)
    axs[2, 0].set_ylabel("ours-only", fontsize=11)
    fig.suptitle(f"{sv} {eye}  |  PLEX {adv:.2f} vs ours {final:.2f} mm2  |  "
                 f"ours-only {ours_only_a:.2f} + PLEX-only {plex_only_a:.2f} "
                 f"(net {ours_only_a-plex_only_a:+.2f})  Dice {dice:.2f}  |  "
                 f"red=PLEX  lime=ourGA  yellow=candidate", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(OUT, exist_ok=True)
    op = os.path.join(OUT, f"{sv}_{eye.upper()}_audit.png")
    fig.savefig(op, dpi=90); plt.close(fig)
    print(f"  wrote {op}")
    return dict(overlap=overlap_a, ours_only=ours_only_a, plex_only=plex_only_a, dice=dice)


def cohort_eyes():
    idx = json.load(open(os.path.join(LIB_DIR, "index.json")))
    out = []
    for e in idx:
        if e.get("plex_area_mm2") is None:
            continue
        sub = e["subject"]                       # e.g. NHAMD-003-001-V1
        out.append((sub, e.get("visit"), e["eye"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", nargs="?"); ap.add_argument("visit", nargs="?"); ap.add_argument("eye", nargs="?")
    ap.add_argument("--set", action="store_true")
    ap.add_argument("--cohort", action="store_true")
    ap.add_argument("--no-render", action="store_true", help="metrics only (fast cohort table)")
    a = ap.parse_args()
    if a.cohort:
        eyes = cohort_eyes()
    elif a.set or not a.subject:
        eyes = STD_SET
    else:
        eyes = [(a.subject, a.visit, a.eye)]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    spp = os.path.join(RESULTS_DIR, "ga_spatial_decomp.csv")
    gtp = os.path.join(RESULTS_DIR, "ga_gate_attribution.csv")
    with open(spp, "w", newline="") as sf, open(gtp, "w", newline="") as gf:
        sw = csv.writer(sf); gw = csv.writer(gf)
        sw.writerow(["subject", "eye", "adjudication", "plex_mm2", "ours_mm2", "overlap",
                     "ours_only", "plex_only", "dice", "net", "symmetric_disagreement"])
        gw.writerow(["subject", "eye", "plex_mm2", "candidate", "after_hyper", "after_fill", "after_size", "final"])
        for s, v, e in eyes:
            try:
                audit_eye(s, v, e, sw, gw, render=not a.no_render)
            except Exception as ex:
                import traceback; print(f"  !! {s} {v} {e}: {ex}"); traceback.print_exc()
    print(f"\nwrote {spp}\nwrote {gtp}")


if __name__ == "__main__":
    main()
