#!/usr/bin/env python
r"""Build a single, self-contained HTML explainer of how the reader computes Geographic Atrophy (GA).

Audience: a practising retina/ophthalmology specialist. The page walks through HOW the software turns a
Spectralis OCT volume into a per-location yes/no GA decision and a mm^2 area, using REAL B-scans and the
ACTUAL pipeline (reader/core/oac_ga.py: prep -> footprint); every figure is generated from the real
intermediate arrays, not hand-drawn.

Two worked eyes (the only two with a hand-corrected/validated BM):
  005 OD  focal ~1 mm^2 GA, has an in-frame manual gold label (Dice ~0.93)   -> the clean hero
  008 OS  large confluent ~15 mm^2 GA, no gold                                -> the hole-fill / large demo

Output: outputs/explain/ga_explainer.html  (images base64-inlined; loose PNGs also written for QC).

Run (from repo root):
  oct_env\Scripts\python.exe src\explain_ga.py
"""
import base64
import csv
import inspect
import io
import os
import re
import sys

# --- imports: put src/ (m3_*, qcviz, paths) and repo root (reader...) on the path, like src/oac_area.py ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes, gaussian_filter

import m3_projections as mp
import m3_slab
import qcviz as qv
from paths import DATA_DIR, OUT_DIR, RESULTS_DIR
from reader.core import e2e_source, layers, oac_ga
from reader.core import footprint as fp
from reader.core import projection as proj
from reader.core import render
from reader.core.layer_store import JsonSidecarLayerStore

CORR_DIR = os.path.join(_REPO, "reader", "data_store", "corrections")
COHORT_DIR = os.path.join(_REPO, "cohort")
OUT = os.path.join(OUT_DIR, "explain")
AX = mp.AX                                   # axial um/px (~3.87)
ENF = proj.ENFACE_MMPP                       # 6/512 mm/px en-face scale

# Every number this page quotes or draws is READ OUT OF THE DETECTOR, never retyped here. The page once
# hardcoded the OLD deep hypertransmission slab (BM+130..+250 um) and drew it on the B-scans while the
# detector had already moved to the shallow band, so the figure contradicted the algorithm it described.
_FPD = {p.name: p.default for p in inspect.signature(oac_ga.footprint).parameters.values()
        if p.default is not inspect.Parameter.empty}
FRAC = _FPD["frac"]                  # 0.50: GA candidate where smoothed RPE-loss < FRAC * healthy baseline
MIN_DIAM_UM = _FPD["min_diam_um"]    # 250 um: cRORA size rule (longest dimension of the component)
MIN_DEPTH = _FPD["min_depth"]        # 0.27: a component must reach COMPLETE loss somewhere
HYPER_ABS = _FPD["hyper_abs"]        # 0.10: absolute floor on the sub-BM transmission gate
HYPER_KEEP = _FPD["hyper_keep"]      # 0.40: relative part of the gate (fraction of the eye's own p75)
HYPER_FRAC = _FPD["hyper_frac"]      # 0.70: hole-fill threshold (fraction of the lesion's own p60)
CLOSE_MM = _FPD["close_mm"]
RPE_UM = mp.OAC_RPE_UM               # (-50, -8) um: the RPE/EZ/IZ band sampled ABOVE BM
SLAB_UM = oac_ga.OAC_HYPER_UM        # (20, 60) um BELOW BM: the SHALLOW sub-BM transmission band

# colours (RGB), kept consistent across every figure + the HTML legend
C_BM = (255, 255, 0)          # Bruch's membrane
C_RPE_BAND = (255, 60, 200)   # RPE OAC sampling band (BM-50..-8 um), matches reader BAND_DEFS
C_SLAB = (0, 180, 0)          # sub-BM hypertransmission band (BM+20..+60 um)
C_GA = (0, 255, 0)            # our GA footprint
C_ADV = (40, 110, 255)        # PLEX advRPE reference
C_CUT = (0, 230, 230)         # the en-face cut line for the profile
C_PARTIAL = (255, 150, 0)     # components dropped for partial (incomplete) RPE loss

EYES = [
    {"tag": "005OD", "subject": "NHAMD-003-005", "visit": "V3", "eye": "OD",
     "role": "hero", "label": "005 OD, focal GA"},
    {"tag": "008OS", "subject": "NHAMD-003-008", "visit": "V1", "eye": "OS",
     "role": "demo", "label": "008 OS, large confluent GA"},
]


# ------------------------------------------------------------------ data loading (mirrors src/oac_area.py)
def resolve(subject, visit, eye):
    """(row dict, absolute E2E path) from results/bm_worklist.csv. subject may carry a trailing -V<n>."""
    want = subject if re.search(r"-V\d+$", subject) else f"{subject}-{visit}"
    eye = eye.upper()
    path = os.path.join(RESULTS_DIR, "bm_worklist.csv")
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["subject"] == want and r["eye"].upper() == eye:
                return r, os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
    raise SystemExit(f"{want} {eye} not found in {path}")


def load_gold_native(subject, eye, n_bscans):
    """In-frame manual GA columns (label==1) as a native (n,W) float map, or None. (Only 005 OD has it.)"""
    lab = os.path.join(OUT_DIR, "ga_bscan_dataset", "labels")
    rows, found, W = [], False, None
    for i in range(n_bscans):
        p = os.path.join(lab, f"{subject}_{eye}_b{i:04d}.png")
        if os.path.exists(p):
            found = True
            col = (np.array(Image.open(p)) == 1).any(axis=0)
            W = len(col)
            rows.append(col)
        else:
            rows.append(None)
    if not found:
        return None
    return np.array([(r if r is not None else np.zeros(W, bool)) for r in rows], np.float32)


def load_eye(spec):
    """Open the eye's 6x6 volume, fold in the corrected BM, run the REAL detector, gather references."""
    row, e2e = resolve(spec["subject"], spec["visit"], spec["eye"])
    subject, eye = row["subject"], row["eye"].upper()
    adv = float(row["advRPE_area_mm2"])
    if not os.path.exists(e2e):
        raise SystemExit(f"E2E missing: {e2e}")

    raw = e2e_source.open_e2e(e2e)
    idx = e2e_source.default_volume_index(raw, eye)
    ov = e2e_source.load_volume(raw, idx)
    lstore = JsonSidecarLayerStore(CORR_DIR)
    _ilm, bm = layers.effective_surfaces(ov, lstore)
    corrected = lstore.corrected_indices(ov.eid, ov.eye)

    P = oac_ga.prep(ov, bm)                                   # rpe6, rpe_nat, loss6, hyper6, core, base, g_base
    mask, area = oac_ga.footprint(P, FRAC, MIN_DIAM_UM)       # the EXACT production call

    gold_native = load_gold_native(subject, eye, ov.n_bscans)
    gold_mask = (proj.to_enface(gold_native, ov.fov_mm) > 0.5) if gold_native is not None else None
    gold_area = float(gold_mask.sum()) * oac_ga.MMPP2 if gold_mask is not None else None
    dice = None
    if gold_mask is not None:
        dice = 2 * float((mask & gold_mask).sum()) / (float(mask.sum()) + float(gold_mask.sum()) + 1e-9)

    loc = e2e_source.localizer_image(raw, eye)
    print(f"{subject} {eye}: vol_idx={idx} n={ov.n_bscans} H={ov.H} W={ov.W} "
          f"fov={tuple(round(f,2) for f in ov.fov_mm)} bm_src={ov.bm_src} corrected={len(corrected)} "
          f"-> area={area:.3f} advRPE={adv:.3f}" + (f" gold={gold_area:.3f} Dice={dice:.3f}" if dice else ""))

    return dict(spec, ov=ov, bm=bm, P=P, mask=mask, area=area, adv=adv,
                gold_mask=gold_mask, gold_area=gold_area, dice=dice, loc=loc,
                subject=subject, eye=eye, fov=ov.fov_mm, n=ov.n_bscans, W=ov.W, corrected=len(corrected))


# ------------------------------------------------------------------ B-scan selection
def pick_indices(d):
    """Auto-pick a lesion B-scan (through the most GA) and a clearly-healthy B-scan (intact RPE, off-lesion).

    En-face is row-flipped to fundus orientation, so en-face row r maps to B-scan idx = (n-1) - r/(H-1)*(n-1).
    """
    ov, mask, P = d["ov"], d["mask"], d["P"]
    n, Henf = d["n"], mask.shape[0]
    row_counts = mask.sum(axis=1)
    r_star = int(np.argmax(row_counts)) if mask.any() else Henf // 2
    lesion_idx = int(np.clip(round((n - 1) - r_star / (Henf - 1) * (n - 1)), 0, n - 1))

    def ga_count(i):                                          # GA pixels on this B-scan's mapped en-face row
        r = int(round((n - 1 - i) / (n - 1) * (Henf - 1)))
        return int(mask[max(0, r - 1):min(Henf, r + 2)].sum())

    rpe_nat = P["rpe_nat"]                                    # native RPE-loss OAC: high = RPE present = healthy
    interior = range(int(n * 0.15), int(n * 0.85) + 1)
    clean = [i for i in interior if ga_count(i) == 0] or list(interior)
    healthy_idx = max(clean, key=lambda i: float(np.nanmedian(rpe_nat[i])))
    return healthy_idx, lesion_idx, r_star


# ------------------------------------------------------------------ figure helpers
def _crop_window(bm_row, H):
    lo = max(0, int(np.nanmin(bm_row)) - 70)
    hi = min(H, int(np.nanmax(bm_row) + max(250.0, SLAB_UM[1] + 60.0) / AX) + 25)
    return lo, hi


def _stretch(img, factor=3):
    return cv2.resize(img, (img.shape[1], img.shape[0] * factor), interpolation=cv2.INTER_NEAREST)


def fig_bscan_anatomy(bscan, bm_row, title):
    """Raw B-scan with BM (yellow), the RPE OAC band (magenta) and the hypertransmission slab (green)."""
    rgb = qv.ensure_rgb(qv.norm8(bscan))
    H, W = bscan.shape[:2]
    band = rgb.copy()
    for x in range(W):
        b = bm_row[x]
        if not np.isfinite(b):
            continue
        y0, y1 = int(round(b + RPE_UM[0] / AX)), int(round(b + RPE_UM[1] / AX))
        if y1 > y0:
            band[max(0, y0):min(H, y1), x] = C_RPE_BAND
        y2, y3 = int(round(b + SLAB_UM[0] / AX)), int(round(b + SLAB_UM[1] / AX))
        if y3 > y2:
            band[max(0, y2):min(H, y3), x] = C_SLAB
    rgb = cv2.addWeighted(rgb, 0.6, band, 0.4, 0)             # blend (identity outside the painted bands)
    for x in range(W):
        yb = int(round(bm_row[x]))
        if 0 <= yb < H:
            rgb[max(0, yb - 1):yb + 1, x] = C_BM
    lo, hi = _crop_window(bm_row, H)
    return qv.add_title(_stretch(rgb[lo:hi]), title)


def fig_oac_heatmap(oac_slice, bm_row, title):
    """Same slice as an OAC heatmap (inferno): the RPE shows as a bright attenuation band that drops out in GA."""
    H, W = oac_slice.shape
    lo, hi = _crop_window(bm_row, H)
    u = qv.norm8(oac_slice[lo:hi])
    cm = cv2.cvtColor(cv2.applyColorMap(u, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    cm = _stretch(cm)
    for x in range(W):
        b = bm_row[x]
        yb = int(round((b - lo) * 3))
        if 0 <= yb < cm.shape[0]:
            cm[yb:yb + 1, x] = C_BM
        for off in (RPE_UM[0], RPE_UM[1]):                   # thin RPE-band outline so the heatmap shows through
            yr = int(round((b + off / AX - lo) * 3))
            if 0 <= yr < cm.shape[0]:
                cm[yr:yr + 1, x] = C_RPE_BAND
    return qv.add_title(cm, title)


def _overlay_array(rpe6, mask):
    """RPE-loss en-face (dark=GA) + translucent green GA fill + contour (matches oac_ga.overlay_png)."""
    rgb = qv.ensure_rgb(qv.norm8(np.nan_to_num(rpe6))).astype(np.float32)
    m = np.asarray(mask, bool)
    rgb[m] = 0.45 * rgb[m] + 0.55 * np.array(C_GA, np.float32)
    return qv.draw_contour(rgb.astype(np.uint8), m, color=C_GA, thick=1)


def _fill(rgb, region, color, a=0.55):
    out = rgb.astype(np.float32)
    r = np.asarray(region, bool)
    out[r] = (1 - a) * out[r] + a * np.array(color, np.float32)
    return out.astype(np.uint8)


def fig_enface_localizer(d, lesion_idx):
    """The RPE-loss en-face (with the profile cut line) beside the IR localizer (with the B-scan line)."""
    rpe6, n = d["P"]["rpe6"], d["n"]
    enf = qv.ensure_rgb(qv.norm8(rpe6))
    Henf = enf.shape[0]
    r = int(round((n - 1 - lesion_idx) / (n - 1) * (Henf - 1)))
    cv2.line(enf, (0, r), (enf.shape[1] - 1, r), C_CUT, 1, cv2.LINE_AA)
    tiles = [enf]
    titles = ["OAC RPE-loss en-face (dark = GA)"]
    bar_on = [True]
    if d["loc"] is not None:
        loc = qv.ensure_rgb(qv.norm8(d["loc"]))
        scale = Henf / loc.shape[0]
        loc = cv2.resize(loc, (max(1, int(loc.shape[1] * scale)), Henf))
        yl = int(round((n - 1 - lesion_idx) / (n - 1) * (Henf - 1)))
        cv2.line(loc, (0, yl), (loc.shape[1] - 1, yl), C_GA, 2, cv2.LINE_AA)
        tiles.append(loc)
        titles.append("IR localizer (B-scan line)")
        bar_on.append(False)
    return qv.panel(tiles, titles, mm_per_px=ENF, bar_on=bar_on)


def fig_profile(d, r_star):
    """matplotlib: along the cut row through the lesion, measured OAC vs the fitted healthy baseline vs the
    50%-of-baseline cutoff, GA segment shaded. This is the per-location decision."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P, mask = d["P"], d["mask"]
    loss = P["loss6"][r_star]
    base = P["base"][r_star]
    core = P["core"][r_star]
    cols = np.where(core)[0]
    if cols.size < 5:
        cols = np.arange(P["loss6"].shape[1])
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    xmm = (np.arange(x0, x1)) * ENF
    loss_c, base_c, core_c = loss[x0:x1], base[x0:x1], core[x0:x1]
    thr_c = FRAC * base_c
    ga_here = (loss_c < thr_c) & core_c

    fig, ax = plt.subplots(figsize=(9.2, 3.4), dpi=130)
    ax.plot(xmm, base_c, "--", color="#1f77b4", lw=2.0, label="healthy-RPE baseline (fitted, lesion excluded)")
    ax.plot(xmm, thr_c, ":", color="#d62728", lw=2.0, label="50% of baseline = GA cutoff")
    ax.plot(xmm, loss_c, "-", color="#111111", lw=2.2, label="measured RPE-loss OAC (this row)")
    ax.fill_between(xmm, 0, np.nanmax(base_c) * 1.05, where=ga_here, color="#2ca02c", alpha=0.22,
                    step="mid", label="GA called here")
    # off-field / rim / vignette (not measured) hatched grey
    ax.fill_between(xmm, 0, np.nanmax(base_c) * 1.05, where=~core_c, color="#999999", alpha=0.25, step="mid",
                    hatch="//", label="not measured (rim / vignette)")
    ax.set_xlim(xmm.min(), xmm.max())
    ax.set_ylim(0, float(np.nanmax(base_c)) * 1.08)
    ax.set_xlabel("position across the macula (mm)")
    ax.set_ylabel("OAC (relative units)")
    ax.set_title(f"Per-location GA decision along one cut through the lesion: {d['label']}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def fig_projection(d):
    """The projection step made visible: the raw per-A-scan map as acquired, then the same map flipped and
    resampled to square pixels. Returns (native_strip, enface_square)."""
    P, ov = d["P"], d["ov"]
    nat = qv.ensure_rgb(qv.norm8(np.nan_to_num(P["rpe_nat"])))          # (n_bscans, W), one value per A-scan
    n, W = nat.shape[:2]
    fast_um = d["fov"][0] / W * 1000.0
    slow_um = d["fov"][1] / n * 1000.0
    nat_big = cv2.resize(nat, (W, n * 3), interpolation=cv2.INTER_NEAREST)   # 3x tall, else it is a sliver
    # The in-image title must fit the image width (qv.add_title clips it); the spacing figures that do not
    # fit are carried by the HTML caption instead.
    del fast_um, slow_um
    left = qv.add_title(nat_big, f"as acquired: {n} rows x {W} columns")
    enf = qv.ensure_rgb(qv.norm8(np.nan_to_num(P["rpe6"])))
    right = qv.add_title(enf, f"projected: square {ENF * 1000:.0f} um pixels")
    return left, right


MARGIN_MM = inspect.signature(oac_ga.prep).parameters["margin_mm"].default   # 0.30 mm low-SNR rim


def fig_field(d):
    """Which pixels are measured at all. Splits the exclusion into its two causes, because on a clean eye the
    vignette gate removes nothing and a caption claiming otherwise would be false."""
    P = d["P"]
    rpe6, core = P["rpe6"], P["core"]
    valid = gaussian_filter((rpe6 > 1e-6).astype(np.float32), 2.0) > 0.5
    after_rim = binary_erosion(valid, iterations=int(round(MARGIN_MM / ENF)))
    vignette = after_rim & ~core                       # dropped by the signal gate, not by the rim

    rgb = qv.ensure_rgb(qv.norm8(np.nan_to_num(rpe6)))
    out = _fill(rgb, ~core, C_PARTIAL, a=0.45)
    out = qv.draw_contour(out, core, color=(255, 255, 255), thick=1)

    tot = float(valid.sum())
    d["field_rim_pct"] = 100.0 * float((valid & ~after_rim).sum()) / tot
    d["field_vig_pct"] = 100.0 * float(vignette.sum()) / tot
    # A handful of stray pixels is not "a vignetted corner"; only claim one when it is visible.
    d["field_note"] = (f"the {MARGIN_MM:.2f} mm rim ({d['field_rim_pct']:.0f}% of the field)"
                       + (f", and the corners where the scan vignettes into noise "
                          f"({d['field_vig_pct']:.1f}%)" if d["field_vig_pct"] >= 0.05 else
                          ". This eye has no vignetted corner, so the signal gate removes nothing here"))
    return qv.add_title(out, f"orange = not measured ({100.0 * (1 - core.sum() / tot):.0f}% of the field)")


def _footprint_stages(P):
    """Expose the canonical production stages using the explainer's introspected constants."""
    s = oac_ga.footprint_stages(
        P, frac=FRAC, min_diam_um=MIN_DIAM_UM, hyper_fill=True, close_mm=CLOSE_MM,
        hyper_frac=HYPER_FRAC, hyper_keep=HYPER_KEEP, fill_all_holes=True,
        hyper_abs=HYPER_ABS, min_depth=MIN_DEPTH,
    )
    # Historical drawing names are retained locally; the detector logic now has one implementation.
    return dict(b0=s["rpe_candidate"], b1=s["hyper_kept"], rejected=s["hyper_rejected"],
                holes_filled=s["holes_filled"], filled_all=s["filled"], sized=s["sized"],
                partial=s["partial_rejected"], final=s["final"])


def fig_hyper(d):
    """3 tiles: the hypertransmission map; criterion-1 corner rejection; criterion-2 hole-fill + final."""
    P = d["P"]
    s = _footprint_stages(P)
    rpe6 = P["rpe6"]
    base_rgb = qv.ensure_rgb(qv.norm8(rpe6))

    hyp = qv.ensure_rgb(render.windowed(gaussian_filter(P["hyper6"], 0), (
        float(np.nanpercentile(P["hyper6"][P["core"]], 5)),
        float(np.nanpercentile(P["hyper6"][P["core"]], 99)))))

    t_corner = qv.draw_contour(_fill(base_rgb, s["rejected"], C_ADV), s["b0"], color=(255, 255, 255), thick=1)
    t_hole = _fill(base_rgb, s["holes_filled"], (0, 230, 230))
    t_hole = _fill(t_hole, s["partial"], C_PARTIAL)
    t_hole = qv.draw_contour(t_hole, s["final"], color=C_GA, thick=1)
    return qv.panel(
        [hyp, t_corner, t_hole],
        # Titles are clipped at the tile width, so keep each under ~52 characters.
        ["transmission: bright = light reaches choroid",
         f"blue = rejected, no transmission ({int(s['rejected'].sum())} px)",
         f"cyan = holes filled | orange = partial loss dropped"],
        mm_per_px=ENF)


def _adv_reference_tile(d):
    """PLEX advRPE substrate (grayscale) with the advRPE GA mask as a blue contour."""
    base = os.path.join(COHORT_DIR, d["subject"], d["eye"])
    sub_p = os.path.join(base, "advrpe_subrpe_enface.png")
    mask_p = os.path.join(base, "ga_mask.png")
    if not (os.path.exists(sub_p) and os.path.exists(mask_p)):
        return None
    sub = np.array(Image.open(sub_p).convert("L"))
    gm = np.array(Image.open(mask_p).convert("L")) > 127
    rgb = qv.ensure_rgb(qv.norm8(sub))
    return qv.draw_contour(rgb, gm, color=C_ADV, thick=2)


def fig_final(d):
    """Our footprint + area beside the PLEX advRPE reference (+ the gold FP/FN tile + Dice when available)."""
    rpe6, mask = d["P"]["rpe6"], d["mask"]
    ours = _overlay_array(rpe6, mask)
    tiles, titles, bar_on = [ours], [f"our OCT-only GA: {d['area']:.2f} mm²"], [True]
    adv = _adv_reference_tile(d)
    if adv is not None:
        tiles.append(adv)
        titles.append(f"PLEX advRPE reference: {d['adv']:.2f} mm²")
        bar_on.append(False)
    if d["gold_mask"] is not None:
        g = d["gold_mask"]
        fpfn = (qv.norm8(np.nan_to_num(rpe6)).astype(np.float32) * 0.45).astype(np.uint8)
        fpfn = qv.ensure_rgb(fpfn)
        fpfn[mask & g] = (0, 230, 0)
        fpfn[mask & ~g] = (255, 40, 40)
        fpfn[~mask & g] = (40, 110, 255)
        tiles.append(fpfn)
        titles.append(f"vs gold: hit=green FP=red FN=blue  Dice {d['dice']:.2f}")
        bar_on.append(True)
    return qv.panel(tiles, titles, mm_per_px=ENF, bar_on=bar_on)


# ------------------------------------------------------------------ build all figures + encode
def enc(x):
    b = x if isinstance(x, (bytes, bytearray)) else render.to_png(x)
    return base64.b64encode(b).decode("ascii")


def worked_example(d):
    """Two REAL locations from this eye, run through the decision exactly as the detector does.

    Returns the actual numbers at one healthy A-scan and one A-scan in the middle of the lesion, so the
    reader can follow the arithmetic instead of taking the prose on trust. Nothing here is invented: every
    value is read out of the same arrays the detector thresholds.
    """
    P, mask = d["P"], d["mask"]
    loss, base, core, h = P["loss6"], P["base"], P["core"], P["hyper6"]
    s = _footprint_stages(P)

    keep_thr = max(HYPER_KEEP * float(np.percentile(h[core], 75)), HYPER_ABS)
    ratio = loss / np.maximum(base, 1e-6)

    # Lesion point: the deepest RPE loss inside the final footprint. Healthy point: the median-brightest
    # core pixel well outside the lesion (dilate the mask so we do not sit on its shoulder). Rejected
    # point: the darkest-transmitting pixel that LOOKED like RPE loss but was thrown out by the
    # transmission gate -- without it, the table implies transmission is what separates healthy from
    # atrophic, when in fact healthy macula transmits plenty and the gate only kills the dead corners.
    from scipy.ndimage import binary_dilation
    far = core & ~binary_dilation(mask, iterations=int(round(0.5 / ENF)))
    if not mask.any() or not far.any():
        return None
    ly, lx = np.unravel_index(np.argmin(np.where(mask, ratio, np.inf)), ratio.shape)
    cand = np.where(far)
    j = int(np.argmin(np.abs(ratio[cand] - float(np.median(ratio[far])))))
    hy, hx = int(cand[0][j]), int(cand[1][j])

    def row(name, y, x):
        r = float(ratio[y, x])
        return {
            "name": name,
            "loss": float(loss[y, x]), "base": float(base[y, x]), "ratio": r,
            "below_half": r < FRAC,
            "hyper": float(h[y, x]), "transmits": float(h[y, x]) > keep_thr,
            "ga": bool(mask[y, x]),
        }

    rows = [row("healthy macula", hy, hx)]
    if s["rejected"].any():
        ry, rx = np.unravel_index(np.argmin(np.where(s["rejected"], h, np.inf)), h.shape)
        rows.append(row("dark corner of the field", int(ry), int(rx)))
    rows.append(row("centre of the lesion", ly, lx))
    return {"keep_thr": keep_thr, "frac": FRAC, "min_depth": MIN_DEPTH,
            "n_partial": int(s["partial"].sum()), "rows": rows}


def build_figures(d):
    healthy_idx, lesion_idx, r_star = pick_indices(d)
    ov = d["ov"]
    oac = mp.oac_volume(ov.vol)

    # The illustrated stages must BE the pipeline, not a lookalike of it.
    stages = _footprint_stages(d["P"])
    if not np.array_equal(stages["final"], d["mask"]):
        raise SystemExit("explain_ga: _footprint_stages has drifted from oac_ga.footprint; fix before publishing")
    d["example"] = worked_example(d)
    d["n_rejected"] = int(stages["rejected"].sum())

    imgs = {}
    # always
    imgs["teaser"] = _overlay_array(d["P"]["rpe6"], d["mask"])
    imgs["anat_lesion"] = fig_bscan_anatomy(ov.vol[lesion_idx], d["bm"][lesion_idx],
                                            f"GA lesion B-scan (#{lesion_idx})")
    imgs["hyper"] = fig_hyper(d)
    imgs["final"] = fig_final(d)
    if d["role"] == "hero":
        imgs["anat_healthy"] = fig_bscan_anatomy(ov.vol[healthy_idx], d["bm"][healthy_idx],
                                                 f"healthy macula B-scan (#{healthy_idx})")
        imgs["oac_healthy"] = fig_oac_heatmap(oac[healthy_idx], d["bm"][healthy_idx],
                                              f"OAC, healthy (#{healthy_idx}): bright RPE band")
        imgs["oac_lesion"] = fig_oac_heatmap(oac[lesion_idx], d["bm"][lesion_idx],
                                             f"OAC, lesion (#{lesion_idx}): RPE band gone")
        imgs["proj_native"], imgs["proj_enface"] = fig_projection(d)
        imgs["field"] = fig_field(d)
        imgs["enface"] = fig_enface_localizer(d, lesion_idx)
        imgs["profile"] = fig_profile(d, r_star)
    d["_idx"] = (healthy_idx, lesion_idx, r_star)
    return {k: enc(v) for k, v in imgs.items()}


# ------------------------------------------------------------------ HTML
def img_tag(b64, cls="fig"):
    return f'<img class="{cls}" src="data:image/png;base64,{b64}" />'


CSS = """
:root{--ink:#1a1a1a;--mut:#5b6470;--line:#e2e6ea;--accent:#0b6;--bg:#fafbfc;}
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
  line-height:1.6;margin:0;background:var(--bg)}
.wrap{max-width:960px;margin:0 auto;padding:32px 22px 80px}
h1{font-size:30px;margin:.2em 0 .1em;line-height:1.2}
.sub{color:var(--mut);font-size:16px;margin:0 0 8px}
h2{font-size:22px;margin:1.8em 0 .2em;padding-top:.4em;border-top:2px solid var(--line)}
.step{display:inline-block;background:var(--accent);color:#fff;border-radius:6px;font-size:13px;
  font-weight:700;padding:1px 9px;margin-right:8px;vertical-align:middle}
p{margin:.5em 0}
.fig{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#000;margin:.6em 0}
.cap{color:var(--mut);font-size:13.5px;margin:.1em 0 1.3em}
.key{display:flex;flex-wrap:wrap;gap:14px;background:#fff;border:1px solid var(--line);border-radius:10px;
  padding:12px 16px;margin:14px 0;font-size:14px}
.key b{font-weight:600}
.sw{display:inline-block;width:14px;height:14px;border-radius:3px;margin-right:6px;vertical-align:-2px;
  border:1px solid #0003}
.callout{background:#eef7f1;border:1px solid #bfe6cf;border-left:4px solid var(--accent);
  border-radius:8px;padding:12px 16px;margin:14px 0}
.warn{background:#fff7ed;border:1px solid #f3d3a8;border-left:4px solid #e08a2b;border-radius:8px;
  padding:12px 16px;margin:14px 0;font-size:14.5px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:680px){.grid2{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14.5px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left}
th{background:#f1f4f7}
.num{font-variant-numeric:tabular-nums;font-weight:700}
/* decision flow */
.flow{margin:18px 0}
.frow{display:flex;align-items:center;gap:12px;margin:8px 0;flex-wrap:wrap}
.node{border:1.5px solid #c7ccd2;border-radius:10px;padding:8px 12px;background:#fff;font-size:14px;
  min-width:150px}
.dec{border-color:#0b6;background:#f3fbf6;font-weight:600}
.term-no{border-color:#c3c9cf;background:#f3f4f6;color:#5b6470}
.term-ga{border-color:#0b6;background:#e7f7ee;font-weight:700}
.arrow{color:#9aa3ab;font-size:20px}
.lblno{color:#b23;font-size:12.5px;font-weight:700}
.lblyes{color:#0a7a45;font-size:12.5px;font-weight:700}
footer{color:var(--mut);font-size:12.5px;margin-top:40px;border-top:1px solid var(--line);padding-top:14px}
code{background:#eef1f4;padding:1px 5px;border-radius:4px;font-size:13px}
"""


def legend_block():
    def sw(c):
        return f'<span class="sw" style="background:rgb{c}"></span>'
    return f"""<div class="key">
  <span>{sw(C_BM)}<b>BM</b> (Bruch's membrane)</span>
  <span>{sw(C_RPE_BAND)}<b>RPE band</b>, where attenuation is read ({RPE_UM[0]} to {RPE_UM[1]}&nbsp;µm above BM)</span>
  <span>{sw(C_SLAB)}<b>transmission band</b> (+{SLAB_UM[0]:.0f} to +{SLAB_UM[1]:.0f}&nbsp;µm below BM)</span>
  <span>{sw(C_GA)}<b>our GA footprint</b></span>
  <span>{sw(C_ADV)}<b>PLEX advRPE reference</b></span>
</div>"""


def example_table(d):
    """The decision arithmetic at two real locations in this eye. Every number comes from the detector."""
    ex = d.get("example")
    if not ex:
        return ""
    yes = '<b style="color:#0a7d43">yes</b>'
    no = '<b style="color:#b04a00">no</b>'
    verdict_ga = '<b style="color:#0a7d43">GA</b>'
    verdict_no = "<b>not GA</b>"

    def tr(r):
        return (f"<tr><td>{r['name']}</td>"
                f"<td class='num'>{r['loss']:.3f}</td>"
                f"<td class='num'>{r['base']:.3f}</td>"
                f"<td class='num'>{r['ratio'] * 100:.0f}%</td>"
                f"<td>{yes if r['below_half'] else no}</td>"
                f"<td class='num'>{r['hyper']:.3f}</td>"
                f"<td>{yes if r['transmits'] else no}</td>"
                f"<td>{verdict_ga if r['ga'] else verdict_no}</td></tr>")

    rows = "".join(tr(r) for r in ex["rows"])
    has_corner = any(r["name"].startswith("dark corner") for r in ex["rows"])
    corner = ("It exists to throw out the middle row: the dark corner of the field, where there is no retina at "
              "all. That corner reads as low attenuation and would otherwise be called atrophy, but it transmits "
              "almost nothing, so it is rejected."
              if has_corner else
              "It exists to throw out the dark corners of the field, where there is no retina at all. Such a "
              "corner reads as low attenuation and would otherwise be called atrophy, but it transmits almost "
              "nothing. This particular eye has no such corner, so here the rule changes nothing.")
    return f"""<table>
<tr><th>location in {d['label']}</th><th>attenuation<br/>measured</th><th>healthy<br/>baseline</th>
    <th>% of<br/>baseline</th><th>below<br/>{ex['frac'] * 100:.0f}%?</th><th>transmission<br/>measured</th>
    <th>above<br/>{ex['keep_thr']:.3f}?</th><th>verdict</th></tr>
{rows}
</table>
<p class="sub" style="margin-top:6px">Read the healthy row first. Its attenuation sits close to the baseline the
software fitted for that spot, so it is nowhere near the halfway cutoff and the RPE is judged present. In the
lesion the attenuation has collapsed to a small fraction of that same baseline, and far more light is reaching
the choroid. Both criteria agree, so the location counts as GA.</p>
<div class="callout"><b>Note what the transmission column is, and is not.</b> Healthy macula transmits a good deal
of light too, and it comfortably clears the cutoff. Transmission is <i>not</i> what tells atrophy apart from
healthy retina; the attenuation column does that. {corner} The cutoff shown here
({ex['keep_thr']:.3f}) is derived from this eye's own tissue, but it can never fall below a fixed floor of
{HYPER_ABS:.2f}: on a healthy eye with no lesion, a purely relative cutoff would sink until image noise crept
over it and was reported as GA.</div>"""


def stage_table(eyes):
    """Pixels and mm2 surviving each stage, for every worked eye. The single clearest example on the page:
    it shows which rules actually move the number and which are quiet guards."""
    names = [
        ("b0", f"candidates: below {FRAC * 100:.0f}% of the healthy baseline"),
        ("b1", "after requiring transmission"),
        ("filled", "after filling interior holes that transmit"),
        ("filled_all", "after closing every remaining enclosed hole"),
        ("sized", f"after the {MIN_DIAM_UM:.0f} µm size rule"),
        ("final", f"after the complete-loss rule (below {MIN_DEPTH * 100:.0f}%)"),
    ]
    st = {}
    for d in eyes:
        s = _footprint_stages(d["P"])
        st[d["tag"]] = {"b0": int(s["b0"].sum()), "b1": int(s["b1"].sum()),
                        "filled": int((s["b1"] | s["holes_filled"]).sum()),
                        "filled_all": int(s["filled_all"].sum()),
                        "sized": int(s["sized"].sum()), "final": int(s["final"].sum())}

    head = "".join(f"<th colspan='2'>{d['label']}</th>" for d in eyes)
    sub = "".join("<th class='num'>pixels</th><th class='num'>mm²</th>" for _ in eyes)
    rows = ""
    for key, label in names:
        cells = ""
        for d in eyes:
            px = st[d["tag"]][key]
            cells += f"<td class='num'>{px:,}</td><td class='num'>{px * ENF * ENF:.2f}</td>"
        strong = " style='font-weight:600'" if key == "final" else ""
        rows += f"<tr{strong}><td>{label}</td>{cells}</tr>"
    table = f"<table><tr><th>stage</th>{head}</tr><tr><th></th>{sub}</tr>{rows}</table>"

    # The commentary is generated from the same counts, so it can never contradict the table above it.
    def mm2(tag, a, b):
        return (st[tag][b] - st[tag][a]) * ENF * ENF

    hero, demo = eyes[0]["tag"], eyes[1]["tag"]
    rej = {d["tag"]: st[d["tag"]]["b0"] - st[d["tag"]]["b1"] for d in eyes}
    notes = f"""<p>This table is worth a moment, because it shows which rules do the work and which are quiet
guards.</p>
<ul>
<li><b>The baseline comparison does nearly all of it.</b> Everything after the first row moves the focal eye by
{abs(mm2(hero, 'b0', 'final')):.2f}&nbsp;mm&sup2; and the large eye by
{abs(mm2(demo, 'b0', 'final')):.2f}&nbsp;mm&sup2;. The question &ldquo;has the attenuation collapsed to less than
half of the healthy baseline?&rdquo; is what finds the atrophy.</li>
<li><b>The transmission requirement removes almost nothing here</b>: {rej[hero]:,} pixels on the focal eye and
{rej[demo]:,} on the large one. That is not a failure. Both eyes genuinely have atrophy, and atrophy transmits.
The rule exists to protect eyes that do <i>not</i> have atrophy, where a dark corner of the field would otherwise
be reported as a lesion.</li>
<li><b>Hole filling and the complete-loss rule pull in opposite directions on the large eye.</b> Closing the
interior gaps inside the confluent lesion adds {mm2(demo, 'b1', 'filled_all'):.2f}&nbsp;mm&sup2;; the
complete-loss rule then removes {abs(mm2(demo, 'sized', 'final')):.2f}&nbsp;mm&sup2; of satellite patches whose
RPE was only thinned, never truly lost. On the focal eye the same two steps cancel: closing the gaps creates a
handful of small islands, and the size rule deletes every one of them, so the reported area is exactly what the
baseline comparison produced.</li>
</ul>"""
    return table + notes


def decision_flow():
    A = '<span class="arrow">&rarr;</span>'
    return f"""<div class="flow">
  <div class="frow"><div class="node">A-scan location in the macular field</div>{A}
    <div class="node dec">inside the measurement field?</div>
    <span class="lblno">no &rarr;</span><div class="node term-no">not measured (edge rim / vignette)</div></div>
  <div class="frow"><span class="lblyes">yes &darr;</span></div>
  <div class="frow"><div class="node dec">mean attenuation in the RPE band &lt; {FRAC * 100:.0f}% of the local healthy baseline?</div>
    <span class="lblno">no &rarr;</span><div class="node term-no">RPE present &rarr; GA-free</div></div>
  <div class="frow"><span class="lblyes">yes &darr; (RPE atrophic)</span></div>
  <div class="frow"><div class="node dec">does light reach the choroid just under BM (hypertransmission)?</div>
    <span class="lblno">no &rarr;</span><div class="node term-no">no transmission &rarr; reject (corner / artefact)</div></div>
  <div class="frow"><span class="lblyes">yes &darr;</span></div>
  <div class="frow"><div class="node">GA candidate pixel
    <div style="font-size:12.5px;color:#5b6470">interior holes that transmit are filled; non-transmitting holes kept (RPE sparing)</div></div></div>
  <div class="frow"><div class="node dec">does this patch reach below {MIN_DEPTH * 100:.0f}% of baseline somewhere?</div>
    <span class="lblno">no &rarr;</span><div class="node term-no">partial loss, not complete atrophy &rarr; discard</div></div>
  <div class="frow"><span class="lblyes">yes &darr;</span></div>
  <div class="frow"><div class="node dec">is the patch at least {MIN_DIAM_UM:.0f}&nbsp;µm along its longest axis (cRORA)?</div>
    <span class="lblno">no &rarr;</span><div class="node term-no">too small &rarr; discard</div></div>
  <div class="frow"><span class="lblyes">yes &darr;</span>
    <div class="node term-ga">GA. Counts toward area = pixels &times; (pixel size)&sup2;</div></div>
</div>"""


def build_html(eyes):
    hero = next(e for e in eyes if e["role"] == "hero")
    demo = next(e for e in eyes if e["role"] == "demo")
    HI = hero["imgs"]
    DI = demo["imgs"]
    hf = f"{hero['fov'][0]:.1f}×{hero['fov'][1]:.1f} mm"

    def fig(b64, cap):
        return f'{img_tag(b64)}<div class="cap">{cap}</div>'

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>How the software measures Geographic Atrophy from OCT</title>
<style>{CSS}</style></head>
<body><div class="wrap">

<h1>How the software measures Geographic Atrophy from an OCT scan</h1>
<p class="sub">A visual walk-through for clinicians: from a stack of B-scans to a yes or no GA decision at every
location, and a total area in mm&sup2;. Every image below is produced by the actual algorithm on a real
patient scan.</p>

{legend_block()}

<p>The scan is a Spectralis macular cube (here {hf}, {hero['n']} B-scans). The software asks a single question at
<b>each A-scan location</b>: <i>is the RPE atrophic, with light passing through into the choroid beneath (the
cRORA definition)?</i> It then adds up the &ldquo;yes&rdquo; locations into an area. Here is the final answer for
our hero example, before we explain how it gets there:</p>

{fig(HI['teaser'], f"<b>{hero['label']}.</b> Face-on map of RPE loss (dark = RPE gone). Green = the GA the software measured ({hero['area']:.2f} mm²). The rest of the page explains every step that produced this.")}

<h2><span class="step">1</span>Why not simply read brightness? Optical attenuation instead</h2>
<p>Raw sub-RPE brightness on OCT depends on the choroid, on the illumination, and on the device gain. It is
not a clean readout of &ldquo;is the RPE there.&rdquo; Instead, each voxel is converted to its <b>optical
attenuation coefficient</b> (Vermeer 2014), which compares the light scattered back at that depth against all
the light still travelling deeper:</p>

<div class="callout" style="font-family:ui-monospace,monospace;font-size:14px">
attenuation at depth z &nbsp;=&nbsp; signal(z) &nbsp;&divide;&nbsp; ( 2 &times; sum of all signal below z )
</div>

<p>Because the denominator is everything below, the measure does not care how bright the image is overall, nor
how reflective this particular choroid happens to be. That matters: a naturally dark choroid makes the sub-RPE
region look empty even when the RPE is intact, which is exactly the trap a raw-brightness method falls into.
The intact RPE is a strong attenuator, so on an attenuation image it becomes a crisp <b>bright band</b>, and
where the RPE is gone that band simply disappears.</p>

<div class="grid2">
  <div>{img_tag(HI['oac_healthy'])}</div>
  <div>{img_tag(HI['oac_lesion'])}</div>
</div>
<div class="cap">Same eye, two real B-scans rendered as attenuation (inferno; brighter = more attenuation). Left:
healthy macula, a continuous bright RPE band. Right: through the lesion, the bright band drops out exactly
where the RPE is atrophic. The thin magenta lines mark the band the software samples (just above BM).</div>

<h2><span class="step">2</span>One number per A-scan</h2>
<p>Bruch's membrane (BM) is the anchor, because it is still there after the RPE above it has gone. For every
A-scan the software takes the <b>mean attenuation in a thin band just above BM</b> ({abs(RPE_UM[0])} to
{abs(RPE_UM[1])}&nbsp;&micro;m above it, spanning the EZ, IZ and RPE). A high value means the RPE is present; a
low value suggests it is missing. It uses the <i>mean</i> and not the peak, because a single bright speckle
would flip a peak reading back to &ldquo;RPE present.&rdquo;</p>

<div class="grid2">
  <div>{img_tag(HI['anat_healthy'])}</div>
  <div>{img_tag(HI['anat_lesion'])}</div>
</div>
<div class="cap">Real B-scans. <span style="color:#ff3cc8">Magenta</span> = the band the attenuation is read from;
<span style="color:#0b0">green</span> = the thin band just under BM used to test transmission
(+{SLAB_UM[0]:.0f} to +{SLAB_UM[1]:.0f}&nbsp;&micro;m); <span style="color:#cc0">yellow</span> = BM. Left healthy,
right through the lesion. Notice both signs together: the RPE band vanishes and, in the same columns, the
choroid immediately below BM brightens.</div>

<h2><span class="step">3</span>Stacking the A-scans into a face-on map</h2>
<p>One number per A-scan turns the whole cube into a single flat image. Two things have to be fixed before that
image can be measured.</p>
<ul>
<li><b>The pixels are not square.</b> As acquired, the map has one row per B-scan and one column per A-scan.
Along a B-scan the A-scans are about {hero['fov'][0] / hero['W'] * 1000:.0f}&nbsp;&micro;m apart; between B-scans
the spacing is about {hero['fov'][1] / hero['n'] * 1000:.0f}&nbsp;&micro;m, roughly five times coarser. Each pixel
is therefore a tall thin sliver of retina, not a square patch. The map is resampled onto square pixels of
{ENF * 1000:.0f}&nbsp;&micro;m before anything is measured. Area alone could survive without this, by multiplying
by the true sliver area, but the <i>shape</i> rules could not: asking whether a lesion is at least
{MIN_DIAM_UM:.0f}&nbsp;&micro;m across only means something once a step sideways and a step downward cover the same
distance.</li>
<li><b>The stack is upside down.</b> The B-scan order runs opposite to the fundus, so the map is flipped to match
the infrared localizer. Area is unaffected by a flip; where the lesion <i>is</i> is not.</li>
</ul>
<p>Each B-scan is also acquired separately, so its overall brightness can jump slightly from the one before it.
That shows up as fine horizontal banding, which is removed by levelling each row against its neighbours. A lesion
spans many B-scans, so it survives this; a one-row jump does not.</p>

<div class="grid2">
  <div>{img_tag(HI['proj_native'])}</div>
  <div>{img_tag(HI['proj_enface'])}</div>
</div>
<div class="cap">The projection step on the hero eye. Left: the map as acquired, one row per B-scan (shown
stretched three times taller, or it would be an unreadable sliver). Right: the same map on square
{ENF * 1000:.0f}&nbsp;&micro;m pixels and flipped to fundus orientation. This is the image everything downstream
measures.</div>

<p>Written out in full, the face-on map you are looking at is made like this:</p>
<ol>
<li>For each A-scan, take the mean attenuation between {abs(RPE_UM[0])} and {abs(RPE_UM[1])}&nbsp;&micro;m above
BM. That is one number per A-scan, so the cube becomes a {hero['n']}&nbsp;&times;&nbsp;{hero['W']} array.</li>
<li>Level the rows against their neighbours, on that array, before anything else touches it.</li>
<li>Reverse the row order, so the map faces the same way as the fundus.</li>
<li>Rescale it, once, with a centred affine transform and bilinear interpolation, from its
{hero['fov'][1] / hero['n'] * 1000:.0f}&nbsp;&times;&nbsp;{hero['fov'][0] / hero['W'] * 1000:.0f}&nbsp;&micro;m
sampling onto square {ENF * 1000:.0f}&nbsp;&micro;m pixels, into a
{round(max(hero['fov']) / ENF)}&nbsp;&times;&nbsp;{round(max(hero['fov']) / ENF)} frame. Nothing is cropped;
anything outside the scanned field is left at zero.</li>
</ol>

<p>Two details about that image are worth stating plainly, because they are easy to assume wrongly.</p>
<ul>
<li><b>The grey levels are relative.</b> Every face-on map on this page is stretched between the 1st and 99th
percentile of its own values, so that the lesion is visible. Dark means low attenuation, which means the RPE is
missing. It does not mean a calibrated physical quantity, and two maps on this page cannot be compared by
brightness.</li>
<li><b>The map the software measures is not quite the map it shows.</b> Before any threshold is applied, the
projection is blurred with a Gaussian of about {2.0 * ENF * 1000:.0f}&nbsp;&micro;m. A single noisy pixel then
cannot create or destroy a lesion. Every cutoff, baseline and hole-fill described below acts on that blurred
copy; the figures show the unblurred one, because it is sharper to look at.</li>
</ul>

<h3 style="font-size:17px;margin:1.2em 0 .2em">Which pixels are measured at all</h3>
<p>Not the whole frame. Two regions are excluded before anything is decided, and they are excluded in both
directions: they can never be called GA, and they never contribute to the healthy baseline either.</p>
<ul>
<li>A <b>rim of 0.30&nbsp;mm</b> is eroded from the edge of the scanned field, where the signal is weakest.</li>
<li>The <b>vignette</b>: any location whose whole-column mean brightness falls below half the field median is
dropped. This test deliberately ignores BM and looks at the entire A-scan, because a column with no retina at all
is dim from top to bottom, whereas a column through severe GA still has its inner retina and a bright choroid.
That is why this gate removes the dead corners without cutting the centre out of a large lesion.</li>
</ul>

{fig(HI['field'], f"The measured field on the hero eye. Orange is excluded: {hero['field_note']}. These are the hatched grey regions in the profile figure further down.")}

{fig(HI['enface'], "Left: the finished RPE-loss map (dark = RPE gone). Right: the infrared localizer with the line marking the lesion B-scan shown above. The cyan line marks the cross-section used in the next figure.")}

<h2><span class="step">4</span>The decision: compare against a healthy baseline, not a fixed number</h2>
<p>Attenuation here is a <i>relative</i> measure, and healthy RPE does not read the same everywhere: it falls off
from the fovea toward the periphery, and it differs between eyes. No single fixed cutoff can work. So the software
builds a <b>healthy-RPE baseline out of the eye in front of it</b>:</p>
<ul>
<li>It finds the fovea from the data, as the centre of the brightest healthy RPE in the central 3&nbsp;mm.</li>
<li>In each 0.25&nbsp;mm ring around that centre it takes the <b>75th percentile</b> of the measured value. Atrophy
is dark, so it sits in the bottom of each ring's distribution and does not move the 75th percentile.</li>
<li>That radial profile is forced to <b>fall, or stay flat, with distance from the fovea</b>. It is never allowed
to rise. A gentle correction then restores the natural nasal/temporal asymmetry, again discarding any pixel that
sits well below the local trend.</li>
</ul>
<p>The point of the monotone constraint is subtle and it is the heart of the design. If the baseline were free to
follow the data, a large lesion would pull the baseline down over itself, and the lesion would then be compared
against a threshold it had lowered. It would hide. Because the profile can only fall with eccentricity and is
built from the healthy upper quartile, <b>a lesion cannot lower the threshold that is about to judge it</b>, no
matter how large it is.</p>
<p>A location becomes a GA candidate where its measured attenuation drops <b>below {FRAC * 100:.0f}% of that
local baseline</b>.</p>

{fig(HI['profile'], f"One horizontal cut through the lesion (the cyan line above). Black = the measured value; blue dashed = the healthy baseline the software built for this eye; red dotted = the {FRAC*100:.0f}% cutoff. Where black falls under red (inside the measured field), GA is called (green). A flat threshold would mis-call the naturally dimmer periphery; the fitted baseline follows it down.")}

<div class="callout"><b>This is the heart of it.</b> &ldquo;GA here&rdquo; means: the attenuation of the RPE band
has collapsed to less than half of what healthy RPE would give <i>at that location in this eye</i>.</div>

<h2><span class="step">5</span>The second cRORA criterion: hypertransmission</h2>
<p>The software now asks whether light actually reaches the choroid in a thin band just under BM
(+{SLAB_UM[0]:.0f} to +{SLAB_UM[1]:.0f}&nbsp;&micro;m). The band is deliberately shallow. A deeper band drifts
into bright sclera where the choroid thins at the edge of the field, and fakes transmission there.</p>
<p>This criterion is not the one that recognises atrophy. Healthy macula transmits a fair amount of light as
well. What it does is repair the two opposite ways in which the attenuation test, on its own, goes wrong:</p>
<ul>
<li><b>Field corners and no-retina edges</b> read as low attenuation, which looks exactly like RPE loss, but
there is nothing there to transmit. Requiring some transmission drops them out.</li>
<li><b>Centres of large, heterogeneous GA</b> sometimes contain bright material sitting above BM, which mimics
&ldquo;RPE present&rdquo; and leaves a <b>hole</b> in the map. Interior holes that transmit like the atrophy
around them are <b>filled in</b>; holes that do not transmit, such as an island of spared RPE, are left open.</li>
</ul>
<p>The cutoff is mostly relative to the eye's own tissue, but it can never drop below a fixed floor of
{HYPER_ABS:.2f}. Without that floor, a healthy eye with no lesion at all would set itself a vanishingly small
cutoff, and image noise would climb over it and be reported as atrophy.</p>

{fig(HI['hyper'], "Left: the transmission map (bright = light reaches the choroid). Middle: the candidates from the attenuation test (white outline), with any pixel rejected for not transmitting shown in blue" + (f"; on this eye there are none, which is the point of the rule rather than a failure of it" if hero['n_rejected'] == 0 else f"; {hero['n_rejected']} pixels are rejected here") + ". Right: interior holes that transmit like the surrounding atrophy are filled (cyan), patches that never reach complete loss are dropped (orange), and the final footprint is outlined in green.")}

<h2><span class="step">6</span>Complete atrophy, not partial thinning</h2>
<p>cRORA means <i>complete</i> RPE and outer-retinal atrophy. An RPE that is merely thinned, for example over a
druse or at the fading edge of a lesion, can dip below the halfway cutoff without ever being truly absent. So each
surviving patch must reach <b>below {MIN_DEPTH * 100:.0f}% of its local baseline somewhere inside it</b>. A patch
that only ever hovers around 30 or 40% is attenuated, not atrophic, and is discarded whole.</p>

<h2><span class="step">7</span>From pixels to an area</h2>
<p>Finally the size rule. A patch is kept only if its <b>longest dimension is at least
{MIN_DIAM_UM:.0f}&nbsp;&micro;m</b>, which on this scan is about {0.25 / ENF:.0f} pixels across. That is the
smallest lesion the cRORA definition recognises, and it discards isolated specks. Note that this is the
<i>longest</i> axis, not the narrowest: a long, thin sliver of atrophy is kept, as it should be. The surviving
pixels are counted and multiplied by the pixel area to give mm&sup2;. Below, our OCT-only result sits beside the
independent <b>Zeiss PLEX advRPE</b> reference for the same eye.</p>

{fig(HI['final'], f"{hero['label']}: our OCT-only GA {hero['area']:.2f} mm² vs the PLEX advRPE reference {hero['adv']:.2f} mm²; against the in-frame manual outline the spatial agreement is Dice {hero['dice']:.2f}.")}

<table>
<tr><th>Eye</th><th>Our OCT-only area</th><th>PLEX advRPE reference</th><th>Spatial agreement</th></tr>
<tr><td>{hero['label']}</td><td class="num">{hero['area']:.2f} mm²</td><td class="num">{hero['adv']:.2f} mm²</td><td class="num">Dice {hero['dice']:.2f}</td></tr>
<tr><td>{demo['label']}</td><td class="num">{demo['area']:.2f} mm²</td><td class="num">{demo['adv']:.2f} mm²</td><td>no in-frame gold</td></tr>
</table>

<h2>A worked example, with the real numbers</h2>
<p>The prose above is easier to trust with arithmetic attached. Below are genuine locations in
{hero['label']} put through exactly the decision the software makes. The values are read straight out of the
detector, not recomputed for the page.</p>

{example_table(hero)}

<h2>Where the area actually comes from</h2>
<p>The same accounting for the whole eye. Each row is the number of pixels still standing after that rule, and
the area they represent. The last row is the number the software reports.</p>

{stage_table(eyes)}

<h2>The decision, as one flow</h2>
<p>What happens at every single A-scan location:</p>
{decision_flow()}

<h2>Second example: large confluent GA ({demo['label']})</h2>
<p>The same pipeline on a large, multifocal lesion, where the hole-filling criterion matters most, because
heterogeneous GA centres leave interior gaps that must be recovered.</p>

{fig(DI['anat_lesion'], "A real B-scan through the large lesion: the RPE band is absent across a wide span and the choroid below BM is strikingly bright.")}
{fig(DI['hyper'], "Hole-filling in action: interior gaps inside the confluent lesion (cyan) are recovered because they transmit light like the atrophy around them. In orange, the satellite patches thrown out by the complete-loss rule: their RPE is attenuated, not absent.")}
{fig(DI['final'], f"Our OCT-only GA {demo['area']:.2f} mm² vs the PLEX advRPE reference {demo['adv']:.2f} mm² on a large lesion.")}

<h2>What this is, and what it isn't</h2>
<div class="warn">
<ul style="margin:.2em 0 0;padding-left:1.1em">
<li><b>Research prototype, not a cleared device.</b> Do not use for clinical decisions.</li>
<li>The <b>PLEX advRPE</b> comparison is itself an automatic deep-learning output (a <i>silver</i> reference), not a
human grader. The only human <i>gold</i> here is the small in-frame manual outline on {hero['label']}.</li>
<li>Attenuation is used as a <b>relative, within-eye</b> measure (normalised to each eye's own healthy RPE); the 50% cutoff
is not an absolute, externally-calibrated threshold.</li>
<li>Absolute mm&sup2; assumes a standard <b>model-eye scaling</b> of the Spectralis field (no axial-length
correction), so areas carry that assumption.</li>
<li>These are <b>two illustrative eyes</b>; cohort-level accuracy is established separately, not from two cases.</li>
</ul>
</div>

<footer>Generated by <code>src/explain_ga.py</code> from the live pipeline
(<code>reader/core/oac_ga.py</code>). Images are the algorithm's own intermediate outputs on real patient scans.
OCT-only: the computed area uses the OCT volume alone; PLEX is shown only as an external reference.</footer>

</div></body></html>"""
    return html


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    eyes = []
    for spec in EYES:
        d = load_eye(spec)
        d["imgs"] = build_figures(d)
        # also dump loose PNGs for QC
        for name, b64 in d["imgs"].items():
            with open(os.path.join(OUT, f"{d['tag']}_{name}.png"), "wb") as f:
                f.write(base64.b64decode(b64))
        eyes.append(d)
    html = build_html(eyes)
    out_html = os.path.join(OUT, "ga_explainer.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nwrote {out_html}  ({len(html)/1e6:.2f} MB, {sum(len(e['imgs']) for e in eyes)} figures)")
    print(f"loose QC PNGs in {OUT}")


if __name__ == "__main__":
    main()
