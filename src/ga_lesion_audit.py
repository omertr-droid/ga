#!/usr/bin/env python
"""Lesion-level OCT audit for GA detector disagreements.

Area agreement is deliberately secondary here.  For every PLEX component and every OCT-only
component, this script records the detector stage that accepted/rejected the same *location* and
renders representative outer-retina B-scans through it.  The output vocabulary is neutral:
``overlap``, ``ours-only`` and ``PLEX-only`` do not imply that either automatic method is truth.

The B-scan panels show the evidence needed for clinical adjudication:

* red bar: registered PLEX label at that B-scan;
* lime bar: our final OCT-only footprint;
* yellow bar: raw OAC RPE-loss candidate;
* orange bar: a cRORA-sized component rejected by the complete-loss/depth rule;
* purple bar: a current-final component the provisional component-median hyper gate would remove;
* yellow curve: DL Bruch's membrane;
* magenta/cyan curve: strong/faded automatically tracked RPE peak (a cue, not ground truth).

Run from the repository root::

    oct_env\\Scripts\\python.exe src\\ga_lesion_audit.py NHAMD-003-014 V1 OD
    oct_env\\Scripts\\python.exe src\\ga_lesion_audit.py --set

Outputs:
  outputs/ga_lesion_audit/<eye>_lesion_audit.png
  results/ga_lesion_components.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OCT_BM_DL", "1")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import measure

import bm_dl
import ga_error_audit as common
import m3_projections as mp
from paths import OUT_DIR, RESULTS_DIR
from reader.core import e2e_source, layers, oac_ga, projection
from reader.core.layer_store import JsonSidecarLayerStore
from viewer.core import ga_native


OUT = os.path.join(OUT_DIR, "ga_lesion_audit")
MMPP = oac_ga.MMPP
MMPP2 = oac_ga.MMPP2
AX_UM = projection.AX
PROVISIONAL_HYPER_P50 = 0.60

# The cases that span the observed reference-error / detector-miss conflict.  005 OD is the fixed
# in-frame-gold sentinel; the other rows are selected by spatial disagreement, not by net area alone.
AUDIT_SET = [
    ("NHAMD-003-006", "V3", "OD"),
    ("NHAMD-003-010", "V1", "OD"),
    ("NHAMD-003-014", "V1", "OD"),
    ("NHAMD-003-001", "V1", "OD"),
    ("NHAMD-003-001", "V1", "OS"),
    ("NHAMD-003-026", "V3", "OD"),
    ("NHAMD-003-026", "V3", "OS"),
    ("NHAMD-003-008", "V1", "OS"),
    ("NHAMD-003-011", "V3", "OD"),
    ("NHAMD-003-011", "V3", "OS"),
    ("NHAMD-003-015", "V3", "OD"),
    ("NHAMD-003-003", "V3", "OD"),
    ("NHAMD-003-003", "V3", "OS"),
    ("NHAMD-003-005", "V3", "OD"),
    ("NHAMD-003-005", "V3", "OS"),
]


COMPONENT_COLUMNS = [
    "subject", "eye", "adjudication", "registration_flag", "source", "component",
    "area_mm2", "plex_fraction", "ours_fraction", "candidate_fraction",
    "hyper_kept_fraction", "filled_fraction", "sized_fraction", "depth_rejected_fraction",
    "final_fraction", "loss_base_min", "loss_base_p01", "loss_base_p05",
    "loss_base_median", "hyper_median", "core_edge_distance_p10_mm",
    "rpe_absent_fraction", "rpe_faded_fraction", "rpe_intact_fraction",
    "rpe_to_bm_median_um", "retina_thickness_median_um", "bm_slope_p95_px_per_ascan",
    "native_bscan_min", "native_bscan_max", "native_x_min", "native_x_max", "native_bscans_covered",
    "representative_bscan", "representative_x0", "representative_x1",
]


def _registration_flags():
    out = {}
    path = os.path.join(RESULTS_DIR, "registration.csv")
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[(r["subject"], r["eye"].upper())] = r.get("reg_flag", "")
    return out


REG_FLAGS = _registration_flags()


@dataclass
class EyeData:
    subject: str
    visit: str
    eye: str
    row: dict
    ov: object
    bm: np.ndarray
    prep: dict
    stages: dict
    plex: np.ndarray
    meta: dict
    rpe_row: np.ndarray
    rpe_prom: np.ndarray

    @property
    def slug(self):
        return common.bundle_slug(self.subject, self.visit, self.eye)

    @property
    def subject_visit(self):
        return self.subject if self.subject.endswith(f"-{self.visit}") else f"{self.subject}-{self.visit}"


def load_eye(subject, visit, eye):
    row, e2e = common.resolve(subject, visit, eye)
    if row is None or not os.path.exists(e2e):
        raise FileNotFoundError(f"No worklist/E2E row for {subject} {visit} {eye}")
    raw = e2e_source.open_e2e(e2e)
    idx = e2e_source.default_volume_index(raw, eye.upper())
    ov = e2e_source.load_volume(raw, idx)

    bp = os.path.join(common.LIB_DIR, common.bundle_slug(subject, visit, eye), "bundle.npz")
    bm = None
    if os.path.exists(bp):
        try:
            with np.load(bp, allow_pickle=False) as z:
                if "bm_dl" in z.files:
                    candidate = np.asarray(z["bm_dl"], np.float32)
                    if candidate.shape == (ov.n_bscans, ov.W):
                        bm = candidate
        except (OSError, ValueError, KeyError):
            bm = None
    if bm is None and bm_dl.available():
        bm = bm_dl.segment_volume(ov.vol).astype(np.float32)
    elif bm is None:
        _, bm = layers.effective_surfaces(ov, JsonSidecarLayerStore(common.CORR_DIR))

    p = oac_ga.prep(ov, bm, baseline="radial2")
    s = oac_ga.footprint_stages(p)
    plex, meta = common.plex_label_enface(common.bundle_slug(subject, visit, eye), p["rpe6"].shape[0])
    if plex is None:
        plex = np.zeros_like(p["core"], bool)
        meta = {}
    plex &= p["core"]
    rpe_row, rpe_prom = mp.rpe_surface(ov.vol, bm)
    return EyeData(subject, visit, eye.upper(), row, ov, bm, p, s, plex, meta or {}, rpe_row, rpe_prom)


def scalar_enface_to_native(a, ov):
    """Same centred inverse geometry as ga_native.enface_to_native, but linear for scalar cues."""
    fh = max(1, int(round(ov.fov_mm[1] / MMPP)))
    fw = max(1, int(round(ov.fov_mm[0] / MMPP)))
    field = ga_native.center_extract(np.asarray(a, np.float32), fh, fw)
    nat = cv2.resize(field, (ov.W, ov.n_bscans), interpolation=cv2.INTER_LINEAR)
    return nat[::-1] if getattr(ov, "enface_flip", True) else nat


def bool_native(a, ov):
    return ga_native.enface_to_native(
        np.asarray(a, bool), ov.fov_mm, ov.n_bscans, ov.W, getattr(ov, "enface_flip", True)
    )


def components(mask, min_px=4):
    lbl = measure.label(np.asarray(mask, bool))
    out = []
    for rp in sorted(measure.regionprops(lbl), key=lambda x: x.area, reverse=True):
        if rp.area >= min_px:
            out.append((int(rp.label), lbl == rp.label, rp))
    return out


def bounds_for_row(row, W, margin=28, min_width=150):
    xs = np.flatnonzero(row)
    if not xs.size:
        return 0, W
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    width = x1 - x0
    grow = max(margin, (min_width - width + 1) // 2)
    x0, x1 = max(0, x0 - grow), min(W, x1 + grow)
    if x1 - x0 < min_width:
        missing = min_width - (x1 - x0)
        x0 = max(0, x0 - missing // 2)
        x1 = min(W, x0 + min_width)
        x0 = max(0, x1 - min_width)
    return x0, x1


def representative(mask, ov):
    nat = bool_native(mask, ov)
    per = nat.sum(axis=1)
    if not per.any():
        return 0, 0, ov.W, nat
    i = int(np.argmax(per))
    x0, x1 = bounds_for_row(nat[i], ov.W)
    return i, x0, x1, nat


def _frac(a, comp):
    den = int(comp.sum())
    return float((np.asarray(a, bool) & comp).sum()) / den if den else 0.0


def component_record(d: EyeData, source, number, comp, ratio_nat, hyper_nat, core_dist, bm_slope):
    p, s, ov = d.prep, d.stages, d.ov
    i, x0, x1, nat = representative(comp, ov)
    vals = np.asarray(p["loss6"] / np.maximum(p["base"], 1e-6), float)[comp]
    hypers = np.asarray(p["hyper6"], float)[comp]
    edge = core_dist[comp]
    pr = d.rpe_prom[nat]
    absent = float(np.mean(pr < 1.15)) if pr.size else float("nan")
    faded = float(np.mean((pr >= 1.15) & (pr < 1.50))) if pr.size else float("nan")
    intact = float(np.mean(pr >= 1.50)) if pr.size else float("nan")
    sep = (d.bm - d.rpe_row) * AX_UM
    sep_vals = sep[nat & (d.rpe_prom >= 1.15)]
    thickness = (d.bm - ov.ilm) * AX_UM
    thick_vals = thickness[nat]
    slope_vals = bm_slope[nat]
    yy, xx = np.where(nat)
    adj = common.ADJUDICATION.get((d.subject_visit, d.eye), {})
    return {
        "subject": d.subject_visit,
        "eye": d.eye,
        "adjudication": adj.get("verdict", ""),
        "registration_flag": REG_FLAGS.get((d.subject_visit, d.eye), ""),
        "source": source,
        "component": number,
        "area_mm2": comp.sum() * MMPP2,
        "plex_fraction": _frac(d.plex, comp),
        "ours_fraction": _frac(s["final"], comp),
        "candidate_fraction": _frac(s["rpe_candidate"], comp),
        "hyper_kept_fraction": _frac(s["hyper_kept"], comp),
        "filled_fraction": _frac(s["filled"], comp),
        "sized_fraction": _frac(s["sized"], comp),
        "depth_rejected_fraction": _frac(s["partial_rejected"], comp),
        "final_fraction": _frac(s["final"], comp),
        "loss_base_min": float(np.min(vals)) if vals.size else float("nan"),
        "loss_base_p01": float(np.percentile(vals, 1)) if vals.size else float("nan"),
        "loss_base_p05": float(np.percentile(vals, 5)) if vals.size else float("nan"),
        "loss_base_median": float(np.median(vals)) if vals.size else float("nan"),
        "hyper_median": float(np.median(hypers)) if hypers.size else float("nan"),
        "core_edge_distance_p10_mm": float(np.percentile(edge, 10) * MMPP) if edge.size else float("nan"),
        "rpe_absent_fraction": absent,
        "rpe_faded_fraction": faded,
        "rpe_intact_fraction": intact,
        "rpe_to_bm_median_um": float(np.median(sep_vals)) if sep_vals.size else float("nan"),
        "retina_thickness_median_um": float(np.median(thick_vals)) if thick_vals.size else float("nan"),
        "bm_slope_p95_px_per_ascan": float(np.percentile(slope_vals, 95)) if slope_vals.size else float("nan"),
        "native_bscan_min": int(yy.min()) if yy.size else 0,
        "native_bscan_max": int(yy.max()) if yy.size else 0,
        "native_x_min": int(xx.min()) if xx.size else 0,
        "native_x_max": int(xx.max()) if xx.size else 0,
        "native_bscans_covered": int(np.unique(yy).size) if yy.size else 0,
        "representative_bscan": i,
        "representative_x0": x0,
        "representative_x1": x1,
        "_mask": comp,
        "_native": nat,
    }


def eye_components(d: EyeData):
    ratio_nat = scalar_enface_to_native(d.prep["loss6"] / np.maximum(d.prep["base"], 1e-6), d.ov)
    hyper_nat = scalar_enface_to_native(d.prep["hyper6"], d.ov)
    core_dist = distance_transform_edt(d.prep["core"])
    bm_slope = np.abs(np.gradient(d.bm, axis=1))
    rows = []
    for source, mask in (
        ("PLEX", d.plex),
        # Canonical learning unit: every cRORA-sized proposal BEFORE the current complete-loss veto.
        # This includes both final positives and the rejected 014-like components a better combiner
        # must learn to adjudicate.
        ("sized_candidate", d.stages["sized"] & d.prep["core"]),
        ("ours", d.stages["final"] & d.prep["core"]),
    ):
        for number, (_, comp, _) in enumerate(components(mask), start=1):
            rows.append(component_record(d, source, number, comp, ratio_nat, hyper_nat, core_dist, bm_slope))
    return rows, ratio_nat, hyper_nat


def _contour(ax, m, color, lw=1.4):
    if np.asarray(m, bool).any():
        ax.contour(np.asarray(m, bool).astype(float), levels=[0.5], colors=[color], linewidths=lw)


def _plot_masked_curve(ax, x, y, keep, **kw):
    yy = np.where(keep, y, np.nan)
    ax.plot(x, yy, **kw)


def _plot_intervals(ax, row, y, color, lw):
    for a, b in ga_native.intervals(np.asarray(row, bool)):
        ax.plot([a, b], [y, y], color=color, lw=lw, solid_capstyle="butt")


def cases_for_panel(d: EyeData, records, limit=6):
    """Cover distinct components first, then multiple cross-sections of the largest PLEX lesion."""
    cases = []
    used = set()
    plex_records = [r for r in records if r["source"] == "PLEX"]
    ours_only_records = [r for r in records if r["source"] == "ours" and r["plex_fraction"] < 0.25]
    weak_hyper_records = [
        r for r in records if r["source"] == "ours" and r["hyper_median"] < PROVISIONAL_HYPER_P50
    ]

    def add_case(rec, i):
        key = int(i)
        if key in used or len(cases) >= limit:
            return
        row = rec["_native"][key]
        if not row.any():
            return
        x0, x1 = bounds_for_row(row, d.ov.W)
        cases.append((rec, key, x0, x1))
        used.add(key)

    # Reserve panel space for BOTH directions and for the exact final components the provisional robust
    # hyper gate would remove.  A many-focus PLEX mask must never crowd those cases out of the figure.
    for rec in plex_records[:3]:
        add_case(rec, rec["representative_bscan"])
    for rec in ours_only_records[:1]:
        add_case(rec, rec["representative_bscan"])
    for rec in weak_hyper_records[:2]:
        add_case(rec, rec["representative_bscan"])
    for rec in plex_records[3:] + ours_only_records[1:]:
        add_case(rec, rec["representative_bscan"])

    # Add cross-sections across the largest disputed component, avoiding near-duplicate B-scans.
    for rec in plex_records + ours_only_records:
        nat = rec["_native"]
        rows = np.flatnonzero(nat.sum(axis=1) > 0)
        if not rows.size:
            continue
        for q in (0.15, 0.35, 0.50, 0.65, 0.85):
            target = int(rows[int(round(q * (len(rows) - 1)))])
            if all(abs(target - old) >= 2 for old in used):
                add_case(rec, target)
            if len(cases) >= limit:
                break
        if len(cases) >= limit:
            break
    return sorted(cases, key=lambda z: z[1])


def render_eye(d: EyeData, records, ratio_nat, hyper_nat):
    p, s, ov = d.prep, d.stages, d.ov
    ours = s["final"] & p["core"]
    overlap = ours & d.plex
    ours_only = ours & ~d.plex
    plex_only = d.plex & ~ours
    weak_hyper = np.zeros_like(ours)
    for _, comp, _ in components(ours, min_px=1):
        if float(np.median(p["hyper6"][comp])) < PROVISIONAL_HYPER_P50:
            weak_hyper |= comp
    cases = cases_for_panel(d, records)

    final_nat = bool_native(ours, ov)
    plex_nat = bool_native(d.plex, ov)
    cand_nat = bool_native(s["rpe_candidate"], ov)
    depth_nat = bool_native(s["partial_rejected"], ov)
    weak_hyper_nat = bool_native(weak_hyper, ov)

    fig = plt.figure(figsize=(22, 15.5), facecolor="white")
    gs = GridSpec(3, 12, figure=fig, height_ratios=[0.90, 1.12, 1.12], hspace=0.31, wspace=0.25)
    top = [fig.add_subplot(gs[0, j * 3:(j + 1) * 3]) for j in range(4)]
    bottoms = [fig.add_subplot(gs[1 + j // 3, (j % 3) * 4:(j % 3 + 1) * 4]) for j in range(6)]

    base = common.norm8(p["rpe6"])
    top[0].imshow(base, cmap="gray")
    _contour(top[0], d.plex, "red", 1.5)
    _contour(top[0], ours, "lime", 1.5)
    top[0].set_title("RPE-loss map\nred = PLEX outline · lime = ours", fontsize=9.5)

    rel = np.dstack([base, base, base]) * 0.35
    rel[plex_only] = [0.20, 0.42, 1.00]
    rel[ours_only] = [1.00, 0.75, 0.05]
    rel[overlap] = [0.05, 0.90, 0.15]
    top[1].imshow(rel)
    top[1].set_title(
        "Neutral spatial relation (not truth)\n"
        f"overlap {overlap.sum()*MMPP2:.2f} · ours-only {ours_only.sum()*MMPP2:.2f} · "
        f"PLEX-only {plex_only.sum()*MMPP2:.2f} mm2", fontsize=9.5,
    )

    stage = np.dstack([base, base, base]) * 0.32
    stage[s["rpe_candidate"]] = [1.00, 0.90, 0.00]
    stage[s["hyper_rejected"]] = [0.90, 0.05, 0.05]
    stage[s["partial_rejected"]] = [1.00, 0.40, 0.00]
    stage[s["final"]] = [0.00, 0.90, 0.05]
    stage[weak_hyper] = [0.78, 0.10, 1.00]
    top[2].imshow(stage)
    _contour(top[2], d.plex, "white", 1.0)
    top[2].set_title(
        "Detector path\nyellow candidate · red hyper reject · orange depth reject · green final · purple weak-hyper",
        fontsize=9.5,
    )

    prom6 = projection.to_enface(d.rpe_prom.astype(np.float32), ov.fov_mm, getattr(ov, "enface_flip", True))
    im = top[3].imshow(prom6, cmap="magma", vmin=0.8, vmax=2.2)
    _contour(top[3], d.plex, "cyan", 1.2)
    _contour(top[3], ours, "lime", 1.2)
    top[3].set_title("Tracked RPE-peak prominence (cue only)\ncyan = PLEX · lime = ours", fontsize=9.5)
    fig.colorbar(im, ax=top[3], fraction=0.046, pad=0.02)

    for ax in top:
        ax.axis("off")

    for ax, case in zip(bottoms, cases):
        rec, i, x0, x1 = case
        bmseg = d.bm[i, x0:x1]
        y0 = max(0, int(np.floor(np.percentile(bmseg, 3) - 75)))
        y1 = min(ov.H, int(np.ceil(np.percentile(bmseg, 97) + 48)))
        if y1 - y0 < 100:
            extra = (100 - (y1 - y0)) // 2 + 1
            y0, y1 = max(0, y0 - extra), min(ov.H, y1 + extra)
        crop = ov.vol[i, y0:y1, x0:x1].astype(np.float32)
        lo, hi = np.percentile(crop, (0.8, 99.7))
        disp = np.clip((crop - lo) / (hi - lo + 1e-6), 0, 1) ** 0.85
        ax.imshow(disp, cmap="gray", extent=(x0, x1, y1, y0), aspect="auto")
        xs = np.arange(x0, x1)
        ax.plot(xs, d.bm[i, x0:x1], color="yellow", lw=1.1)
        pr = d.rpe_prom[i, x0:x1]
        rr = d.rpe_row[i, x0:x1]
        _plot_masked_curve(ax, xs, rr, pr >= 1.50, color="magenta", lw=1.0)
        _plot_masked_curve(ax, xs, rr, (pr >= 1.15) & (pr < 1.50), color="cyan", lw=1.0)

        # Four independent location bars; this is the anti-cancellation view.
        ys = [y1 - 7, y1 - 13, y1 - 19, y1 - 25, y1 - 31]
        _plot_intervals(ax, plex_nat[i], ys[0], "red", 4.0)
        _plot_intervals(ax, final_nat[i], ys[1], "lime", 4.0)
        _plot_intervals(ax, cand_nat[i], ys[2], "yellow", 3.0)
        _plot_intervals(ax, depth_nat[i], ys[3], "orangered", 3.0)
        _plot_intervals(ax, weak_hyper_nat[i], ys[4], "violet", 3.0)

        target = rec["_native"][i, x0:x1]
        vals = pr[target]
        rr_ratio = ratio_nat[i, x0:x1][target]
        hh = hyper_nat[i, x0:x1][target]
        intact = 100.0 * np.mean(vals >= 1.50) if vals.size else float("nan")
        faded = 100.0 * np.mean((vals >= 1.15) & (vals < 1.50)) if vals.size else float("nan")
        absent = 100.0 * np.mean(vals < 1.15) if vals.size else float("nan")
        title = (
            f"{rec['source']} C{rec['component']} · B-scan {i} · x {x0}:{x1}\n"
            f"RPE tracker: strong {intact:.0f}% · faded {faded:.0f}% · absent {absent:.0f}%\n"
            f"OAC loss/base med {np.median(rr_ratio):.2f} · hyper med {np.median(hh):.2f}"
        )
        ax.set_title(title, fontsize=8.7)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_xlabel("A-scan")
        ax.set_ylabel("axial pixel")

    for ax in bottoms[len(cases):]:
        ax.axis("off")

    adj = common.ADJUDICATION.get((d.subject_visit, d.eye), {}).get("verdict", "none")
    reg = REG_FLAGS.get((d.subject_visit, d.eye), "unknown")
    ours_area = float(ours.sum()) * MMPP2
    plex_area = float(d.row["advRPE_area_mm2"])
    fig.suptitle(
        f"{d.subject_visit} {d.eye} — lesion-level OCT audit | PLEX {plex_area:.2f}, ours {ours_area:.2f} mm2"
        f" | adjudication: {adj} | registration: {reg}\n"
        "B-scan bars: red PLEX · lime final ours · yellow raw OAC candidate · orange depth-vetoed · "
        "purple final component with hyper median <0.60. "
        "RPE tracker: magenta strong · cyan faded; it is an algorithmic cue, not the adjudicator.",
        fontsize=13.2, y=0.995,
    )
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{d.slug}_lesion_audit.png")
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path


def clean_record(r):
    out = {}
    for k in COMPONENT_COLUMNS:
        v = r.get(k, "")
        if isinstance(v, (float, np.floating)):
            out[k] = "" if not np.isfinite(v) else round(float(v), 5)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", nargs="?")
    ap.add_argument("visit", nargs="?")
    ap.add_argument("eye", nargs="?")
    ap.add_argument("--set", action="store_true", help="run the targeted spatial disagreement set")
    ap.add_argument("--no-csv", action="store_true", help="render without replacing the consolidated table")
    ap.add_argument("--no-render", action="store_true", help="rebuild component table without redrawing panels")
    args = ap.parse_args()
    if args.set or not args.subject:
        eyes = AUDIT_SET
    else:
        if not args.visit or not args.eye:
            ap.error("subject, visit and eye are all required")
        eyes = [(args.subject, args.visit, args.eye)]

    all_rows = []
    for subject, visit, eye in eyes:
        print(f"\n{subject}-{visit} {eye}")
        try:
            d = load_eye(subject, visit, eye)
            records, ratio_nat, hyper_nat = eye_components(d)
            all_rows.extend(clean_record(r) for r in records)
            if args.no_render:
                print(f"  {len(records)} anatomical components")
            else:
                path = render_eye(d, records, ratio_nat, hyper_nat)
                print(f"  {len(records)} anatomical components -> {path}")
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}")
            traceback.print_exc()

    if not args.no_csv:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "ga_lesion_components.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COMPONENT_COLUMNS)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {csv_path} ({len(all_rows)} components)")


if __name__ == "__main__":
    main()
