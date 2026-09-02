"""Field-validity mask: detect machine-inserted saturated 'white band' columns.

Some Spectralis B-scans carry a fully-saturated WHITE vertical band at a frame edge -- pixel value
at the volume max for (nearly) the full column height. It is out-of-field fill the device inserts,
NOT anatomy, and it silently corrupts BM segmentation, the OAC GA area, the BM-DL training labels,
and the human B-scan display (see docs / the project plan).

This module is the single source of truth for "which A-scans are real in-field signal". It is PURE
(numpy only, no I/O, deterministic from the volume), so nothing needs persisting -- every consumer
recomputes it cheaply at load time and reads `OctVolume.field_valid`.

The physical signature that separates machine-fill from genuine bright tissue (RPE / hypertransmission):
a real OCT A-scan ALWAYS has a near-black vitreous above the retina, so a column whose VITREOUS rows are
themselves saturated cannot be real signal. We require both a saturated column AND a saturated vitreous.
"""
import numpy as np

SAT_REL = 0.98        # a pixel counts as "saturated" at >= SAT_REL * vmax
SAT_FRAC = 0.90       # fraction of a column's pixels that must be saturated
VIT_FRAC = 0.90       # fraction of the vitreous rows that must be saturated
VIT_FRAC_OF_H = 0.10  # vitreous band = the top 10% of rows (near-black in any real OCT)
VIT_MIN_ROWS = 8


def invalid_mask(vol, *, sat_rel=SAT_REL, sat_frac=SAT_FRAC, vit_frac=VIT_FRAC,
                 vit_rows=None, edge_only=True, vmax=None):
    """(n,H,W) float volume -> (n,W) bool INVALID mask (True = machine-fill / out-of-field).

    A column x of B-scan b is invalid iff BOTH:
      1. saturation fraction: mean(vol[b,:,x] >= sat_rel*vmax) >= sat_frac, and
      2. vitreous bright: the top `vit_rows` rows are also saturated (mean >= vit_frac) -- impossible
         in real OCT, so this is what cleanly separates fill from genuine bright tissue.
    `vmax` defaults to float(vol.max()); the test is purely relative-to-max so it is identical for
    0..1 float and 0..255 raw E2E volumes. `edge_only` keeps only invalid runs anchored to a frame edge
    (the observed artifact); set False to also flag interior saturated columns.
    """
    v = np.asarray(vol, np.float32)
    if v.ndim != 3:
        raise ValueError(f"invalid_mask expects (n,H,W), got {v.shape}")
    n, H, W = v.shape
    if vmax is None:
        vmax = float(v.max())
    if not np.isfinite(vmax) or vmax <= 0:
        return np.zeros((n, W), bool)
    if vit_rows is None:
        vit_rows = max(VIT_MIN_ROWS, int(round(VIT_FRAC_OF_H * H)))
    vit_rows = min(vit_rows, H)

    sat = v >= (sat_rel * vmax)                      # (n,H,W) bool
    colsat = sat.mean(axis=1) >= sat_frac            # (n,W) whole-column saturated
    vit = sat[:, :vit_rows, :].mean(axis=1) >= vit_frac   # (n,W) vitreous saturated
    inv = colsat & vit
    if edge_only:
        inv = _edge_anchor(inv)
    return inv


def _edge_anchor(inv):
    """(n,W) bool -> keep only invalid runs contiguous with col 0 (left) or col W-1 (right), per B-scan.

    A real specular artifact mid-frame would be a freak; the machine fill is always an edge run, so we
    flood-fill inward from each frame edge through the invalid columns and drop anything not reached.
    """
    inv = np.asarray(inv, bool)
    n, W = inv.shape
    out = np.zeros_like(inv)
    for i in range(n):
        row = inv[i]
        if not row.any():
            continue
        x = 0                                        # grow from the left edge
        while x < W and row[x]:
            out[i, x] = True
            x += 1
        x = W - 1                                    # grow from the right edge
        while x >= 0 and row[x]:
            out[i, x] = True
            x -= 1
    return out


def _edge_runs(row):
    """A (W,) bool row -> ('L'|'R' tag, width) for the left/right edge-anchored runs."""
    W = len(row)
    lw = 0
    while lw < W and row[lw]:
        lw += 1
    rw = 0
    while rw < W and row[W - 1 - rw]:
        rw += 1
    return lw, rw


def summarize(inv, fov_mm=None):
    """(n,W) bool -> per-B-scan QC list: {bscan, edge, width_px, frac[, width_mm]}.

    `edge` is 'L'/'R'/'LR'/'' from which frame edges carry a band. width_px = max(left,right) run.
    """
    inv = np.asarray(inv, bool)
    n, W = inv.shape
    mmpp = (fov_mm[0] / W) if (fov_mm and fov_mm[0]) else None
    out = []
    for i in range(n):
        row = inv[i]
        lw, rw = _edge_runs(row)
        edge = ("L" if lw else "") + ("R" if rw else "")
        width_px = int(max(lw, rw))
        rec = {"bscan": i, "edge": edge, "width_px": width_px, "frac": float(row.mean())}
        if mmpp is not None:
            rec["width_mm"] = round(width_px * mmpp, 4)
        out.append(rec)
    return out


def eye_metrics(inv, fov_mm=None):
    """(n,W) bool -> whole-eye QC rollup for the field-validity sidecar."""
    inv = np.asarray(inv, bool)
    n, W = inv.shape
    per = summarize(inv, fov_mm)
    widths = [r["width_px"] for r in per]
    banded = [r for r in per if r["width_px"] > 0]
    mmpp = (fov_mm[0] / W) if (fov_mm and fov_mm[0]) else None
    max_w = int(max(widths)) if widths else 0
    return {
        "n_bscans_with_band": len(banded),
        "frac_bscans_with_band": round(len(banded) / n, 4) if n else 0.0,
        "max_band_width_px": max_w,
        "max_band_width_mm": round(max_w * mmpp, 4) if mmpp is not None else None,
        "total_invalid_frac": round(float(inv.mean()), 6),
    }
