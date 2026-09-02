#!/usr/bin/env python
"""Export the hand-validated BM eyes into a training-ready B-scan -> BM-surface dataset.

For each hand-validated eye (auto-discovered from the reader's BM Library — every device-BM worklist eye
with >=1 validated B-scan), pull every validated B-scan's image + its GOLD BM surface, plus two reference
surfaces and a per-column "hardness" weight, and pack them for off-machine (Colab) training of a DL BM
segmenter.

Per B-scan we store:
  * image       - the model-input B-scan (render.bscan_model_input): masked-contrast norm8 with the
                  saturated 'white band' columns NEUTRALIZED (replaced by their nearest valid neighbour),
                  so the net is never fed an all-white stripe. Inference MUST use the same transform for
                  parity (the human display, render.bscan_png, keeps the band shaded instead).
  * bm  (gold)  - effective_surfaces(ov, store)[1]: device BM with your per-B-scan corrections folded in.
                  Accepted (un-edited) scans correctly fall back to the device BM you signed off on.
  * device_bm   - ov.bm (the gap-filled device baseline effective_surfaces starts from) = the eval/device
                  reference AND the source of the hardness signal.
  * classical_bm- src/bm.py's current classical segmenter (computed here, locally; Colab won't have it) =
                  the second eval baseline.
  * weight      - per-column loss weight 1 + a*min(|device-gold|/tau, cap): ~1 on accepted/flat columns,
                  up to ~10 on the GA-dive columns you pulled the device out of. Drives the training to
                  spend its capacity where the device is wrong (the whole point), and down-weights the
                  rubber-stamped accepted columns (only as good as the device there). Forced to 0 on
                  excluded columns (saturated 'white band' + impossible) so the model is never taught a
                  surface where BM is unknowable.
  * valid       - (N,W) bool ignore mask: False where the column is excluded (saturated machine-fill band or
                  impossible). weight is already 0 there; a trainer that does not multiply by `weight`
                  should mask the loss with `valid`.
  * edited      - whether you hand-corrected this B-scan (a bscan_<i>.json exists) vs accepted the device.
  * ilm/device_ilm - gold ILM (effective_surfaces[0]) and device ILM. Reference channels for a possible
                  future ILM-anchored experiment; the model is BM-only and current training ignores them.

Out (outputs/bm_dataset/, gitignored, regenerable):
  npz/<subject>_<eye>.npz   images uint8(N,H,W); bm/device_bm/classical_bm/weight/ilm/device_ilm
                            float32(N,W); valid bool(N,W); bscan_idx int32(N); edited bool(N);
                            H,W,fov_x_mm,fov_y_mm scalars.
  manifest.csv              one row per B-scan (provenance + hardness summary).
  splits.json               leave-one-patient-out folds (k = #patients; a patient's eyes share a fold).
  qc/*.png                  (--qc) gold/device/classical overlays on the hardest B-scans -> eyeball labels.
  bm_dataset.zip            npz/ + manifest.csv + splits.json, one-shot Colab upload.

Run from the repo root:
  oct_env\\Scripts\\python.exe src\\export_bm_dataset.py            # full export + zip
  oct_env\\Scripts\\python.exe src\\export_bm_dataset.py --qc       # + QC overlays (do this once, eyeball them)
  oct_env\\Scripts\\python.exe src\\export_bm_dataset.py --stats     # dry run: print label stats, write nothing
"""
import argparse
import csv
import json
import os
import sys
import zipfile

import cv2
import numpy as np

from paths import DATA_DIR, OUT_DIR, REPO_ROOT, RESULTS_DIR

sys.path.insert(0, REPO_ROOT)                          # so `import reader` resolves from a src/ script
import bm as bmseg                                      # noqa: E402  classical BM baseline (no torch)
from reader.core import e2e_source, render             # noqa: E402
from reader.core.layer_store import JsonSidecarLayerStore  # noqa: E402
from reader.core.layers import effective_surfaces      # noqa: E402

AXIAL_UM_PER_PX = 3.8716699928045273
OUT = os.path.join(OUT_DIR, "bm_dataset")
NPZ_DIR = os.path.join(OUT, "npz")
QC_DIR = os.path.join(OUT, "qc")
CORR_ROOT = os.path.join(REPO_ROOT, "reader", "data_store", "corrections")
WORKLIST = os.path.join(RESULTS_DIR, "bm_worklist.csv")

# Eyes are auto-discovered (see _discover_eyes): every device-BM eye in the worklist; export_eye() then
# keeps only those with >=1 validated B-scan, so the reader's growing BM-validation set flows in with no
# code edit. Override on the CLI with positional "SUBJECT EYE ..." pairs.

# Per-column hardness weight: w = 1 + ALPHA * min(|device-gold| / TAU_PX, CAP).
W_ALPHA, W_TAU_PX, W_CAP = 1.0, 10.0, 9.0
HARD_DELTA_PX = 10.0   # a column counts as "hard" where the device disagrees with gold by > this (~39 um)

# Label-sanity guard: BM must sit below ILM (the retina has thickness) and never in the top of the image.
# A column is anatomically impossible if BM is at/above ILM or in the top CEIL_PX rows. If >MAX_BAD_FRAC of
# a row is impossible the correction is corrupt -> drop that B-scan; a few stray columns -> revert them to
# device BM and don't supervise them (weight 0). Nothing on disk is touched; drops are logged for re-validation.
MIN_THICK_PX, CEIL_PX, MAX_BAD_FRAC = 8.0, 30.0, 0.2

MANIFEST_COLS = ["subject", "eye", "visit", "patient", "bscan", "edited", "n_hard_cols", "n_device_gap_cols",
                 "max_delta_px", "mean_weight", "bm_src", "n_bscans", "H", "W", "fov_x_mm", "fov_y_mm",
                 "advRPE_area_mm2", "npz", "row"]


# --------------------------------------------------------------------------- small helpers
def _patient(subject):
    p = subject.split("-")
    return p[2] if len(p) > 2 else subject


def _visit(subject):
    for p in subject.split("-"):
        if len(p) >= 2 and p[0] in "Vv" and p[1:].isdigit():
            return p.upper()
    return ""


def _find_6x6(raw, eye):
    refs = [r for r in raw.refs if r.eye == eye and getattr(r, "is_6x6", False)]
    return refs[0] if refs else None


def _worklist():
    """(subject, eye) -> {e2e_file, advRPE_area_mm2, has_device_bm} from results/bm_worklist.csv."""
    out = {}
    with open(WORKLIST, newline="") as f:
        for r in csv.DictReader(f):
            out[(r["subject"], r["eye"])] = r
    return out


def _discover_eyes(wl):
    """Candidate eyes for export: every device-BM eye in the worklist (sorted). export_eye() drops any
    with 0 validated B-scans, so this auto-tracks the reader's growing BM-validation set (no hardcoded list)."""
    eyes = [(s, e) for (s, e), r in wl.items()
            if str(r.get("has_device_bm", "")).strip().lower() == "true"]
    return sorted(eyes)


def _col_weight(delta_px):
    return 1.0 + W_ALPHA * np.minimum(np.abs(delta_px) / W_TAU_PX, W_CAP)


def _fill_nan(row):
    """Linear-interpolate any NaNs along the A-scan axis so stored surfaces are finite (defensive)."""
    row = np.asarray(row, float)
    m = np.isfinite(row)
    if m.all():
        return row
    if not m.any():
        return np.zeros_like(row)
    x = np.arange(len(row))
    row = row.copy()
    row[~m] = np.interp(x[~m], x[m], row[m])
    return row


def _interp_row(row, invalid_row=None):
    """Per-column linear interp of a RAW device BM row across its gaps (NaN / non-positive / saturated-band
    columns), with NO cross-B-scan smoothing — exactly the surface the reader's BM tab draws (ov.bm_display)
    and the human validated. m2_bm.fill_bm additionally 2D median+gaussian smooths, which dives up to ~40px
    from the device line across gap runs (e.g. 002 OD b75); that dive is never displayed, so it must never
    become a training label."""
    row = np.asarray(row, float).copy()
    m = np.isfinite(row) & (row > 0)
    if invalid_row is not None:
        m = m & ~np.asarray(invalid_row, bool)
    if 5 < m.sum() < len(row):
        row[~m] = np.interp(np.flatnonzero(~m), np.flatnonzero(m), row[m])
    return row


def _draw_overlay(gray, gold, dev, cls, hard):
    """gold=green, device=yellow, classical=cyan, hard columns faintly tinted red (BGR for cv2)."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    H, W = gray.shape
    if hard is not None and hard.any():
        tint = rgb.copy()
        tint[:, hard] = (0, 0, 255)
        rgb = cv2.addWeighted(rgb, 0.85, tint, 0.15, 0)

    def poly(curve, color):
        if curve is None:
            return
        pts = np.array([[x, int(round(np.clip(curve[x], 0, H - 1)))]
                        for x in range(W) if np.isfinite(curve[x])], np.int32)
        if len(pts) > 1:
            cv2.polylines(rgb, [pts], False, color, 1, cv2.LINE_AA)

    poly(cls, (255, 255, 0))    # classical (cyan)
    poly(dev, (0, 255, 255))    # device   (yellow)
    poly(gold, (0, 255, 0))     # gold     (green)
    return rgb


# --------------------------------------------------------------------------- per-eye export
def export_eye(subject, eye, meta, store, want_classical=False):
    """Return (summary_dict, manifest_rows, npz_dict) or None. Does NOT write (caller decides)."""
    e2e_file = meta["e2e_file"]
    path = e2e_file if os.path.isabs(e2e_file) else os.path.join(DATA_DIR, e2e_file)
    if not os.path.exists(path):
        print(f"  {subject} {eye}: SKIP (E2E not found: {path})")
        return None
    raw = e2e_source.open_e2e(path)
    ref = _find_6x6(raw, eye)
    if ref is None:
        print(f"  {subject} {eye}: SKIP (no 6x6 volume)")
        return None
    ov = e2e_source.load_volume(raw, ref.index)
    eid = ov.eid

    validated = store.bm_validated(eid, eye)
    if not validated:
        print(f"  {subject} {eye}: SKIP (0 validated B-scans for eid={eid} — corrections dir mismatch?)")
        return None
    corrected = set(store.corrected_indices(eid, eye))

    ilm_full, _ = effective_surfaces(ov, store)         # ILM (BM>ILM guard + reference); BM rebuilt below
    device_ilm = np.asarray(ov.ilm, float)              # device ILM (reference channel; model is BM-only)
    if ov.bm_src != "device":
        print(f"  {subject} {eye}: WARNING bm_src={ov.bm_src!r} (expected 'device')")

    invalid = ov.field_invalid                          # saturated machine-fill cols (BM unknowable there)
    if invalid is None:
        invalid = np.zeros((ov.n_bscans, ov.W), bool)
    # GOLD BM = the surface the human actually VALIDATED in the reader = the RAW device contour
    # (ov.bm_display, gaps linearly interpolated) + global shift + per-B-scan corrections. NOT the smoothed
    # ov.bm that effective_surfaces uses: m2_bm.fill_bm's 2D smoothing dives up to ~40px from the device
    # line across gap runs, and the reader's BM tab draws bm_display — so that dive was never seen/approved
    # (the cause of "reader good, notebook bad"). device_bm is that same raw contour (the true eval baseline).
    gb = float((store.get_global(eid, eye) or {}).get("bm", 0.0) or 0.0)
    device_bm = np.stack([_interp_row(ov.bm_display[i], invalid[i]) for i in range(ov.n_bscans)])
    bm_full = device_bm + gb
    # MERGE the correction OVER the device line (null -> keep device), exactly as the reader's
    # layers.effective_surfaces does. A stored correction can be SPARSE (nulls over device-covered cols):
    # "Fill blanks (DL)" writes only the former-gap columns, leaving every device-covered column null. The
    # old `bm_full[bi] = c["bm"]` replaced the row wholesale, so _fill_nan then INTERPOLATED across those
    # nulls -> a fabricated gold label where the human actually accepted the device BM. Merging makes the
    # gold = the surface the human validated (device where un-edited, correction where drawn/DL-filled).
    corr_finite = np.zeros((ov.n_bscans, ov.W), bool)        # (n,W) True = a human/DL value is set here
    for bi in range(ov.n_bscans):
        c = store.get_corrected(eid, eye, bi)
        if c and c.get("bm") is not None:
            cb = np.array([np.nan if v is None else v for v in c["bm"]], float)
            if cb.shape[0] == ov.W:
                m = np.isfinite(cb)
                bm_full[bi] = np.where(m, cb, bm_full[bi])    # nulls fall back to the validated device line
                corr_finite[bi] = m
    # Device GAPS = columns the device never segmented (raw bm_display NaN/non-positive). Their gold value is
    # only a linear-interp GUESS, so on UN-corrected B-scans we NEVER supervise it (weight 0) and we REPORT
    # those B-scans for re-validation — the human draws BM there to make a real label; until then the model
    # is not taught a guess. (A corrected B-scan's correction already spans the full width, so no gap.)
    raw_disp = np.asarray(ov.bm_display, float)
    device_gap = ~(np.isfinite(raw_disp) & (raw_disp > 0))   # (n,W) True = device did not segment this column
    # A column is truly UNLABELED only when the device left a gap AND no human/DL correction fills it. This
    # is the honest per-column rule (reader.core.layers._bm_missing_row): a B-scan merely being "corrected"
    # no longer implies a full-width label (sparse "Fill blanks (DL)" corrections exist), so we mask by the
    # ACTUAL coverage, not by "is this B-scan in `corrected`".
    unlabeled = device_gap & ~corr_finite                    # (n,W) True = no real BM here -> never supervise
    # Fold saturation into the integrity guard so the band is treated as impossible: those columns get
    # reverted to device AND weight 0 (NOT the device-vs-gold MAX weight that would teach the wrong surface).
    bad_all = (bm_full <= ilm_full + MIN_THICK_PX) | (bm_full < CEIL_PX) | invalid   # impossible cols (n,W)
    kept = [i for i in validated if bad_all[i].mean() <= MAX_BAD_FRAC]
    excluded = [i for i in validated if bad_all[i].mean() > MAX_BAD_FRAC]

    classical = None
    if want_classical:
        try:
            classical = bmseg.segment_volume(ov.vol)    # graph-search baseline over the whole volume
        except Exception as ex:                          # noqa: BLE001
            print(f"  {subject} {eye}: classical baseline failed ({type(ex).__name__}: {ex}); storing NaN")

    N, H, W = len(kept), ov.H, ov.W
    images = np.zeros((N, H, W), np.uint8)
    bm_arr = np.zeros((N, W), np.float32)
    dev_arr = np.zeros((N, W), np.float32)
    cls_arr = np.full((N, W), np.nan, np.float32)
    wt_arr = np.zeros((N, W), np.float32)
    valid_arr = np.ones((N, W), bool)               # False where the column is excluded (band/impossible)
    ilm_arr = np.zeros((N, W), np.float32)
    dev_ilm_arr = np.zeros((N, W), np.float32)
    idx_arr = np.zeros(N, np.int32)
    edited_arr = np.zeros(N, bool)

    pat, vis = _patient(subject), _visit(subject)
    adv = meta.get("advRPE_area_mm2", "")
    rows, n_repaired = [], 0
    for r, i in enumerate(kept):
        images[r] = render.bscan_model_input(ov, i)   # masked-contrast norm8 + saturated band NEUTRALIZED
        gold = _fill_nan(bm_full[i]).astype(np.float32)
        dev = _fill_nan(device_bm[i]).astype(np.float32)
        gap_i = unlabeled[i]                             # honest: device gap NOT covered by a finite correction
        bcol = bad_all[i] | gap_i                        # exclude impossible/saturated AND truly-unlabeled cols
        if bad_all[i].any():                             # revert a few stray impossible columns to device
            gold = gold.copy(); gold[bad_all[i]] = dev[bad_all[i]]; n_repaired += int(bad_all[i].sum())
        bm_arr[r], dev_arr[r] = gold, dev
        ilm_arr[r] = _fill_nan(ilm_full[i]).astype(np.float32)
        dev_ilm_arr[r] = _fill_nan(device_ilm[i]).astype(np.float32)
        if classical is not None:
            cls_arr[r] = _fill_nan(classical[i]).astype(np.float32)
        delta = np.abs(dev - gold)
        wt = _col_weight(delta).astype(np.float32)
        if bcol.any():
            wt[bcol] = 0.0                               # don't supervise reverted / saturated / device-gap cols
        wt_arr[r] = wt
        valid_arr[r] = ~bcol                            # ignore mask (saturation + impossible + device-gap cols)
        idx_arr[r] = i
        edited_arr[r] = i in corrected
        n_hard = int((delta > HARD_DELTA_PX).sum())
        rows.append({
            "subject": subject, "eye": eye, "visit": vis, "patient": pat, "bscan": i,
            "edited": int(i in corrected), "n_hard_cols": n_hard, "n_device_gap_cols": int(gap_i.sum()),
            "max_delta_px": round(float(delta.max()), 2),
            "mean_weight": round(float(wt_arr[r].mean()), 3), "bm_src": ov.bm_src, "n_bscans": ov.n_bscans,
            "H": H, "W": W, "fov_x_mm": round(float(ov.fov_mm[0]), 4), "fov_y_mm": round(float(ov.fov_mm[1]), 4),
            "advRPE_area_mm2": adv, "npz": f"{subject}_{eye}.npz", "row": r,
        })

    npz = {
        "images": images, "bm": bm_arr, "device_bm": dev_arr, "classical_bm": cls_arr, "weight": wt_arr,
        "valid": valid_arr,                             # (N,W) bool: False = excluded col (mask the loss)
        "ilm": ilm_arr, "device_ilm": dev_ilm_arr,
        "bscan_idx": idx_arr, "edited": edited_arr,
        "H": np.int32(H), "W": np.int32(W),
        "fov_x_mm": np.float32(ov.fov_mm[0]), "fov_y_mm": np.float32(ov.fov_mm[1]),
    }
    n_edited = int(edited_arr.sum())
    n_hard_b = int(sum(1 for rr in rows if rr["n_hard_cols"] > 0))
    gap_rows = sorted(((rr["bscan"], rr["n_device_gap_cols"]) for rr in rows if rr["n_device_gap_cols"] > 0),
                      key=lambda x: -x[1])
    n_gap_cols = sum(g for _, g in gap_rows)
    summary = {
        "subject": subject, "eye": eye, "patient": pat, "n_bscans": N, "n_edited": n_edited,
        "n_accepted": N - n_edited, "n_hard_bscans": n_hard_b, "n_excluded": len(excluded),
        "n_repaired_cols": n_repaired, "n_gap_bscans": len(gap_rows), "n_gap_cols": n_gap_cols, "bm_src": ov.bm_src,
        "bm_row_min": round(float(bm_arr.min()), 1), "bm_row_med": round(float(np.median(bm_arr)), 1),
        "bm_row_max": round(float(bm_arr.max()), 1), "H": H, "W": W,
        "fov_x_mm": round(float(ov.fov_mm[0]), 3), "advRPE_area_mm2": adv,
    }
    print(f"  {subject} {eye}: N={N} edited={n_edited} accepted={N - n_edited} hard_bscans={n_hard_b} "
          f"bm_src={ov.bm_src} H={H} W={W} fov_x={summary['fov_x_mm']}mm "
          f"bm_row[min/med/max]={summary['bm_row_min']}/{summary['bm_row_med']}/{summary['bm_row_max']}")
    if excluded or n_repaired:
        print(f"  {subject} {eye}: LABEL GUARD -> dropped B-scans {excluded} (corrupt BM correction); "
              f"reverted {n_repaired} stray cols to device. Re-validate these in the reader for a clean label.")
    if gap_rows:
        print(f"  {subject} {eye}: DEVICE GAPS (interp guess — NOT supervised; re-validate to LABEL these): "
              f"{len(gap_rows)} B-scans / {n_gap_cols} cols. Worst: {['b%d:%d' % (b, g) for b, g in gap_rows[:12]]}")
    return summary, rows, npz


def write_qc(subject, eye, npz, k_hard=6, k_flat=2):
    """Overlay gold/device/classical on the k_hard hardest + k_flat flattest B-scans -> qc/*.png."""
    os.makedirs(QC_DIR, exist_ok=True)
    bm, dev, cls = npz["bm"], npz["device_bm"], npz["classical_bm"]
    idx, images = npz["bscan_idx"], npz["images"]
    delta_max = np.abs(dev - bm).max(axis=1)
    order = np.argsort(-delta_max)                       # hardest first
    pick = list(order[:k_hard]) + list(order[order.size - k_flat:][::-1])
    for r in pick:
        hard = np.abs(dev[r] - bm[r]) > HARD_DELTA_PX
        clr = None if not np.isfinite(cls[r]).any() else cls[r]
        ov = _draw_overlay(images[r], bm[r], dev[r], clr, hard)
        name = f"{subject}_{eye}_b{int(idx[r]):04d}_d{delta_max[r]:05.1f}px.png"
        cv2.imwrite(os.path.join(QC_DIR, name), ov)
    return len(pick)


def _safe_write(final_path, write_fn):
    """Write content via a temp file then atomically replace `final_path`, tolerating a LOCKED target
    (outputs/ is under OneDrive and these files get opened in Excel mid-run). If the target is locked,
    leave the fresh content at `<final>.new` and warn instead of crashing a multi-minute export. Returns
    the path actually written (so build_zip can pack the fresh copy regardless of its name)."""
    tmp = final_path + ".tmp"
    write_fn(tmp)
    try:
        os.replace(tmp, final_path)
        return final_path
    except PermissionError:
        alt = final_path + ".new"
        os.replace(tmp, alt)                          # the .new name is ours, not the one Excel holds
        print(f"  WARNING: {os.path.basename(final_path)} is locked (open in Excel / OneDrive sync?) -> "
              f"wrote {os.path.basename(alt)} instead; close it + re-run for the clean name.")
        return alt


def write_splits(summaries):
    """True leave-one-patient-out folds (k = #patients; no eye/visit leakage — an eye's fellow eye /
    other visits share its patient fold). Returns the path written (resilient to a locked target)."""
    pats = sorted({s["patient"] for s in summaries})
    k = max(1, len(pats))
    fold_of = {p: i % k for i, p in enumerate(pats)}
    folds = {str(f): [] for f in range(k)}
    eye_fold = {}
    for s in summaries:
        key = f"{s['subject']}_{s['eye']}"
        f = fold_of[s["patient"]]
        folds[str(f)].append(key)
        eye_fold[key] = f
    data = {"k": k, "by": "patient", "folds": folds, "eye_fold": eye_fold, "patient_fold": fold_of}

    def _wr(p):
        with open(p, "w") as fh:
            json.dump(data, fh, indent=1)
    return _safe_write(os.path.join(OUT, "splits.json"), _wr)


def build_zip(manifest_path=None, splits_path=None):
    """Pack npz/ + the manifest + splits into bm_dataset.zip (resilient to a locked zip). The manifest/
    splits paths may be the `.new` fallbacks (see _safe_write); they're still stored under the clean
    arcnames inside the zip, so the zip is always correct even when Excel holds the loose files."""
    zpath = os.path.join(OUT, "bm_dataset.zip")
    mpath = manifest_path or os.path.join(OUT, "manifest.csv")
    spath = splits_path or os.path.join(OUT, "splits.json")

    def _wr(p):
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(os.listdir(NPZ_DIR)):
                if f.endswith(".npz"):
                    z.write(os.path.join(NPZ_DIR, f), arcname=f"npz/{f}")
            for src, arc in ((mpath, "manifest.csv"), (spath, "splits.json")):
                if src and os.path.exists(src):
                    z.write(src, arcname=arc)
    out = _safe_write(zpath, _wr)
    print(f"  zip -> {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    return out


# --------------------------------------------------------------------------- CLI
def run_export(pairs=None, want_classical=False, want_zip=True, want_qc=False, write=True):
    """Export the validated BM eyes -> npz/ + manifest.csv + splits.json (+ bm_dataset.zip). Reused by
    the CLI (main) and the reader's 'Export BM dataset' button. `pairs`: optional [(subject,eye), ...]
    to limit the eyes. Returns a summary dict (or None if nothing was exported)."""
    if write:
        os.makedirs(NPZ_DIR, exist_ok=True)
    wl = _worklist()
    eyes = pairs if pairs else _discover_eyes(wl)
    print(f"{len(eyes)} candidate eyes (device-BM); keeping those with >=1 validated B-scan")
    summaries, manifest = [], []
    for subject, eye in eyes:
        meta = wl.get((subject, eye))
        if meta is None:
            print(f"  {subject} {eye}: SKIP (not in bm_worklist.csv)")
            continue
        res = export_eye(subject, eye, meta, JsonSidecarLayerStore(CORR_ROOT), want_classical=want_classical)
        if res is None:
            continue
        summary, rows, npz = res
        summaries.append(summary)
        manifest.extend(rows)
        if write:
            np.savez_compressed(os.path.join(NPZ_DIR, f"{subject}_{eye}.npz"), **npz)
        if want_qc:
            n = write_qc(subject, eye, npz)
            print(f"  {subject} {eye}: wrote {n} QC overlays")

    if not summaries:
        print("No eyes exported.")
        return None

    tot = sum(s["n_bscans"] for s in summaries)
    ed = sum(s["n_edited"] for s in summaries)
    exc = sum(s.get("n_excluded", 0) for s in summaries)
    rep = sum(s.get("n_repaired_cols", 0) for s in summaries)
    n_pat = len({s["patient"] for s in summaries})
    guard = f"  [guard: {exc} B-scans dropped, {rep} cols reverted]" if (exc or rep) else ""
    print(f"\n{len(summaries)} eyes, {tot} B-scans ({ed} edited / {tot - ed} accepted), "
          f"{n_pat} patients.{guard}")

    zip_path = None
    if write:
        def _wr_manifest(p):
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=MANIFEST_COLS, extrasaction="ignore")
                w.writeheader(); w.writerows(manifest)
        mpath = _safe_write(os.path.join(OUT, "manifest.csv"), _wr_manifest)
        spath = write_splits(summaries)
        if want_zip:
            zip_path = build_zip(mpath, spath)
        print(f"DONE -> {OUT}/  (npz/ manifest.csv splits.json"
              f"{' qc/' if want_qc else ''}{'' if not want_zip else ' bm_dataset.zip'})")
    else:
        print("(--stats: nothing written)")
    return {"n_eyes": len(summaries), "n_bscans": tot, "n_edited": ed, "n_patients": n_pat,
            "n_excluded": exc, "n_repaired_cols": rep, "out_dir": OUT, "zip_path": zip_path}


def main():
    ap = argparse.ArgumentParser(description="Export the validated BM eyes for DL training.")
    ap.add_argument("pairs", nargs="*",
                    help="optional SUBJECT EYE ... to limit the eyes (default: all validated device-BM eyes)")
    ap.add_argument("--qc", action="store_true", help="also write QC overlay PNGs (eyeball the labels)")
    ap.add_argument("--stats", action="store_true", help="dry run: print label stats, write nothing")
    ap.add_argument("--classical", action="store_true",
                    help="ALSO compute the classical bm.py graph-search baseline (slow; OFF by default)")
    ap.add_argument("--no-zip", action="store_true", help="skip building bm_dataset.zip")
    args = ap.parse_args()

    pairs = ([(args.pairs[i], args.pairs[i + 1]) for i in range(0, len(args.pairs) - 1, 2)]
             if args.pairs else None)
    run_export(pairs=pairs, want_classical=(args.classical and not args.stats),
               want_zip=not args.no_zip, want_qc=args.qc, write=not args.stats)


if __name__ == "__main__":
    main()
