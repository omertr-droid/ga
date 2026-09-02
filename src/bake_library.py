"""Bake the doctor viewer's built-in library: one compact bundle per eye that has a PLEX GA measurement.

Run ONCE on the dev machine (has the E2Es + advRPE references + oct_env). The library set = every pairing
row with qc_status==ok AND a PLEX GA value (advRPE_area_mm2 present, non-NaN). Controls (value 0.0) are
KEPT — a 0 is a real measurement; only NaN/missing is dropped (e.g. 013 OS). For each such eye it:
  - opens the E2E, loads the 6x6 (97-line) volume, applies the validated/device BM (effective_surfaces),
  - runs the OAC GA detector (our OCT-only number) + the real per-B-scan locator lines,
  - ALSO recomputes GA on the DL Bruch's-membrane and caches that variant in the bundle (the viewer's
    default view + the offline DL toggle — no E2E/model needed at serve time),
  - registers the advRPE (PLEX) GA -> a reference outline + reads PLEX's own area (NEVER blended),
  - writes viewer/data_store/library/<subject>_<eye>/ (meta.json + bundle.npz + PNGs),
prunes bundles no longer in the library set, and writes index.json.

Usage (from repo root):
  oct_env\\Scripts\\python.exe src\\bake_library.py                 # bake the full library set (qc_ok + PLEX)
  oct_env\\Scripts\\python.exe src\\bake_library.py --only 005      # just eyes whose slug contains '005' (no prune)
  oct_env\\Scripts\\python.exe src\\bake_library.py --qc            # also dump locator QC overlays
  oct_env\\Scripts\\python.exe src\\bake_library.py --reindex       # rebuild index.json + prune, no E2E reopen
The shipped app then needs NONE of this code path for library scans — only numpy/PIL/cv2.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from reader.core import e2e_source, fieldmask, layers as core_layers, projection as proj   # noqa: E402
from reader.core.layer_store import JsonSidecarLayerStore                            # noqa: E402
from viewer.core import bundle, locator, plex, viewmodel                             # noqa: E402

DATA = os.path.join(_REPO, "data")
PAIRING = os.path.join(_REPO, "results", "spectralis_ga_pairing.csv")
CORR_DIR = os.path.join(_REPO, "reader", "data_store", "corrections")
COHORT = os.path.join(_REPO, "cohort")
QC_DIR = os.path.join(_REPO, "outputs", "viewer_bake_qc")


def _has_plex(row):
    """True if the row carries a real PLEX GA value (non-NaN). 0.0 counts; only NaN/missing is dropped."""
    try:
        return not math.isnan(float(row.get("advRPE_area_mm2")))
    except (TypeError, ValueError):
        return False


def library_rows():
    """Pairing-CSV rows that belong in the doctor library: qc_status==ok AND a PLEX GA value (controls
    with value 0.0 kept). The library is defined purely by 'has a PLEX GA measurement, in scope'."""
    with open(PAIRING, newline="") as f:
        return [r for r in csv.DictReader(f)
                if (r.get("qc_status") or "").strip() == "ok" and _has_plex(r)]


def library_slugs():
    return {bundle.slug_for(r["subject"], r["eye"]) for r in library_rows()}


def prune_stale(keep):
    """Delete every library bundle dir whose slug is NOT in `keep` (e.g. now out-of-scope 009/017).
    Returns the slugs removed."""
    import shutil
    removed = []
    for d in glob.glob(os.path.join(bundle.LIBRARY_DIR, "*")):
        if not os.path.isdir(d):
            continue
        slug = os.path.basename(d)
        if slug not in keep:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(slug)
            print(f"  pruned stale bundle {slug}")
    return removed


def attach_plex_corrections(folder):
    """Copy each doctor's PLEX-GA-correction image (named `<patient>_<eye>.png`, e.g. `001_OD.png`) into
    the matching library bundle as `plex_correction.png` and set meta['has_plex_correction']=True. The
    viewer then shows it ABOVE the projection (a reference, not registered/blended). Rules:
      - only eyes already in the library get it (matched by patient number + eye → bundle slug);
      - an image with no matching library eye is skipped (NO scan is added);
      - library eyes with no image are left unchanged.
    No bake / no E2E — just patches existing bundles. Restart the viewer (or re-package) to pick it up."""
    import re
    import shutil
    imgs = sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))
    if not imgs:
        print(f"no images found in {folder}")
        return
    dirs = {os.path.basename(d): d for d in glob.glob(os.path.join(bundle.LIBRARY_DIR, "*"))
            if os.path.isdir(d)}
    attached, skipped = 0, []
    for img in imgs:
        stem = os.path.splitext(os.path.basename(img))[0]
        m = re.match(r"^(\d+)[ _-]+(O[DS])$", stem, re.IGNORECASE)
        if not m:
            skipped.append((stem, "unparsable filename (want <patient>_<eye>)"))
            continue
        patient, eye = m.group(1), m.group(2).upper()
        cand = [s for s in dirs if f"-{patient}-V" in s and s.endswith("_" + eye)]   # one visit/patient
        if not cand:
            skipped.append((stem, "no matching library eye — not added"))
            continue
        slug = cand[0]
        shutil.copyfile(img, os.path.join(dirs[slug], "plex_correction.png"))
        mp = os.path.join(dirs[slug], "meta.json")
        meta = json.load(open(mp))
        meta["has_plex_correction"] = True
        with open(mp, "w") as f:
            json.dump(meta, f, indent=2)
        attached += 1
        print(f"  attached {stem} -> {slug}")
    for stem, why in skipped:
        print(f"  skip {stem}: {why}")
    print(f"attached {attached} correction(s); skipped {len(skipped)}")


def plex_paths(subject, eye):
    edir = os.path.join(COHORT, subject, eye)
    return (os.path.join(edir, "advrpe_subrpe_enface.png"),
            os.path.join(edir, "ga_mask.png"))


def index_row(m):
    """One library-list row: identity + areas (ours on the DL BM + PLEX) for the comparison cards."""
    return {
        "slug": m["slug"], "subject": m["subject"], "eye": m["eye"],
        "patient_id": m.get("patient_id"), "visit": m.get("visit"), "acq_date": m.get("acq_date"),
        "n_bscans": m.get("n_bscans"), "fov_mm": m.get("fov_mm"),
        "oac_area_mm2": m.get("oac_area_mm2"), "oac_area_dl_mm2": m.get("oac_area_dl_mm2"),
        "plex_area_mm2": m.get("plex_area_mm2"), "is_control": m.get("is_control"),
        "bm_dl_baked": m.get("bm_dl_baked"),
    }


def reindex():
    """Prune bundles no longer in the library set, then rebuild index.json from the remaining bundles'
    meta.json (no E2E reopen) — the fast way to refresh the library after a QC change."""
    keep = library_slugs()
    prune_stale(keep)
    rows = []
    for d in sorted(glob.glob(os.path.join(bundle.LIBRARY_DIR, "*"))):
        mp = os.path.join(d, "meta.json")
        if not os.path.isdir(d) or not os.path.exists(mp):
            continue
        rows.append(index_row(json.load(open(mp))))
    bundle.write_index(sorted(rows, key=lambda r: r["slug"]))
    print(f"reindexed {len(rows)} eye(s) -> {bundle.LIBRARY_DIR}/index.json")


def qc_overlay(loc_png, loc_lines, n, path):
    img = cv2.imdecode(np.frombuffer(loc_png, np.uint8), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mid = n // 2
    for i, (x1, y1, x2, y2) in enumerate(loc_lines):
        col = (0, 255, 0) if i == mid else (110, 110, 110)
        cv2.line(rgb, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
                 col, 2 if i == mid else 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, rgb)


def bake_eye(row, store, want_qc=False):
    subject, eye = row["subject"], row["eye"]
    e2e_path = os.path.join(DATA, row["e2e_file"])
    slug = bundle.slug_for(subject, eye)
    raw = e2e_source.open_e2e(e2e_path)
    idx = e2e_source.default_volume_index(raw, eye)
    ov = e2e_source.load_volume(raw, idx)
    ilm, bm = core_layers.effective_surfaces(ov, store)              # validated/device BM (the base view)

    vm = viewmodel.compute(raw, ov, ilm, bm)

    # cached DL Bruch's-membrane variant — the viewer's default view + the offline toggle. Additive: the
    # base stays the validated/device computation; DL only recomputes the BM-dependent layers (ILM kept).
    vm_dl, dl_area = None, None
    try:
        import bm_dl
        if bm_dl.available():
            bm_dl_surf = bm_dl.segment_volume(ov.vol).astype(np.float32)
            vm_dl = viewmodel.compute(raw, ov, ilm, bm_dl_surf)
            dl_area = round(float(vm_dl["oac_area_mm2"]), 4)
        else:
            print("    (DL BM not baked: no model available — re-bake once outputs/bm_dl is present)")
    except Exception as e:                                           # noqa: BLE001
        print(f"    (DL BM not baked: {e!r})")
        vm_dl = None

    # PLEX (advRPE) reference: outline (two-map lock) + area (straight from the pairing CSV).
    subrpe, ga_mask = plex_paths(subject, eye)
    polys = []
    if os.path.exists(subrpe) and os.path.exists(ga_mask):
        label = plex.registered_label(ov.vol, bm, ov.fov_mm, subrpe, ga_mask,
                                      getattr(ov, "enface_flip", True))
        polys = plex.outline_polygons(label, vm["out"])
    try:
        plex_area = round(float(row.get("advRPE_area_mm2")), 4)
    except (TypeError, ValueError):
        plex_area = None

    patient_id = getattr(raw.vols[idx], "patient_id", None)
    fvm = fieldmask.eye_metrics(ov.field_invalid, ov.fov_mm)   # saturated-band extent (area = lower bound)
    meta = {
        "schema": bundle.SCHEMA, "slug": slug, "subject": subject, "visit": row.get("visit"),
        "eye": eye, "patient_id": patient_id, "acq_date": row.get("date"),
        "n_bscans": ov.n_bscans, "H": ov.H, "W": ov.W, "fov_mm": [round(float(x), 3) for x in ov.fov_mm],
        "axial_um_per_px": viewmodel.AXIAL_UM_PER_PX, "enface_mmpp": float(proj.ENFACE_MMPP),
        "enface_out": int(vm["out"]), "feature": viewmodel.FEATURE,
        "slab_um": [float(x) for x in viewmodel.SLAB_UM],
        "oac_area_mm2": round(float(vm["oac_area_mm2"]), 4),
        "oac_area_dl_mm2": dl_area,                               # our GA on the DL BM (None if not baked)
        "bm_dl_baked": bool(vm_dl is not None),
        "bm_source": getattr(ov, "bm_src", None),                # the base view's BM source (device/auto)
        "plex_area_mm2": plex_area, "plex_source": "spectralis_ga_pairing.advRPE_area_mm2",
        "plex_polygons": polys, "localizer_sid": vm["localizer_sid"],
        # False on a reverse-scanned raster (003-016/003-130): the en-face rows were not flipped.
        "enface_flip": bool(vm.get("enface_flip", True)),
        "is_control": bool(plex_area is not None and plex_area < 0.05),
        "sat_band_bscans": fvm["n_bscans_with_band"],
        "sat_band_max_width_mm": fvm["max_band_width_mm"],
        "area_is_lower_bound": bool(fvm["n_bscans_with_band"] > 0),
    }
    bundle.write_bundle(slug, vm, meta, vm_dl=vm_dl)
    if want_qc and vm.get("localizer_png") is not None and vm.get("loc_lines") is not None:
        qc_overlay(vm["localizer_png"], vm["loc_lines"], ov.n_bscans,
                   os.path.join(QC_DIR, f"{slug}_loc.png"))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter on subject/slug (e.g. 005); no prune")
    ap.add_argument("--qc", action="store_true", help="also write locator QC overlays")
    ap.add_argument("--reindex", action="store_true",
                    help="just rebuild index.json from existing bundles' meta + prune (no E2E reopen)")
    ap.add_argument("--attach-plex-corrections", default=None, metavar="DIR",
                    help="copy doctor PLEX-GA-correction images (<patient>_<eye>.png) from DIR into the "
                         "matching library bundles (no bake/E2E); unmatched images skipped")
    args = ap.parse_args()

    if args.attach_plex_corrections:
        attach_plex_corrections(args.attach_plex_corrections)
        return

    if args.reindex:
        reindex()
        return

    store = JsonSidecarLayerStore(CORR_DIR)
    rows = library_rows()
    intended = {bundle.slug_for(r["subject"], r["eye"]) for r in rows}
    print(f"library set: {len(intended)} qc_ok eyes with a PLEX GA value (controls kept)")

    # carry forward existing index rows ONLY for eyes still in the library set (drops stale eyes)
    by_slug = {r["slug"]: r for r in bundle.read_index() if r["slug"] in intended}
    baked = 0
    for row in rows:
        slug = bundle.slug_for(row["subject"], row["eye"])
        if args.only and args.only.lower() not in slug.lower():
            continue
        e2e_path = os.path.join(DATA, row["e2e_file"])
        if not os.path.exists(e2e_path):
            print(f"  SKIP {row['subject']} {row['eye']}: E2E missing (keeping any existing bundle)")
            continue
        print(f"  baking {row['subject']} {row['eye']} …", flush=True)
        try:
            meta = bake_eye(row, store, want_qc=args.qc)
        except Exception as e:                                     # noqa: BLE001
            import traceback
            print(f"  FAILED {row['subject']} {row['eye']}: {e!r}")
            traceback.print_exc()
            continue
        by_slug[meta["slug"]] = index_row(meta)
        baked += 1
        dl = f"DL {meta['oac_area_dl_mm2']}" if meta.get("bm_dl_baked") else "DL n/a"
        print(f"    -> OAC {meta['oac_area_mm2']} mm² ({dl}) · PLEX {meta['plex_area_mm2']} mm² · "
              f"{len(meta['plex_polygons'])} outline(s)")

    # a full run makes the library set authoritative: drop bundles no longer in it (e.g. 009/017).
    # a --only run is partial, so it never prunes.
    if not args.only:
        prune_stale(intended)

    rows_out = sorted(by_slug.values(), key=lambda r: r["slug"])
    bundle.write_index(rows_out)
    print(f"baked {baked} eye(s); index has {len(rows_out)} entr(y/ies) -> {bundle.LIBRARY_DIR}")


if __name__ == "__main__":
    main()
