#!/usr/bin/env python
"""Locked 2x2 GA experiment: scan support x OCT radiometry.

This is an isolated research runner.  It never rewrites viewer bundles or changes detector defaults.
The parity arm reproduces the shipped 97-line/display-space/DL-BM result; the other arms change exactly
one of:

  * intensity: Spectralis display/log values vs inverse-log linear values (globally gain-matched),
  * support: current native 97-line field vs full 30-degree prep followed by scan-centred 6-mm counting.

PLEX is read only after OCT masks are complete and is used solely for evaluation.  Hand-adjudicated
PLEX false positives (006 OD and 010 OD) remain in the cohort as corrected negative controls.

Run from the repository root::

  oct_env\\Scripts\\python.exe src\\ga_experiment_matrix.py --self-test
  oct_env\\Scripts\\python.exe src\\ga_experiment_matrix.py --only 005 --protocol current97 --intensity display
  oct_env\\Scripts\\python.exe src\\ga_experiment_matrix.py
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import measure

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bm_dl  # noqa: E402
import m3_projections as mp  # noqa: E402
import register_qc as reg  # noqa: E402
from paths import DATA_DIR, OUT_DIR, RESULTS_DIR  # noqa: E402
from reader.core import e2e_source, fieldmask, oac_ga  # noqa: E402
from reader.core import projection as proj  # noqa: E402
from viewer.core import ga_native  # noqa: E402


SCHEMA = "ga-experiment-matrix-v2"
INFERENCE_SCHEMA = "ga-oct-inference-v2-series-geometry-fixed-core"
BOOTSTRAP_SEED = 20260711
INDEX = os.path.join(_REPO, "viewer", "data_store", "library", "index.json")
PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
ADJUDICATION = os.path.join(RESULTS_DIR, "plex_adjudication.csv")
LIBRARY = os.path.join(_REPO, "viewer", "data_store", "library")
CACHE_DIR = os.path.join(OUT_DIR, "ga_experiment_cache")
MANIFEST_CSV = os.path.join(RESULTS_DIR, "ga_experiment_manifest.csv")
MATRIX_CSV = os.path.join(RESULTS_DIR, "ga_experiment_matrix.csv")
STAGES_CSV = os.path.join(RESULTS_DIR, "ga_experiment_stages.csv")
COMPONENTS_CSV = os.path.join(RESULTS_DIR, "ga_experiment_components.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "ga_experiment_summary.csv")
REPORT_MD = os.path.join(RESULTS_DIR, "ga_experiment_report.md")

# Every production argument is explicit.  A code-default change cannot silently alter this experiment.
PREP_CONFIG = {
    "reducer": "mean",
    "smooth_px": 2.0,
    "margin_mm": 0.30,
    "baseline": "radial2",
    "trend_order": 2,
    "rpe_hi_pct": 95.0,
    "sig_frac": 0.5,
    "base_cap": 1.15,
    "radial": False,
    "ilm": None,
    "rpe_band": "fixed",
    "noise_floor": False,
    "field_valid": None,
    "quality": False,
}
FOOT_CONFIG = {
    "frac": 0.50,
    "min_diam_um": 250.0,
    "hyper_fill": True,
    "close_mm": 0.15,
    "hyper_frac": 0.7,
    "hyper_keep": 0.4,
    "fill_all_holes": True,
    "hyper_abs": 0.10,
    "min_depth": 0.27,
}
STAGE_KEYS = (
    "rpe_candidate", "hyper_kept", "hyper_rejected", "holes_candidate", "holes_filled",
    "filled", "crora_hole_cleaned", "sized", "partial_rejected", "final",
)
PATIENT_RE = re.compile(r"^NHAMD-003-(\d+)-V\d+$")
LINEAR_A = 8.285
LINEAR_B = 8.3
LINEAR_OFFSET = 2.44e-4
PARITY_TOL_MM2 = 5.01e-5       # baked JSON is rounded to four decimals; no full-pixel tolerance


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    x = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode("ascii"))
    h.update(json.dumps(x.shape).encode("ascii"))
    h.update(x.view(np.uint8))
    return h.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite_float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _patient_key(subject: str) -> str:
    m = PATIENT_RE.match(subject)
    if not m:
        raise ValueError(f"malformed cohort subject: {subject!r}")
    return m.group(1)


def _read_csv(path: str, comments: bool = False) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [ln for ln in f if not (comments and ln.lstrip().startswith("#"))]
    return list(csv.DictReader(lines))


def _write_csv_atomic(path: str, rows: list[dict], fieldnames: list[str] | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _reference_class(area: float | None, verdict: str = "") -> str:
    if verdict == "plex_false_positive":
        return "negative"
    if verdict == "plex_false_negative":
        return "positive"
    if verdict.startswith("plex_partial_"):
        return "positive"                  # a partial verdict explicitly confirms real GA is present
    if area is None:
        return "unknown"
    if area < 0.05:
        return "negative"
    if area >= 0.25:
        return "positive"
    return "indeterminate"


def load_manifest() -> list[dict]:
    """Strictly join the baked 25-eye index to the QC pairing and hand adjudication."""
    with open(INDEX, encoding="utf-8") as f:
        baked = json.load(f)
    pairing = {(r["subject"], r["eye"].upper()): r for r in _read_csv(PAIRING)}
    adjudication = {(r["subject"], r["eye"].upper()): r
                    for r in _read_csv(ADJUDICATION, comments=True)}
    rows = []
    for idx in baked:
        subject, eye = idx["subject"], idx["eye"].upper()
        key = (subject, eye)
        if key not in pairing:
            raise KeyError(f"baked eye missing from pairing: {key}")
        p = pairing[key]
        if (p.get("qc_status") or "").strip() != "ok":
            raise ValueError(f"baked eye is not qc_status=ok: {key}")
        raw_area = _finite_float(idx.get("plex_area_mm2"))
        if raw_area is None:
            raise ValueError(f"baked eye has no PLEX area: {key}")
        adj = adjudication.get(key, {})
        verdict = (adj.get("verdict") or "").strip()
        corrected = (0.0 if verdict == "plex_false_positive" else
                     None if verdict == "plex_false_negative" else raw_area)
        e2e_file = p["e2e_file"]
        e2e_path = e2e_file if os.path.isabs(e2e_file) else os.path.join(DATA_DIR, e2e_file)
        slug = idx["slug"]
        bundle_path = os.path.join(LIBRARY, slug, "bundle.npz")
        e2e_stat = os.stat(e2e_path)
        bundle_stat = os.stat(bundle_path)
        rows.append({
            "schema": SCHEMA,
            "slug": slug,
            "subject": subject,
            "patient": _patient_key(subject),
            "patient_id_display": idx.get("patient_id", ""),
            "visit": idx.get("visit", p.get("visit", "")),
            "eye": eye,
            "e2e_file": e2e_file,
            "e2e_path": os.path.abspath(e2e_path),
            "bundle_path": os.path.abspath(bundle_path),
            "e2e_size_bytes": int(e2e_stat.st_size),
            "e2e_mtime_ns": int(e2e_stat.st_mtime_ns),
            "bundle_size_bytes": int(bundle_stat.st_size),
            "bundle_mtime_ns": int(bundle_stat.st_mtime_ns),
            "bundle_sha256": sha256_file(bundle_path),
            "baked_area_mm2": float(idx.get("oac_area_dl_mm2", idx.get("oac_area_mm2"))),
            "plex_raw_mm2": raw_area,
            "plex_corrected_mm2": corrected,
            "ref_class_raw": _reference_class(raw_area),
            "ref_class_corrected": _reference_class(corrected, verdict),
            "adjudication_verdict": verdict,
            "adjudication_note": adj.get("note", ""),
            "reference_partial": verdict.startswith("plex_partial_"),
            "area_score_included": verdict != "plex_false_negative",
            "area_lower_bound": subject == "NHAMD-003-008-V1" and eye == "OD",
            "has_inframe_gold": subject == "NHAMD-003-005-V3" and eye == "OD",
            "qc_status": p.get("qc_status", ""),
            "qc_reason": p.get("qc_reason", ""),
        })
    rows.sort(key=lambda r: r["slug"])
    if len(rows) != 25:
        raise AssertionError(f"expected baked 25-eye cohort, found {len(rows)}")
    if {(r["subject"], r["eye"]) for r in rows if r["adjudication_verdict"] == "plex_false_positive"} != {
            ("NHAMD-003-006-V3", "OD"), ("NHAMD-003-010-V1", "OD")}:
        raise AssertionError("false-positive reference guardrails drifted")
    return rows


def write_manifest(rows: list[dict]):
    # Avoid leaking machine-specific absolute paths into the compact audit table.
    public = [{k: v for k, v in r.items() if k not in ("e2e_path", "bundle_path")} for r in rows]
    _write_csv_atomic(MANIFEST_CSV, public)


def experiment_provenance() -> dict:
    model = bm_dl.model_path()
    inference = {
        "inference_schema": INFERENCE_SCHEMA,
        "prep": PREP_CONFIG,
        "footprint": FOOT_CONFIG,
        "oac_hyper_um": list(oac_ga.OAC_HYPER_UM),
        "oac_rpe_um": list(mp.OAC_RPE_UM),
        "enface_mmpp": proj.ENFACE_MMPP,
        "linear_inverse": {"a": LINEAR_A, "b": LINEAR_B, "offset": LINEAR_OFFSET,
                           "gain_match": "median BM[-40,+10] to display eye median"},
        "inference_sources": {
            "oac_ga_sha256": sha256_file(oac_ga.__file__),
            "footprint_sha256": sha256_file(os.path.join(_REPO, "reader", "core", "footprint.py")),
            "m3_projections_sha256": sha256_file(mp.__file__),
            "bm_model": os.path.relpath(model, _REPO) if model else None,
            "bm_model_sha256": sha256_file(model),
        },
    }
    evaluation = {
        "index_sha256": sha256_file(INDEX),
        "pairing_sha256": sha256_file(PAIRING),
        "adjudication_sha256": sha256_file(ADJUDICATION),
    }
    payload = {
        "schema": SCHEMA,
        "inference": inference,
        "evaluation": evaluation,
        "runner_sha256": sha256_file(__file__),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
    }
    payload["inference_hash"] = _sha256_bytes(canonical_json(inference).encode("utf-8"))[:16]
    payload["evaluation_hash"] = _sha256_bytes(canonical_json(evaluation).encode("utf-8"))[:16]
    payload["config_hash"] = payload["inference_hash"]       # backwards-compatible CSV field name
    return payload


def _series_bscan_records(raw) -> dict[int, list[dict]]:
    """Read type-10004 position records while preserving their E2E ``series_id``.

    ``oct_converter.read_all_metadata`` drops the enclosing chunk identity, which made same-shaped eyes
    ambiguous. This is the same read-only directory traversal, retaining only the missing series key.
    """
    cached = getattr(raw, "_experiment_series_records", None)
    if cached is not None:
        return cached
    from oct_converter.readers.binary_structs import e2e_binary
    out = defaultdict(list)
    reader = raw.reader
    with open(reader.filepath, "rb") as f:
        chunk_stack = []
        for position in reader.directory_stack:
            f.seek(position + reader.byte_skip)
            directory = e2e_binary.main_directory_structure.parse(f.read(52))
            for _ in range(directory.num_entries):
                sub = e2e_binary.sub_directory_structure.parse(f.read(44))
                if sub.start > sub.pos:
                    chunk_stack.append(sub.start)
        for start in chunk_stack:
            f.seek(start + reader.byte_skip)
            try:
                chunk = e2e_binary.chunk_structure.parse(f.read(60))
            except Exception:
                continue
            if int(chunk.type) != 10004:
                continue
            try:
                meta = e2e_binary.bscan_metadata.parse(f.read(104))
            except Exception:
                continue
            out[int(chunk.series_id)].append({name: getattr(meta, name) for name in meta
                                               if not name.startswith("_")})
    raw._experiment_series_records = dict(out)
    return raw._experiment_series_records


def _validated_geometry(raw, index: int) -> dict:
    """Derive FOV/orientation from records linked to the exact OCT series_id; ambiguity fails closed."""
    ref = raw.refs[index]
    n, W = ref.n_bscans, ref.W
    sid = int(str(getattr(raw.vols[index], "volume_id", "")).split("_")[-1])
    records = [r for r in _series_bscan_records(raw).get(sid, [])
               if r.get("numImages") == n and r.get("imgSizeX") == W and
               r.get("posX1") is not None and r.get("posX2") is not None]
    if len(records) != n:
        raise ValueError(f"series {sid} has {len(records)} position records, expected {n}")
    akt = [int(r.get("aktImage", -1)) for r in records]
    if sorted(akt) != list(range(n)):
        raise ValueError(f"series {sid} lacks exactly one record per aktImage")
    recs = sorted(records, key=lambda r: int(r["aktImage"]))
    p1 = np.array([[float(r["posX1"]), float(r.get("posY1", r.get("centrePosY", 0.0)))] for r in recs])
    p2 = np.array([[float(r["posX2"]), float(r.get("posY2", r.get("centrePosY", 0.0)))] for r in recs])
    vec = p2 - p1
    fast_deg = float(np.median(np.linalg.norm(vec, axis=1)))
    med_vec = np.median(vec, axis=0)
    norm = float(np.linalg.norm(med_vec))
    if norm <= 0:
        raise ValueError("zero fast-scan vector")
    fast_unit = med_vec / norm
    slow_unit = np.array([-fast_unit[1], fast_unit[0]])
    slow = ((p1 + p2) * 0.5) @ slow_unit
    slow_deg = float(np.max(slow) - np.min(slow))
    if fast_deg <= 0 or slow_deg <= 0:
        raise ValueError("non-positive angular field")
    # Use the same model-eye factor as reader.core.calibration without pooling fields.
    from reader.core import calibration
    fov = (fast_deg * calibration.MM_PER_DEG, slow_deg * calibration.MM_PER_DEG)
    dslow = np.diff(slow)
    monotonic = bool(np.all(dslow >= -1e-5) or np.all(dslow <= 1e-5))
    if not monotonic:
        raise ValueError("non-monotonic slow-scan trajectory")
    return {
        "source": "e2e_series_id_metadata",
        "series_id": sid,
        "fov_mm": (float(fov[0]), float(fov[1])),
        "fast_deg": fast_deg,
        "slow_deg": slow_deg,
        # Match e2e_source's fundus-orientation convention using the actual slow coordinate.
        "enface_flip": bool(slow[0] > slow[-1]),
        "n_records": len(recs),
    }


def _volume_view(raw, index: int, exact_geometry: bool = False):
    ref = raw.refs[index]
    vol = np.asarray(raw.vols[index].volume, float)
    invalid = fieldmask.invalid_mask(vol)
    if exact_geometry:
        geom = _validated_geometry(raw, index)
        fov = geom["fov_mm"]
        flip = geom["enface_flip"]
    else:
        geom = {
            "source": "reader_current_pooled_geometry",
            "fov_mm": tuple(float(x) for x in ref.fov_mm),
            "fast_deg": None,
            "slow_deg": None,
            "enface_flip": e2e_source.enface_flip_for(raw, index),
            "n_records": None,
        }
        fov, flip = geom["fov_mm"], geom["enface_flip"]
    if min(fov) <= 0:
        raise ValueError(f"invalid FOV {fov} for volume {index}")
    ov = SimpleNamespace(
        volume_id=f"{raw.eid}:{index}", eid=raw.eid, index=index, eye=ref.eye,
        vol=vol, fov_mm=tuple(float(x) for x in fov), field_valid=~invalid,
        enface_flip=bool(flip), n_bscans=vol.shape[0], H=vol.shape[1], W=vol.shape[2],
    )
    return ov, geom


def _select_wide(raw, eye: str) -> int:
    cands = [r for r in raw.refs if r.eye == eye and r.n_bscans > 5 and r.W >= 768]
    if not cands:
        raise ValueError(f"no wide 30-degree volume for {eye}")
    plausible = []
    for ref in cands:
        geom = _validated_geometry(raw, ref.index)
        if 24.5 <= geom["fast_deg"] <= 35.5 and 19.5 <= geom["slow_deg"] <= 30.5:
            plausible.append(ref)
    if not plausible:
        raise ValueError(f"no angularly plausible 30-degree volume for {eye}")
    pick = max(plausible, key=lambda r: (r.n_bscans, r.W))
    return int(pick.index)


def _load_baked_bm(row: dict, expected_shape: tuple[int, int]) -> np.ndarray:
    with np.load(row["bundle_path"], allow_pickle=False) as z:
        if "bm_dl" not in z.files:
            raise KeyError(f"bundle lacks bm_dl: {row['slug']}")
        bm = np.asarray(z["bm_dl"], np.float32)
    if bm.shape != expected_shape:
        raise ValueError(f"baked BM shape {bm.shape} != {expected_shape}: {row['slug']}")
    return bm


def _load_baked_native(row: dict, expected_shape: tuple[int, int]) -> np.ndarray:
    with np.load(row["bundle_path"], allow_pickle=False) as z:
        if "ga_native_dl" not in z.files:
            raise KeyError(f"bundle lacks ga_native_dl: {row['slug']}")
        mask = np.asarray(z["ga_native_dl"], bool)
    if mask.shape != expected_shape:
        raise ValueError(f"baked native mask shape {mask.shape} != {expected_shape}: {row['slug']}")
    return mask


def _wide_bm(row: dict, ov, model_hash: str | None, volume_hash: str,
             force: bool = False) -> tuple[np.ndarray, str]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{row['slug']}_wide{ov.index}_bm_dl.npz")
    if not force and os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as z:
                bm = np.asarray(z["bm"], np.float32)
                cached_hash = str(z["model_hash"])
                cached_volume = str(z["volume_hash"])
                shape = tuple(int(x) for x in z["volume_shape"])
            if (bm.shape == (ov.n_bscans, ov.W) and shape == tuple(ov.vol.shape) and
                    cached_hash == str(model_hash) and cached_volume == volume_hash):
                return bm, "cache"
        except (OSError, ValueError, KeyError):
            pass
    bm = np.asarray(bm_dl.segment_volume(ov.vol), np.float32)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, bm=bm, model_hash=str(model_hash), volume_hash=volume_hash,
                        volume_shape=np.asarray(ov.vol.shape))
    os.replace(tmp, path)
    return bm, "model"


def inverse_e2e_log(display: np.ndarray) -> tuple[np.ndarray, dict]:
    y = np.asarray(display, np.float32)
    x = np.exp(LINEAR_A * y - LINEAR_B).astype(np.float32) - np.float32(LINEAR_OFFSET)
    x = np.clip(x, 0.0, None)
    return x, {
        "display_zero_fraction": float(np.mean(y <= 0.0)),
        "display_one_fraction": float(np.mean(y >= 1.0)),
        "inverse_clipped_fraction": float(np.mean(x <= 0.0)),
    }


def _gain_matched_linear(ov, bm) -> tuple[np.ndarray, dict]:
    linear, meta = inverse_e2e_log(ov.vol)
    disp_rpe = mp.band(ov.vol, bm, -40.0, 10.0, "mean")
    lin_rpe = mp.band(linear, bm, -40.0, 10.0, "mean")
    valid = np.asarray(ov.field_valid, bool)
    dmed = float(np.median(disp_rpe[valid]))
    lmed = float(np.median(lin_rpe[valid]))
    if not math.isfinite(lmed) or lmed <= 0:
        raise ValueError("linear RPE gain anchor is non-positive")
    scale = dmed / lmed
    linear *= np.float32(scale)
    meta.update({"linear_scale": float(scale), "display_rpe_median": dmed,
                 "linear_rpe_median_before_scale": lmed})
    return linear, meta


def center_to_6mm(mask: np.ndarray) -> np.ndarray:
    return reg.resample(np.asarray(mask, np.float32),
                        (proj.ENFACE_MMPP, proj.ENFACE_MMPP), out=512,
                        interp=cv2.INTER_NEAREST) > 0.5


def _mask_stats(mask: np.ndarray) -> dict:
    m = np.asarray(mask, bool)
    lbl = measure.label(m)
    props = measure.regionprops(lbl)
    largest = max((int(r.area) for r in props), default=0)
    return {
        "pixels": int(m.sum()),
        "area_mm2": float(m.sum()) * oac_ga.MMPP2,
        "components": len(props),
        "largest_component_fraction": float(largest / m.sum()) if m.any() else 0.0,
    }


def _percentiles(values: np.ndarray, prefix: str) -> dict:
    v = np.asarray(values, float)
    if v.size == 0:
        return {f"{prefix}_p{p:02d}": None for p in (0, 1, 5, 25, 50, 75, 95)}
    vals = np.percentile(v, [0, 1, 5, 25, 50, 75, 95])
    return {f"{prefix}_p{p:02d}": float(x) for p, x in zip((0, 1, 5, 25, 50, 75, 95), vals)}


def _component_rows(base_row: dict, P: dict, stages: dict, protocol: str) -> list[dict]:
    sized = np.asarray(stages["sized"], bool)
    final = np.asarray(stages["final"], bool)
    lbl = measure.label(sized)
    ratio = P["loss6"] / np.maximum(P["base"], 1e-6)
    h = P["hyper6"]
    core_dist = distance_transform_edt(P["core"])
    H, W = sized.shape
    out = []
    for rp in measure.regionprops(lbl):
        comp = lbl == rp.label
        measured_comp = center_to_6mm(comp) if protocol == "wide30_scancenter6" else comp
        full_px = int(comp.sum())
        measured_px = int(measured_comp.sum())
        minr, minc, maxr, maxc = rp.bbox
        row = {
            **base_row,
            "component_id": f"{base_row['slug']}:{base_row['config_id']}:{rp.label:03d}",
            "component_label": int(rp.label),
            "kept_by_depth": bool(np.any(final & comp)),
            "full_pixels": full_px,
            "full_area_mm2": full_px * oac_ga.MMPP2,
            "measured_pixels": measured_px,
            "measured_area_mm2": measured_px * oac_ga.MMPP2,
            "in_measurement": bool(measured_px > 0),
            "measurement_fraction": float(measured_px / full_px) if full_px else 0.0,
            "touches_measurement_edge": bool(measured_px > 0 and
                                              (measured_comp[0].any() or measured_comp[-1].any() or
                                               measured_comp[:, 0].any() or measured_comp[:, -1].any())),
            "major_axis_mm": float(rp.axis_major_length) * proj.ENFACE_MMPP,
            "minor_axis_mm": float(rp.axis_minor_length) * proj.ENFACE_MMPP,
            "eccentricity": float(rp.eccentricity),
            "centroid_x_mm": (float(rp.centroid[1]) - (W - 1) / 2.0) * proj.ENFACE_MMPP,
            "centroid_y_mm": (float(rp.centroid[0]) - (H - 1) / 2.0) * proj.ENFACE_MMPP,
            "bbox_min_row": int(minr), "bbox_min_col": int(minc),
            "bbox_max_row": int(maxr), "bbox_max_col": int(maxc),
            "scan_edge_distance_mm": float(min(minr, minc, H - maxr, W - maxc)) * proj.ENFACE_MMPP,
            "touches_core_edge": bool(float(np.min(core_dist[comp])) <= 1.0),
            "loss_base_min": float(np.min(ratio[comp])),
        }
        row.update(_percentiles(ratio[comp], "loss_base"))
        row.update(_percentiles(h[comp], "hyper"))
        out.append(row)
    if int(sum(r["full_pixels"] for r in out)) != int(sized.sum()):
        raise AssertionError("component accounting does not reproduce sized mask")
    return out


def run_cell(row: dict, ov, bm: np.ndarray, protocol: str, intensity: str,
             config_hash: str, geom: dict, radiometry_vol: np.ndarray | None,
             radiometry_meta: dict, bm_source: str, volume_hash: str,
             core_override: np.ndarray | None = None, precomputed_P: dict | None = None,
             ) -> tuple[dict, list[dict], list[dict]]:
    """Run OCT-only inference. PLEX/reference fields in ``row`` are not read here."""
    if precomputed_P is not None:
        if radiometry_vol is not None or core_override is not None:
            raise ValueError("precomputed_P is only valid for the display/support arm")
        P = precomputed_P
    else:
        prep_kw = dict(PREP_CONFIG)
        # In inverse radiometry, the display-derived core is injected so support is byte-identical.
        prep_kw["oac_vol"] = radiometry_vol
        prep_kw["hyper_vol"] = radiometry_vol
        prep_kw["core_override"] = core_override
        P = oac_ga.prep(ov, bm, **prep_kw)
    stages = oac_ga.footprint_stages(P, **FOOT_CONFIG)
    direct_mask, direct_area = oac_ga.footprint(P, **FOOT_CONFIG)
    if not np.array_equal(stages["final"], direct_mask):
        raise AssertionError("canonical stage final != footprint")
    if abs(direct_area - float(direct_mask.sum()) * oac_ga.MMPP2) > 1e-12:
        raise AssertionError("footprint area is not pixel count x MMPP2")

    measured = center_to_6mm(direct_mask) if protocol == "wide30_scancenter6" else direct_mask
    area = float(measured.sum()) * oac_ga.MMPP2
    config_id = f"{protocol}__{intensity}__{config_hash}"
    mstat = _mask_stats(measured)
    parity_delta = (area - row["baked_area_mm2"]
                    if protocol == "current97_native" and intensity == "display" else None)
    native_parity = None
    if protocol == "current97_native" and intensity == "display":
        live_native = ga_native.enface_to_native(direct_mask, ov.fov_mm, ov.n_bscans, ov.W, ov.enface_flip)
        baked_native = _load_baked_native(row, (ov.n_bscans, ov.W))
        native_parity = bool(np.array_equal(live_native, baked_native))
        if abs(parity_delta) > PARITY_TOL_MM2 or not native_parity:
            raise AssertionError(f"baked parity failed: area delta={parity_delta}, native={native_parity}")
    base = {
        "schema": SCHEMA,
        "config_hash": config_hash,
        "config_id": config_id,
        "protocol": protocol,
        "intensity": intensity,
        "slug": row["slug"],
        "subject": row["subject"],
        "patient": row["patient"],
        "visit": row["visit"],
        "eye": row["eye"],
    }
    result = {
        **base,
        "ours_area_mm2": area,
        "ours_mask_pixels": int(measured.sum()),
        "ours_component_count": mstat["components"],
        "ours_presence_component": bool(measured.any()),
        "ours_presence_area_0p15": bool(area >= 0.15),
        "ours_presence_legacy_0p25": bool(area >= 0.25),
        "full_field_area_mm2": float(direct_mask.sum()) * oac_ga.MMPP2,
        "full_field_pixels": int(direct_mask.sum()),
        "baked_area_mm2": row["baked_area_mm2"],
        "parity_delta_mm2": parity_delta,
        "parity_native_mask_exact": native_parity,
        "parity_ok": (True if parity_delta is not None else None),
        "volume_index": ov.index,
        "volume_shape": "x".join(str(x) for x in ov.vol.shape),
        "fov_width_mm": float(ov.fov_mm[0]),
        "fov_height_mm": float(ov.fov_mm[1]),
        "enface_flip": bool(ov.enface_flip),
        "geometry_source": geom["source"],
        "series_id": geom.get("series_id"),
        "fast_deg": geom.get("fast_deg"),
        "slow_deg": geom.get("slow_deg"),
        "slow_spacing_mm": (float(ov.fov_mm[1]) / max(ov.n_bscans - 1, 1)),
        "scan_density_class": ("wide_121_primary" if protocol == "wide30_scancenter6" and ov.n_bscans >= 100
                               else "wide_61_sensitivity" if protocol == "wide30_scancenter6"
                               else "current_native"),
        "field_invalid_fraction": float(1.0 - np.mean(ov.field_valid)),
        "bm_source": bm_source,
        "bm_sha256": sha256_array(bm),
        "volume_sha256": volume_hash,
        "core_sha256": sha256_array(P["core"]),
        "e2e_size_bytes": row["e2e_size_bytes"],
        "e2e_mtime_ns": row["e2e_mtime_ns"],
        "bundle_sha256": row["bundle_sha256"],
        "keep_thr": stages.get("keep_thr"),
        "fill_thr": stages.get("fill_thr"),
        **radiometry_meta,
    }

    stage_rows = []
    # Only the main masks form a sequential waterfall. Rejection/hole masks are diagnostic side branches.
    parents = {"hyper_kept": "rpe_candidate", "filled": "hyper_kept",
               "crora_hole_cleaned": "filled", "sized": "crora_hole_cleaned", "final": "sized"}
    for name in STAGE_KEYS:
        full = np.asarray(stages[name], bool)
        obs = center_to_6mm(full) if protocol == "wide30_scancenter6" else full
        ss = _mask_stats(obs)
        full_ss = _mask_stats(full)
        parent_name = parents.get(name)
        parent = None
        if parent_name is not None:
            parent_full = np.asarray(stages[parent_name], bool)
            parent = center_to_6mm(parent_full) if protocol == "wide30_scancenter6" else parent_full
        stage_rows.append({
            **base, "stage": name, **ss,
            "full_pixels": full_ss["pixels"], "full_area_mm2": full_ss["area_mm2"],
            "full_components": full_ss["components"],
            "parent_stage": parent_name,
            "added_pixels_vs_parent": (None if parent is None else int((obs & ~parent).sum())),
            "removed_pixels_vs_parent": (None if parent is None else int((parent & ~obs).sum())),
            "keep_thr": stages.get("keep_thr"), "fill_thr": stages.get("fill_thr"),
        })
    components = _component_rows(base, P, stages, protocol)
    return result, stage_rows, components


def attach_evaluation(result: dict, row: dict, evaluation_hash: str) -> dict:
    """Join targets only after the OCT-only mask, area, stages, and components are finalized."""
    area = float(result["ours_area_mm2"])
    corrected = row["plex_corrected_mm2"]
    return {
        **result,
        "evaluation_hash": evaluation_hash,
        "plex_raw_mm2": row["plex_raw_mm2"],
        "plex_corrected_mm2": corrected,
        "ref_class_raw": row["ref_class_raw"],
        "ref_class_corrected": row["ref_class_corrected"],
        "adjudication_verdict": row["adjudication_verdict"],
        "reference_partial": row["reference_partial"],
        "area_score_included": row["area_score_included"],
        "delta_raw_mm2": area - row["plex_raw_mm2"],
        "abs_error_raw_mm2": abs(area - row["plex_raw_mm2"]),
        "delta_corrected_mm2": None if corrected is None else area - corrected,
        "abs_error_corrected_mm2": None if corrected is None else abs(area - corrected),
        "area_lower_bound": row["area_lower_bound"],
        "has_inframe_gold": row["has_inframe_gold"],
    }


def _as_bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes")


def _f(v) -> float | None:
    return _finite_float(v)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ctr - half, ctr + half


def _area_metrics(rows: list[dict], reference_mode: str, n_boot: int = 2000,
                  exclude_lower_bound: bool = False) -> dict:
    ref_key = "plex_raw_mm2" if reference_mode == "raw" else "plex_corrected_mm2"
    use = [(r, _f(r.get("ours_area_mm2")), _f(r.get(ref_key))) for r in rows
           if not (exclude_lower_bound and _as_bool(r.get("area_lower_bound")))]
    use = [(r, o, p) for r, o, p in use if o is not None and p is not None]
    ours = np.asarray([o for _, o, _ in use], float)
    ref = np.asarray([p for _, _, p in use], float)
    d = ours - ref
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    out = {
        "n_eyes": len(use),
        "n_patients": len({r["patient"] for r, _, _ in use}),
        "bias": float(d.mean()) if len(d) else None,
        "mae": float(np.abs(d).mean()) if len(d) else None,
        "median_ae": float(np.median(np.abs(d))) if len(d) else None,
        "rmse": float(np.sqrt(np.mean(d * d))) if len(d) else None,
        "loa_low": float(d.mean() - 1.96 * sd) if len(d) else None,
        "loa_high": float(d.mean() + 1.96 * sd) if len(d) else None,
        "within_0p5": int(np.sum(np.abs(d) <= 0.5)),
        "within_1p0": int(np.sum(np.abs(d) <= 1.0)),
    }
    if len(use) and n_boot > 0:
        groups = defaultdict(list)
        for r, o, p in use:
            groups[r["patient"]].append(o - p)
        patients = sorted(groups)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        boot_bias, boot_mae = [], []
        for _ in range(n_boot):
            sample = rng.choice(patients, size=len(patients), replace=True)
            vals = np.concatenate([np.asarray(groups[p], float) for p in sample])
            boot_bias.append(float(vals.mean()))
            boot_mae.append(float(np.abs(vals).mean()))
        out["bias_ci_low"], out["bias_ci_high"] = [float(x) for x in np.percentile(boot_bias, [2.5, 97.5])]
        out["mae_ci_low"], out["mae_ci_high"] = [float(x) for x in np.percentile(boot_mae, [2.5, 97.5])]
    else:
        out.update({"bias_ci_low": None, "bias_ci_high": None, "mae_ci_low": None, "mae_ci_high": None})
    return out


def _presence_metrics(rows: list[dict], reference_mode: str, policy: str) -> dict:
    class_key = "ref_class_raw" if reference_mode == "raw" else "ref_class_corrected"
    if policy == "component_presence":
        pred = lambda r: _as_bool(r.get("ours_presence_component"))
    elif policy == "operating_point_0p15":
        pred = lambda r: _f(r.get("ours_area_mm2")) >= 0.15
    elif policy == "legacy_0p25":
        pred = lambda r: _f(r.get("ours_area_mm2")) >= 0.25
    else:
        raise ValueError(policy)
    tp = fn = tn = fp = 0
    scored_patients = set()
    for r in rows:
        cls = r.get(class_key)
        if cls not in ("positive", "negative"):
            continue
        scored_patients.add(r["patient"])
        call = bool(pred(r))
        if cls == "positive":
            tp += int(call); fn += int(not call)
        else:
            fp += int(call); tn += int(not call)
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    sl, sh = _wilson(tp, tp + fn)
    pl, ph = _wilson(tn, tn + fp)
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": sens, "specificity": spec,
            "sensitivity_ci_low": sl, "sensitivity_ci_high": sh,
            "specificity_ci_low": pl, "specificity_ci_high": ph,
            "n_presence_scored": tp + fn + tn + fp, "n_patients_scored": len(scored_patients)}


def build_summary(matrix_rows: list[dict], n_boot: int = 2000) -> tuple[list[dict], dict]:
    by_config = defaultdict(list)
    for r in matrix_rows:
        by_config[r["config_id"]].append(r)
    long_rows = []
    nested = {}
    for config_id in sorted(by_config):
        rr = by_config[config_id]
        nested[config_id] = {}
        for mode in ("raw", "corrected"):
            area = _area_metrics(rr, mode, n_boot=n_boot)
            area_no_lb = _area_metrics(rr, mode, n_boot=n_boot, exclude_lower_bound=True)
            nested[config_id][mode] = {"area": area, "area_excluding_lower_bound": area_no_lb,
                                       "presence": {}}
            for subset, metrics in (("all", area), ("exclude_lower_bound", area_no_lb)):
                for metric, value in metrics.items():
                    if metric.startswith(("bias_ci", "mae_ci")):
                        continue
                    long_rows.append({
                        "schema": SCHEMA, "config_id": config_id, "reference_mode": mode,
                        "prediction_policy": "area", "subset": subset, "metric": metric,
                        "value": value, "numerator": "", "denominator": "",
                        "n_eyes": metrics["n_eyes"], "n_patients": metrics["n_patients"],
                        "ci_low": (metrics.get("bias_ci_low") if metric == "bias" else
                                   metrics.get("mae_ci_low") if metric == "mae" else ""),
                        "ci_high": (metrics.get("bias_ci_high") if metric == "bias" else
                                    metrics.get("mae_ci_high") if metric == "mae" else ""),
                        "bootstrap_replicates": n_boot, "bootstrap_seed": BOOTSTRAP_SEED,
                        "bootstrap_unit": "patient_cluster",
                    })
            for policy in ("component_presence", "operating_point_0p15", "legacy_0p25"):
                pm = _presence_metrics(rr, mode, policy)
                nested[config_id][mode]["presence"][policy] = pm
                for metric in ("sensitivity", "specificity"):
                    num = pm["tp"] if metric == "sensitivity" else pm["tn"]
                    den = pm["tp"] + pm["fn"] if metric == "sensitivity" else pm["tn"] + pm["fp"]
                    long_rows.append({
                        "schema": SCHEMA, "config_id": config_id, "reference_mode": mode,
                        "prediction_policy": policy, "subset": "confirmed_presence",
                        "metric": metric, "value": pm[metric], "numerator": num, "denominator": den,
                        "n_eyes": pm["n_presence_scored"],
                        "n_patients": pm["n_patients_scored"],
                        "ci_low": pm[f"{metric}_ci_low"], "ci_high": pm[f"{metric}_ci_high"],
                        "bootstrap_replicates": "", "bootstrap_seed": "",
                        "bootstrap_unit": "Wilson-eye-binomial",
                    })
    return long_rows, nested


def _fmt(v, nd=3):
    x = _f(v)
    return "—" if x is None else f"{x:.{nd}f}"


def _paired_effect(arm_rows: list[dict], anchor_rows: list[dict], n_boot: int = 2000) -> dict:
    """Patient-cluster bootstrap of within-eye area and absolute-error changes on a common cohort."""
    anchor = {r["slug"]: r for r in anchor_rows}
    pairs = []
    for r in arm_rows:
        a = anchor.get(r["slug"])
        if a is None:
            continue
        # The lone 61-line wide eye changes sampling density as well as support: sensitivity analysis only.
        if r.get("scan_density_class") == "wide_61_sensitivity":
            continue
        pairs.append((r, a))
    darea = np.asarray([float(r["ours_area_mm2"]) - float(a["ours_area_mm2"]) for r, a in pairs])
    err_pairs = [(r, a) for r, a in pairs if not _as_bool(r.get("area_lower_bound"))]
    derr = np.asarray([abs(float(r["ours_area_mm2"]) - float(r["plex_corrected_mm2"])) -
                       abs(float(a["ours_area_mm2"]) - float(a["plex_corrected_mm2"]))
                       for r, a in err_pairs])
    out = {"n_area": len(pairs), "n_error": len(err_pairs),
           "mean_area_change": float(darea.mean()) if len(darea) else None,
           "mean_abs_error_change": float(derr.mean()) if len(derr) else None}
    if n_boot > 0 and pairs:
        by_patient_area = defaultdict(list)
        by_patient_err = defaultdict(list)
        for r, a in pairs:
            by_patient_area[r["patient"]].append(float(r["ours_area_mm2"]) - float(a["ours_area_mm2"]))
        for r, a in err_pairs:
            by_patient_err[r["patient"]].append(
                abs(float(r["ours_area_mm2"]) - float(r["plex_corrected_mm2"])) -
                abs(float(a["ours_area_mm2"]) - float(a["plex_corrected_mm2"])))
        patients = sorted(by_patient_area)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        ba, be = [], []
        for _ in range(n_boot):
            sample = rng.choice(patients, len(patients), replace=True)
            va = np.concatenate([np.asarray(by_patient_area[p]) for p in sample])
            ve_parts = [np.asarray(by_patient_err[p]) for p in sample if p in by_patient_err]
            ba.append(float(va.mean()))
            if ve_parts:
                be.append(float(np.concatenate(ve_parts).mean()))
        out["area_ci_low"], out["area_ci_high"] = [float(x) for x in np.percentile(ba, [2.5, 97.5])]
        if be:
            out["error_ci_low"], out["error_ci_high"] = [float(x) for x in np.percentile(be, [2.5, 97.5])]
    return out


def write_report(matrix_rows: list[dict], summary: dict, provenance: dict, n_boot: int = 2000):
    by_config = defaultdict(list)
    for r in matrix_rows:
        by_config[r["config_id"]].append(r)
    expected = 25
    lines = [
        "# GA algorithm experiment: scan support × OCT radiometry",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} · schema `{SCHEMA}` · config hash "
        f"`{provenance['config_hash']}`.",
        "",
        "This is an isolated experiment. Production defaults and baked viewer numbers were not changed. "
        "PLEX is evaluation-only; no PLEX data enter OCT inference.",
        "",
        "## Reference handling",
        "",
        "006 OD and 010 OD are hand-adjudicated **PLEX false positives**. They stay in all 25-eye analyses "
        "as corrected 0-mm² negative controls; they are not algorithm false negatives. 001 OS is a partial "
        "PLEX over-call and remains scored unchanged because real GA is present but its exact true area is unknown.",
        "",
        "Presence is reported three ways: a nonempty post-cRORA component (the anatomical endpoint), a "
        "separate pragmatic 0.15-mm² operating point, and the historical 0.25-mm² area rule. The 250-µm "
        "cRORA diameter criterion is a length rule and is not equivalent to 0.25 mm².",
        "",
        "Raw PLEX comparisons are primary. Adjudication was applied only to selected disagreements and in "
        "one direction, so adjusted metrics are an explicitly optimistic upper-bound sensitivity analysis.",
        "",
        "## Full-cohort descriptive results",
        "",
        "Only 25/25 arms appear here. Partial/stopped arms are never compared as if cohorts matched.",
        "",
        "| Arm | Eyes | Raw bias / MAE | Adjusted bias / MAE | Component sensitivity / specificity | 0.15 sensitivity / specificity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cid in sorted(summary):
        if summary[cid]["raw"]["area"]["n_eyes"] != expected:
            continue
        raw = summary[cid]["raw"]["area"]
        m = summary[cid]["corrected"]
        a = m["area"]
        cp = m["presence"]["component_presence"]
        op = m["presence"]["operating_point_0p15"]
        label = cid.split("__" + provenance["config_hash"])[0]
        lines.append(f"| `{label}` | {a['n_eyes']}/{expected} | {_fmt(raw['bias'])} / {_fmt(raw['mae'])} | "
                     f"{_fmt(a['bias'])} / {_fmt(a['mae'])} | "
                     f"{_fmt(cp['sensitivity'], 2)} / {_fmt(cp['specificity'], 2)} | "
                     f"{_fmt(op['sensitivity'], 2)} / {_fmt(op['specificity'], 2)} |")

    partial = [(cid, summary[cid]["raw"]["area"]["n_eyes"]) for cid in sorted(summary)
               if summary[cid]["raw"]["area"]["n_eyes"] != expected]
    if partial:
        lines += ["", "## Partial or stopped arms", "",
                  "These are guardrail/sentinel results only; no cohort-performance claim is made.", "",
                  "| Arm | Completed eyes |", "|---|---:|"]
        for cid, n in partial:
            label = cid.split("__" + provenance["config_hash"])[0]
            lines.append(f"| `{label}` | {n}/{expected} |")

    parity = next((rr for cid, rr in by_config.items() if cid.startswith("current97_native__display__")), [])
    if parity:
        deltas = np.asarray([abs(float(r["parity_delta_mm2"])) for r in parity], float)
        n_ok = sum(_as_bool(r.get("parity_ok")) for r in parity)
        lines += [
            "", "## Reproduction check", "",
            f"The current 97-line/display arm reproduced {n_ok}/{len(parity)} baked DL-BM outputs with an "
            f"exact native-mask match and four-decimal JSON area tolerance ({PARITY_TOL_MM2:.6f} mm²). "
            f"Maximum absolute area delta: {deltas.max():.6f} mm².",
        ]

    lines += ["", "## Guardrail eyes", "",
              "| Arm | 006 OD (negative) | 010 OD (negative) | 014 OD (miss target) | 005 OD (gold anchor) | 008 OS (large GA) |",
              "|---|---:|---:|---:|---:|---:|"]
    slugs = ["NHAMD-003-006-V3_OD", "NHAMD-003-010-V1_OD", "NHAMD-003-014-V1_OD",
             "NHAMD-003-005-V3_OD", "NHAMD-003-008-V1_OS"]
    for cid in sorted(by_config):
        lookup = {r["slug"]: r for r in by_config[cid]}
        label = cid.split("__" + provenance["config_hash"])[0]
        lines.append("| `" + label + "` | " + " | ".join(_fmt(lookup.get(s, {}).get("ours_area_mm2")) for s in slugs) + " |")

    # Paired effects against the exact parity arm, never independent-arm comparisons.
    if parity:
        lines += ["", "## Paired change relative to the shipped arm", "",
                  f"Patient-cluster bootstrap: {n_boot} replicates, seed {BOOTSTRAP_SEED}. The wide primary "
                  "effect excludes the single 61-line eye (001 OS); absolute-error change also excludes "
                  "edge-clipped/lower-bound 008 OD. Negative error change is improvement.", "",
                  "| Arm | Paired eyes | Mean area change (95% CI) | Mean absolute-error change (95% CI) |",
                  "|---|---:|---:|---:|"]
        for cid in sorted(by_config):
            if by_config[cid] is parity:
                continue
            effect = _paired_effect(by_config[cid], parity, n_boot=n_boot)
            if not effect["n_area"]:
                continue
            label = cid.split("__" + provenance["config_hash"])[0]
            area_ci = (f"{_fmt(effect.get('mean_area_change'))} "
                       f"({_fmt(effect.get('area_ci_low'))}, {_fmt(effect.get('area_ci_high'))})")
            err_ci = (f"{_fmt(effect.get('mean_abs_error_change'))} "
                      f"({_fmt(effect.get('error_ci_low'))}, {_fmt(effect.get('error_ci_high'))})")
            lines.append(f"| `{label}` | {effect['n_area']} | {area_ci} | {err_ci} |")

    lines += [
        "", "## Interpretation rules", "",
        "- The inverse-linear arms are locked **radiometry transport tests**: the whole linear volume is "
        "gain-matched by the median BM-relative RPE band, while the current `+0.02` hyper denominator and "
        "`hyper_abs=0.10` are held fixed. They are not yet optimized linear-space detectors.",
        "- Wide arms fit the radial baseline and run morphology on the full 30° scan, then count only a "
        "scan-centred 6×6 mm window. It is not called fovea-centred without an OCT foveal localisation step.",
        "- 24/25 wide eyes use 121 B-scans. 001 OS has only 61 and is a sampling-density sensitivity case, "
        "excluded from the primary paired support effect.",
        "- 008 OD is edge-clipped and its PLEX area is a lower bound. Metrics are emitted with and without it; "
        "paired absolute-error effects exclude it.",
        "- Selection of any new default requires patient-split validation and must preserve 005 OD, the large "
        "008 eyes, 010 OD/016 OD/002 OS/012 OD/OS controls, and must not increase 006 OD.",
        "- Component tables are extracted immediately before the current complete-loss veto. They are suitable "
        "for a later patient-split learned component combiner; target-derived fields are not inference inputs here.",
        "", "## Artifacts", "",
        f"- `{os.path.relpath(MANIFEST_CSV, _REPO)}` — 25-eye reference/QC manifest",
        f"- `{os.path.relpath(MATRIX_CSV, _REPO)}` — per-eye locked matrix",
        f"- `{os.path.relpath(STAGES_CSV, _REPO)}` — stage waterfall",
        f"- `{os.path.relpath(COMPONENTS_CSV, _REPO)}` — pre-depth components",
        f"- `{os.path.relpath(SUMMARY_CSV, _REPO)}` — long-form metrics with cluster/Wilson intervals",
    ]
    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    tmp = REPORT_MD + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, REPORT_MD)


def self_test():
    import inspect

    # All detector controls are locked explicitly.
    prep_params = set(inspect.signature(oac_ga.prep).parameters) - {
        "ov", "bm", "oac_vol", "hyper_vol", "core_override"}
    if prep_params != set(PREP_CONFIG):
        raise AssertionError(f"prep config drift: missing={prep_params - set(PREP_CONFIG)}, "
                             f"extra={set(PREP_CONFIG) - prep_params}")
    foot_params = set(inspect.signature(oac_ga.footprint).parameters) - {"p"}
    if foot_params != set(FOOT_CONFIG):
        raise AssertionError(f"footprint config drift: {foot_params ^ set(FOOT_CONFIG)}")

    # The central resampler must be an exact identity at the working 512 grid.
    test_mask = np.zeros((512, 512), bool)
    test_mask[17:83, 311:419] = True
    if not np.array_equal(center_to_6mm(test_mask), test_mask):
        raise AssertionError("center_to_6mm is not identity on 512x512")

    # Inverse-log round trip away from the censored floor.
    y = np.linspace(0, 1, 10001, dtype=np.float32)
    x, _ = inverse_e2e_log(y)
    ok = x > 0
    y2 = (np.log(x[ok] + LINEAR_OFFSET) + LINEAR_B) / LINEAR_A
    if float(np.max(np.abs(y2 - y[ok]))) > 2e-6:
        raise AssertionError("E2E inverse-log round trip failed")

    # Canonical stage invariants over meaningful control variants.
    H = W = 96
    yy, xx = np.mgrid[:H, :W]
    lesion = (xx - 48) ** 2 + (yy - 48) ** 2 < 25 ** 2
    P = {
        "loss6": np.where(lesion, 0.15, 0.9).astype(np.float32),
        "base": np.ones((H, W), np.float32),
        "core": np.ones((H, W), bool),
        "hyper6": np.where(lesion, 1.0, 0.01).astype(np.float32),
    }
    for hyper_fill in (False, True):
        for fill_all in (False, True):
            for min_depth in (None, 0.27):
                kw = dict(FOOT_CONFIG, hyper_fill=hyper_fill, fill_all_holes=fill_all,
                          min_depth=min_depth)
                s = oac_ga.footprint_stages(P, **kw)
                m, a = oac_ga.footprint(P, **kw)
                if not np.array_equal(s["final"], m):
                    raise AssertionError("stage/footprint parity failed")
                if not np.all(s["hyper_kept"] <= s["rpe_candidate"]):
                    raise AssertionError("hyper_kept is not a candidate subset")
                if np.any(s["holes_filled"] & s["hyper_kept"]):
                    raise AssertionError("filled holes overlap kept seed")
                if not np.all(s["sized"] <= s["crora_hole_cleaned"]):
                    raise AssertionError("sized is not a hole-cleaned subset")
                if not np.all(s["final"] <= s["sized"]):
                    raise AssertionError("final is not a sized subset")
                if abs(a - float(m.sum()) * oac_ga.MMPP2) > 1e-12:
                    raise AssertionError("area conversion drift")

    # Reference fixture: the two false-positive PLEX calls become negatives, not deleted eyes.
    manifest = load_manifest()
    lookup = {r["slug"]: r for r in manifest}
    for slug in ("NHAMD-003-006-V3_OD", "NHAMD-003-010-V1_OD"):
        if lookup[slug]["plex_corrected_mm2"] != 0.0 or lookup[slug]["ref_class_corrected"] != "negative":
            raise AssertionError(f"adjudication fixture failed: {slug}")
    fixture_rows = [{**r, "ours_area_mm2": r["baked_area_mm2"]} for r in manifest]
    raw = _area_metrics(fixture_rows, "raw", n_boot=0)
    corrected = _area_metrics(fixture_rows, "corrected", n_boot=0)
    if abs(raw["bias"] - (-0.580768)) > 1e-6 or abs(raw["mae"] - 0.615984) > 1e-6:
        raise AssertionError(f"raw baked fixture drift: {raw}")
    if abs(corrected["bias"] - (-0.485324)) > 1e-6 or abs(corrected["mae"] - 0.530980) > 1e-6:
        raise AssertionError(f"corrected baked fixture drift: {corrected}")
    print("self-test: PASS (config, radiometry, crop, stages, adjudication, baked metrics)", flush=True)


def _load_result_rows(path: str, config_hash: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = _read_csv(path)
    return [r for r in rows if r.get("config_hash") == config_hash]


def _sanitize_checkpoints(matrix_rows, stage_rows, component_rows):
    """A cell is complete only when all three checkpoint tables agree exactly."""
    stages_by = defaultdict(list)
    comps_by = defaultdict(list)
    for r in stage_rows:
        stages_by[(r["config_id"], r["slug"])].append(r)
    for r in component_rows:
        comps_by[(r["config_id"], r["slug"])].append(r)
    valid = set()
    for m in matrix_rows:
        key = (m["config_id"], m["slug"])
        ss = stages_by.get(key, [])
        if len(ss) != len(STAGE_KEYS) or {r["stage"] for r in ss} != set(STAGE_KEYS):
            continue
        sized = next(r for r in ss if r["stage"] == "sized")
        cc = comps_by.get(key, [])
        expected_components = int(float(sized.get("full_components", sized["components"])))
        expected_pixels = int(float(sized.get("full_pixels", sized["pixels"])))
        full_pixels = sum(int(float(r["full_pixels"])) for r in cc)
        if len(cc) != expected_components or full_pixels != expected_pixels:
            continue
        valid.add(key)
    clean_m = [r for r in matrix_rows if (r["config_id"], r["slug"]) in valid]
    clean_s = [r for r in stage_rows if (r["config_id"], r["slug"]) in valid]
    clean_c = [r for r in component_rows if (r["config_id"], r["slug"]) in valid]
    return clean_m, clean_s, clean_c, valid


@contextmanager
def _run_lock(config_hash: str, break_existing: bool = False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"run_{config_hash}.lock")
    if break_existing and os.path.exists(path):
        os.unlink(path)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"experiment already running (or stale lock): {path}; use --break-lock after checking") from exc
    try:
        os.write(fd, canonical_json({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}).encode())
        os.close(fd)
        yield
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _checkpoint(matrix_rows, stage_rows, component_rows):
    _write_csv_atomic(MATRIX_CSV, sorted(matrix_rows, key=lambda r: (r["config_id"], r["slug"])))
    _write_csv_atomic(STAGES_CSV, sorted(stage_rows, key=lambda r: (r["config_id"], r["slug"], r["stage"])))
    _write_csv_atomic(COMPONENTS_CSV, sorted(component_rows,
                                             key=lambda r: (r["config_id"], r["slug"], int(r["component_label"]))))


def _selected(row: dict, patterns: list[str]) -> bool:
    return not patterns or any(p.lower() in row["slug"].lower() for p in patterns)


def execute(args, provenance=None) -> tuple[list[dict], dict]:
    manifest = load_manifest()
    write_manifest(manifest)
    provenance = provenance or experiment_provenance()
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, f"config_{provenance['config_hash']}.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)

    if args.force:
        matrix_rows, stage_rows, component_rows = [], [], []
    else:
        matrix_rows = _load_result_rows(MATRIX_CSV, provenance["config_hash"])
        stage_rows = _load_result_rows(STAGES_CSV, provenance["config_hash"])
        component_rows = _load_result_rows(COMPONENTS_CSV, provenance["config_hash"])
    # Rejoin current evaluation targets without touching inference artifacts if adjudication changes.
    manifest_by_slug = {r["slug"]: r for r in manifest}
    matrix_rows = [attach_evaluation(r, manifest_by_slug[r["slug"]], provenance["evaluation_hash"])
                   for r in matrix_rows if r.get("slug") in manifest_by_slug]
    matrix_rows, stage_rows, component_rows, completed = _sanitize_checkpoints(
        matrix_rows, stage_rows, component_rows)

    protocols = (["current97_native", "wide30_scancenter6"] if args.protocol == "all" else
                 ["current97_native"] if args.protocol == "current97" else ["wide30_scancenter6"])
    intensities = (["display", "inverse_linear"] if args.intensity == "all" else
                   ["display"] if args.intensity == "display" else ["inverse_linear"])
    chosen = [r for r in manifest if _selected(r, args.only)]
    if not chosen:
        raise ValueError(f"--only patterns matched no eyes: {args.only}")
    groups = defaultdict(list)
    for r in chosen:
        groups[r["e2e_path"]].append(r)

    if not bm_dl.available():
        raise RuntimeError("DL BM model/backend is unavailable")
    model_hash = provenance["inference"]["inference_sources"]["bm_model_sha256"]
    errors = []
    total_cells = len(chosen) * len(protocols) * len(intensities)
    print(f"experiment {provenance['config_hash']}: {len(chosen)} eyes, {total_cells} requested cells", flush=True)

    for e2e_path, eyes in sorted(groups.items()):
        print(f"OPEN {os.path.relpath(e2e_path, _REPO)}", flush=True)
        raw = None
        try:
            raw = e2e_source.open_e2e(e2e_path)
            for eye_row in sorted(eyes, key=lambda r: r["eye"]):
                for protocol in protocols:
                    pending = []
                    for intensity in intensities:
                        cid = f"{protocol}__{intensity}__{provenance['config_hash']}"
                        if args.force or (cid, eye_row["slug"]) not in completed:
                            pending.append(intensity)
                    if not pending:
                        print(f"  SKIP {eye_row['slug']} {protocol} (checkpoint)", flush=True)
                        continue
                    try:
                        if protocol == "current97_native":
                            index = e2e_source.default_volume_index(raw, eye_row["eye"])
                            ov, geom = _volume_view(raw, index, exact_geometry=False)
                            bm = _load_baked_bm(eye_row, (ov.n_bscans, ov.W))
                            bm_source = "baked_dl"
                        else:
                            index = _select_wide(raw, eye_row["eye"])
                            ov, geom = _volume_view(raw, index, exact_geometry=True)
                            need = 6.0 + 2.0 * PREP_CONFIG["margin_mm"]
                            if min(ov.fov_mm) < need:
                                raise ValueError(f"wide field {ov.fov_mm} does not cover 6 mm + rim ({need})")
                            # Full voxel fingerprint prevents a same-shaped replacement E2E reusing stale BM.
                            volume_hash = sha256_array(ov.vol)
                            bm, src = _wide_bm(eye_row, ov, model_hash, volume_hash, force=args.force_bm)
                            bm_source = f"wide_dl_{src}"
                        if protocol == "current97_native":
                            volume_hash = sha256_array(ov.vol)
                        bm_hash = sha256_array(bm)
                        linear_vol = None
                        linear_meta = None
                        support_P = None
                        support_core = None
                        support_core_hash = None
                        if "inverse_linear" in pending:
                            support_kw = dict(PREP_CONFIG)
                            support_kw.update(oac_vol=None, hyper_vol=None, core_override=None)
                            support_P = oac_ga.prep(ov, bm, **support_kw)
                            support_core = np.asarray(support_P["core"], bool).copy()
                            support_core_hash = sha256_array(support_core)
                        for intensity in pending:
                            if intensity == "display":
                                radiometry = None
                                rmeta = {
                                    "display_zero_fraction": float(np.mean(ov.vol <= 0.0)),
                                    "display_one_fraction": float(np.mean(ov.vol >= 1.0)),
                                    "inverse_clipped_fraction": None,
                                    "linear_scale": None,
                                    "display_rpe_median": None,
                                    "linear_rpe_median_before_scale": None,
                                }
                            else:
                                if linear_vol is None:
                                    linear_vol, linear_meta = _gain_matched_linear(ov, bm)
                                radiometry, rmeta = linear_vol, dict(linear_meta)
                            print(f"  RUN {eye_row['slug']} {protocol} {intensity}", flush=True)
                            result, sr, cr = run_cell(
                                eye_row, ov, bm, protocol, intensity, provenance["config_hash"], geom,
                                radiometry, rmeta, bm_source, volume_hash,
                                core_override=(support_core if intensity == "inverse_linear" else None),
                                precomputed_P=(support_P if intensity == "display" and support_P is not None else None),
                            )
                            result = attach_evaluation(result, eye_row, provenance["evaluation_hash"])
                            if result["bm_sha256"] != bm_hash:
                                raise AssertionError("BM changed between radiometry cells")
                            if support_core_hash is not None and result["core_sha256"] != support_core_hash:
                                raise AssertionError("display/inverse core support changed")
                            cell_key = (result["config_id"], eye_row["slug"])
                            matrix_rows = [r for r in matrix_rows if (r["config_id"], r["slug"]) != cell_key]
                            stage_rows = [r for r in stage_rows if (r["config_id"], r["slug"]) != cell_key]
                            component_rows = [r for r in component_rows if (r["config_id"], r["slug"]) != cell_key]
                            matrix_rows.append(result); stage_rows.extend(sr); component_rows.extend(cr)
                            completed.add(cell_key)
                            _checkpoint(matrix_rows, stage_rows, component_rows)
                            print(f"    area={result['ours_area_mm2']:.4f} mm2 components={result['ours_component_count']}", flush=True)
                            del radiometry
                            gc.collect()
                        del bm, ov
                        if support_P is not None:
                            del support_P
                        if linear_vol is not None:
                            del linear_vol
                        gc.collect()
                    except Exception as exc:
                        err = {"slug": eye_row["slug"], "protocol": protocol,
                               "error_type": type(exc).__name__, "error": str(exc)}
                        errors.append(err)
                        print(f"  ERROR {err}", flush=True)
                        if args.fail_fast:
                            raise
        finally:
            if raw is not None:
                try:
                    raw.reader.close()
                except Exception:
                    pass
            del raw
            gc.collect()

    if errors:
        _write_csv_atomic(os.path.join(RESULTS_DIR, "ga_experiment_errors.csv"), errors)
    summary_rows, summary = build_summary(matrix_rows, n_boot=args.bootstrap)
    _write_csv_atomic(SUMMARY_CSV, summary_rows)
    write_report(matrix_rows, summary, provenance, n_boot=args.bootstrap)
    return matrix_rows, provenance


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", action="append", default=[], help="run slugs containing this text (repeatable)")
    p.add_argument("--protocol", choices=("all", "current97", "wide30"), default="all")
    p.add_argument("--intensity", choices=("all", "display", "linear"), default="all")
    p.add_argument("--force", action="store_true", help="discard current-hash result checkpoints")
    p.add_argument("--force-bm", action="store_true", help="recompute cached wide DL BM surfaces")
    p.add_argument("--break-lock", action="store_true", help="remove a verified-stale run lock")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--bootstrap", type=int, default=2000, help="patient-cluster bootstrap replicates")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--report-only", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    provenance = experiment_provenance()
    if args.report_only:
        rows = _load_result_rows(MATRIX_CSV, provenance["config_hash"])
        if not rows:
            raise RuntimeError("no current-hash matrix rows to report")
        summary_rows, summary = build_summary(rows, n_boot=args.bootstrap)
        _write_csv_atomic(SUMMARY_CSV, summary_rows)
        write_report(rows, summary, provenance, n_boot=args.bootstrap)
        print(REPORT_MD)
        return 0
    with _run_lock(provenance["config_hash"], break_existing=args.break_lock):
        rows, provenance = execute(args, provenance=provenance)
    print(f"DONE: {len(rows)} cells -> {MATRIX_CSV}", flush=True)
    print(f"REPORT: {REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
