"""
Full export of a Heidelberg Spectralis HEYEX .E2E file.

Roles (per the approved plan):
  - oct-converter : authoritative image extractor (all B-scans + fundus) + patient/acquisition metadata
  - eyepy         : authoritative structured metadata + retinal-layer segmentations (GA-relevant)
  - cross-check the B-scan counts/dimensions reported by both

Usage:
    python read_e2e.py <input.E2E> <output_dir>
"""
import sys
import os
import io
import json
import traceback
from contextlib import redirect_stdout

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# ----------------------------------------------------------------------------- helpers
def log(summary, msg):
    print(msg)
    summary.append(msg)


def to_uint8(arr):
    """Contrast-normalize an arbitrary numeric 2D array to 8-bit for PNG."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3 and a.shape[-1] in (3, 4):  # already RGB(A)
        a = a.astype(np.float64)
        lo, hi = np.nanmin(a), np.nanmax(a)
    else:
        lo = np.nanpercentile(a, 1)
        hi = np.nanpercentile(a, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    a = np.clip((a - lo) / (hi - lo), 0, 1) * 255.0
    return a.astype(np.uint8)


def save_png(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(to_uint8(arr)).save(path)


def parse_sid(id_str):
    """oct-converter ids look like 'patient_study_SERIES'; return the series id."""
    try:
        return int(str(id_str).split("_")[-1])
    except Exception:
        return None


def eye_tag(laterality):
    """Normalize a laterality value to OD (right) / OS (left) / U (unknown)."""
    s = str(laterality).strip().upper()
    if s in ("R", "OD", "RIGHT", "82"):
        return "OD"
    if s in ("L", "OS", "LEFT", "76"):
        return "OS"
    return "U"


def _safe(fn):
    """Call a zero-arg function, returning None on any error."""
    try:
        return fn()
    except Exception:
        return None


def jsonable(obj):
    """Best-effort conversion of arbitrary metadata into JSON-serializable form."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    # datetime, custom objects, etc.
    for attr in ("isoformat",):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        try:
            return {str(k): jsonable(v) for k, v in vars(obj).items()
                    if not k.startswith("_")}
        except Exception:
            pass
    return repr(obj)


# ----------------------------------------------------------------------------- main
def main():
    in_path = sys.argv[1]
    out_dir = sys.argv[2]

    fundus_dir = os.path.join(out_dir, "fundus")     # subfolders per modality
    bscan_dir = os.path.join(out_dir, "bscans")       # subfolders per volume
    annot_dir = os.path.join(out_dir, "bscans_annotated")
    for d in (out_dir, fundus_dir, bscan_dir):
        os.makedirs(d, exist_ok=True)

    summary = []
    metadata = {"input_file": in_path, "file_size_bytes": os.path.getsize(in_path)}

    log(summary, "=" * 70)
    log(summary, f"E2E export :: {in_path}")
    log(summary, f"size       :: {metadata['file_size_bytes'] / 1e6:.1f} MB")
    log(summary, "=" * 70)

    # ---------------------------------------------------------------- eyepy
    # Runs FIRST: builds series_id -> modality map used to organise the images,
    # plus structured metadata + (optional) layer segmentations.
    # NOTE: eyepy's high-level get_volume()/volumes crashes with
    # KeyError(TypesEnum.layer_annotation) on files without layer segmentation,
    # so we iterate series directly and treat bscans/layers as optional. The one
    # thing eyepy reports reliably for EVERY series is the en-face modality.
    eyepy_bscan_total = 0
    eyepy_dims = []
    layers_found = False
    modality_map = {}  # series_id (int) -> modality string (NIR / BAF / ...)
    try:
        from eyepy.io.he import HeE2eReader
        log(summary, "\n[eyepy] reading series via HeE2eReader ...")
        with HeE2eReader(in_path) as r:
            series = list(r.series)
            log(summary, f"[eyepy] {len(series)} series found")

            ev_meta = []
            for si, s in enumerate(series):
                sid = getattr(s, "id", None)
                try:
                    mod = str(s.enface_modality())
                except Exception:
                    mod = "UNKNOWN"
                if sid is not None:
                    modality_map[int(sid)] = mod

                # B-scans (present only for OCT series)
                try:
                    data = np.asarray(s.get_bscans())
                    if data.ndim == 2:
                        data = data[None, ...]
                    n = int(data.shape[0])
                    bshape = data.shape[1:]
                    eyepy_bscan_total += n
                    eyepy_dims.append(tuple(int(x) for x in bshape))
                    has_bscans = True
                except Exception:
                    data, n, bshape, has_bscans = None, 0, None, False

                # layers are optional — get_layers() KeyErrors when absent
                layers = {}
                try:
                    gl = s.get_layers()
                    if hasattr(gl, "items"):
                        layers = dict(gl)
                except Exception:
                    layers = {}
                lkeys = list(layers.keys())
                if lkeys:
                    layers_found = True

                m = {
                    "series_index": si,
                    "series_id": jsonable(sid),
                    "modality": mod,
                    "num_bscans": n,
                    "bscan_shape": [int(x) for x in bshape] if bshape else None,
                    "layers": lkeys,
                    "meta": jsonable(_safe(s.get_meta)),
                }
                log(summary, f"   series {si} (id={sid}): modality={mod}, "
                             f"{n} B-scans, shape={bshape}, layers={lkeys}")

                if lkeys and has_bscans:
                    layer_heights = {}
                    for name, ann in layers.items():
                        try:
                            layer_heights[str(name)] = jsonable(
                                np.asarray(ann.data if hasattr(ann, "data") else ann))
                        except Exception as le:
                            layer_heights[str(name)] = f"error: {le!r}"
                    m["layer_heights"] = layer_heights

                    os.makedirs(annot_dir, exist_ok=True)
                    for bi in range(n):
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.imshow(to_uint8(data[bi]), cmap="gray")
                        for name, ann in layers.items():
                            try:
                                raw = ann.data if hasattr(ann, "data") else ann
                                heights = np.asarray(raw)[bi]
                                ax.plot(np.arange(heights.shape[0]), heights,
                                        lw=0.8, label=str(name))
                            except Exception:
                                pass
                        ax.set_title(f"series {si} B-scan {bi} — layers")
                        ax.legend(fontsize=6, loc="upper right")
                        ax.axis("off")
                        fig.savefig(os.path.join(
                            annot_dir, f"series{si}_bscan_{bi:03d}.png"),
                            dpi=120, bbox_inches="tight")
                        plt.close(fig)
                ev_meta.append(m)
            metadata["eyepy_series"] = ev_meta

    except Exception as e:
        log(summary, f"\n[eyepy] FAILED (image export still works via "
                     f"oct-converter, but modality labels/layers may be "
                     f"missing): {e!r}")
        metadata["eyepy_error"] = traceback.format_exc()
    metadata["modality_map"] = {str(k): v for k, v in modality_map.items()}

    # ---------------------------------------------------------------- oct-converter
    # Authoritative image extractor. Uses the eyepy modality_map + each image's
    # laterality to sort output into  fundus/<MODALITY>/<EYE>_<sid>.png  and
    # bscans/<EYE>_<n>line_<sid>/bscan_###.png.
    oct_bscan_total = 0
    oct_dims = []
    try:
        from oct_converter.readers import E2E
        e2e = E2E(in_path)

        try:
            metadata["oct_converter_metadata"] = jsonable(e2e.read_all_metadata())
        except Exception as e:
            metadata["oct_converter_metadata_error"] = repr(e)

        # ---- OCT volumes (B-scans) -> bscans/<EYE>_<n>line_<sid>/
        log(summary, "\n[oct-converter] reading OCT volume(s) ...")
        volumes = e2e.read_oct_volume()
        log(summary, f"[oct-converter] {len(volumes)} OCT volume(s) found")
        vol_meta = []
        for vi, v in enumerate(volumes):
            bscans = v.volume
            n = len(bscans)
            shape = np.asarray(bscans[0]).shape if n else None
            oct_bscan_total += n
            if shape is not None:
                oct_dims.append(tuple(int(x) for x in shape))
            sid = parse_sid(getattr(v, "volume_id", None))
            eye = eye_tag(getattr(v, "laterality", None))
            mod = modality_map.get(sid, "OCT")
            folder_name = f"{eye}_{n}line_{sid}"
            folder = os.path.join(bscan_dir, folder_name)
            vm = {
                "volume_index": vi, "series_id": sid, "eye": eye, "modality": mod,
                "laterality": jsonable(getattr(v, "laterality", None)),
                "acquisition_date": jsonable(getattr(v, "acquisition_date", None)),
                "num_bscans": n,
                "bscan_shape": list(shape) if shape is not None else None,
                "pixel_spacing": jsonable(getattr(v, "pixel_spacing", None)),
                "folder": os.path.join("bscans", folder_name),
            }
            vol_meta.append(vm)
            log(summary, f"   vol {vi}: {eye} {n} B-scans shape={shape} "
                         f"sid={sid} -> bscans/{folder_name}/")
            for bi, b in enumerate(bscans):
                save_png(b, os.path.join(folder, f"bscan_{bi:03d}.png"))
        metadata["volumes"] = vol_meta

        # ---- fundus / en-face images -> fundus/<MODALITY>/<EYE>_<sid>.png
        log(summary, "\n[oct-converter] reading fundus image(s) ...")
        try:
            funduses = e2e.read_fundus_image()
            log(summary, f"[oct-converter] {len(funduses)} fundus image(s) found")
            fundus_meta = []
            counts = {}
            for fi, f in enumerate(funduses):
                img = f.image
                sid = parse_sid(getattr(f, "image_id", None))
                eye = eye_tag(getattr(f, "laterality", None))
                mod = modality_map.get(sid, "UNKNOWN")
                rel = os.path.join("fundus", mod, f"{eye}_{sid}.png")
                save_png(img, os.path.join(out_dir, rel))
                counts[mod] = counts.get(mod, 0) + 1
                fundus_meta.append({
                    "fundus_index": fi, "series_id": sid, "eye": eye,
                    "modality": mod, "shape": list(np.asarray(img).shape),
                    "path": rel.replace("\\", "/"),
                })
                log(summary, f"   fundus {fi}: {mod} {eye} sid={sid} -> {rel}")
            metadata["fundus_images"] = fundus_meta
            log(summary, "   fundus modality counts: " +
                ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        except Exception as e:
            log(summary, f"[oct-converter] fundus read failed: {e!r}")
            metadata["fundus_error"] = repr(e)

    except Exception as e:
        log(summary, f"\n[oct-converter] FAILED: {e!r}")
        metadata["oct_converter_error"] = traceback.format_exc()

    # ---------------------------------------------------------------- cross-check
    log(summary, "\n" + "-" * 70)
    log(summary, "CROSS-CHECK")
    log(summary, f"   oct-converter total B-scans: {oct_bscan_total}  dims={oct_dims}")
    log(summary, f"   eyepy total B-scans        : {eyepy_bscan_total}  dims={eyepy_dims}")
    agree = (oct_bscan_total == eyepy_bscan_total and oct_bscan_total > 0)
    log(summary, f"   B-scan counts agree        : {agree}")
    log(summary, f"   layer segmentations found  : {layers_found}")
    metadata["cross_check"] = {
        "oct_converter_bscans": oct_bscan_total,
        "oct_converter_dims": [list(d) for d in oct_dims],
        "eyepy_bscans": eyepy_bscan_total,
        "eyepy_dims": [list(d) for d in eyepy_dims],
        "counts_agree": agree,
        "layers_found": layers_found,
    }

    # ---------------------------------------------------------------- write outputs
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary) + "\n")

    log(summary, "\nDONE. Outputs in: " + out_dir)


if __name__ == "__main__":
    main()
