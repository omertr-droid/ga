#!/usr/bin/env python
"""Export the device-BM eyes as WEAK pre-training labels (device BM, no hand-validation).

Optional warm-start for the BM segmenter: every device-BM 6x6 eye in results/bm_worklist.csv, each
B-scan's image + the gap-filled DEVICE BM as a (weak) label. Device BM DIVES under GA, so this is a
warm-start ONLY — the gold eyes (src/export_bm_dataset.py) fix the GA columns during fine-tuning, and the
training notebook excludes the held-out patient from each fold's pre-training pool (leakage guard).

The notebook only uses this when USE_PRETRAIN=True (off by default). npz keys are BMData-compatible
(images/bm/device_bm/weight/bscan_idx) so the notebook reuses its dataset class unchanged: bm == device_bm
(hardness 0) and weight == 1 (uniform -> plain L1 in pre-training).

Out (outputs/bm_dataset_weak/, gitignored, regenerable):
  npz/<subject>_<eye>.npz   images uint8(N,H,W); bm/device_bm/weight float32(N,W); bscan_idx int32(N);
                            patient str; H,W,fov_x_mm,fov_y_mm scalars.
  bm_dataset_weak.zip       npz/, one-shot Colab upload.

Run from the repo root:
  oct_env\\Scripts\\python.exe src\\export_bm_weak.py            # full export + zip
  oct_env\\Scripts\\python.exe src\\export_bm_weak.py --stats     # dry run: list eyes, write nothing
"""
import argparse
import os
import sys
import zipfile

import cv2
import numpy as np

from paths import DATA_DIR, OUT_DIR, REPO_ROOT

sys.path.insert(0, REPO_ROOT)                          # so `import reader` resolves from a src/ script
from reader.core import e2e_source, render             # noqa: E402
from export_bm_dataset import _find_6x6, _fill_nan, _patient, _worklist  # noqa: E402  (shared helpers)

OUT = os.path.join(OUT_DIR, "bm_dataset_weak")
NPZ_DIR = os.path.join(OUT, "npz")


def export_eye(subject, eye, meta):
    """Return a BMData-compatible npz dict of device-BM weak labels for one eye, or None."""
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
    device_bm = np.asarray(ov.bm, float)
    N, H, W = ov.n_bscans, ov.H, ov.W
    images = np.zeros((N, H, W), np.uint8)
    bm_arr = np.zeros((N, W), np.float32)
    for i in range(N):
        images[i] = cv2.imdecode(np.frombuffer(render.bscan_png(ov, i), np.uint8), cv2.IMREAD_GRAYSCALE)
        bm_arr[i] = _fill_nan(device_bm[i]).astype(np.float32)
    print(f"  {subject} {eye}: N={N} H={H} W={W} bm_src={ov.bm_src} (device weak label)")
    return {
        "images": images, "bm": bm_arr, "device_bm": bm_arr.copy(),
        "weight": np.ones((N, W), np.float32), "bscan_idx": np.arange(N, dtype=np.int32),
        "patient": _patient(subject),
        "H": np.int32(H), "W": np.int32(W),
        "fov_x_mm": np.float32(ov.fov_mm[0]), "fov_y_mm": np.float32(ov.fov_mm[1]),
    }


def build_zip():
    zpath = os.path.join(OUT, "bm_dataset_weak.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(NPZ_DIR)):
            if f.endswith(".npz"):
                z.write(os.path.join(NPZ_DIR, f), arcname=f"npz/{f}")
    print(f"  zip -> {zpath}  ({os.path.getsize(zpath) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description="Export device-BM eyes as weak BM pre-training labels.")
    ap.add_argument("pairs", nargs="*", help="optional SUBJECT EYE ... to limit the eyes (default: all device-BM)")
    ap.add_argument("--stats", action="store_true", help="dry run: list eyes, write nothing")
    ap.add_argument("--no-zip", action="store_true", help="skip building bm_dataset_weak.zip")
    args = ap.parse_args()

    wl = _worklist()
    eyes = sorted((s, e) for (s, e), r in wl.items()
                  if str(r.get("has_device_bm", "")).strip().lower() == "true")
    if args.pairs:
        eyes = [(args.pairs[i], args.pairs[i + 1]) for i in range(0, len(args.pairs) - 1, 2)]
    print(f"{len(eyes)} device-BM eyes")
    if args.stats:
        for s, e in eyes:
            print(f"  {s} {e}")
        return

    os.makedirs(NPZ_DIR, exist_ok=True)
    n = 0
    for s, e in eyes:
        npz = export_eye(s, e, wl[(s, e)])
        if npz is None:
            continue
        np.savez_compressed(os.path.join(NPZ_DIR, f"{s}_{e}.npz"), **npz)
        n += 1

    print(f"\n{n} eyes -> {NPZ_DIR}")
    if n and not args.no_zip:
        build_zip()
    print(f"DONE -> {OUT}/  (npz/{'' if args.no_zip else ' bm_dataset_weak.zip'})")


if __name__ == "__main__":
    main()
