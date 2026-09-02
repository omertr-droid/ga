"""Recompute ONLY the `loc_lines` array of each baked library bundle, in place.

Why this exists: `viewer/core/locator.py` used to min-max normalize a raster's own angular extent to fill
the whole IR, which stretched the per-B-scan locator lines (a 6x6/20deg scan was drawn edge-to-edge, across
the optic disc). Fixing the scale changes only `loc_lines`; every other baked array (vol, bm, ilm, ga_*,
field_invalid, bm_dl, ga_native_dl) and every baked PNG is unaffected, so a full `bake_library.py` run --
which recomputes GA and a DL twin per eye, ~30-60 min -- is unnecessary. This does the same job in ~1-2 min.

The change is SCALING-ONLY and therefore order-preserving: an un-refreshed bundle still highlights the
correct B-scan (right ordering, right vertical sense), it just draws the old stretched extents. Nothing
mirrors. So refreshing is a visual-accuracy fix, not a correctness emergency.

Run from the repo root:
    oct_env\\Scripts\\python.exe src\\refresh_loc_lines.py            # all library bundles
    oct_env\\Scripts\\python.exe src\\refresh_loc_lines.py --dry-run  # report, write nothing
    oct_env\\Scripts\\python.exe src\\refresh_loc_lines.py --only 008
"""
import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

from reader.core import e2e_source                                                    # noqa: E402
from viewer.core import bundle, locator                                               # noqa: E402

sys.path.insert(0, os.path.join(_REPO, "src"))
import bake_library                                                                   # noqa: E402


def refresh(only=None, dry_run=False):
    rows = bake_library.library_rows()
    # group by E2E file: both eyes of a subject share one 250-300 MB decode
    by_file = {}
    for r in rows:
        slug = bundle.slug_for(r["subject"], r["eye"])
        if only and only.lower() not in slug.lower():
            continue
        by_file.setdefault(r["e2e_file"], []).append(r)
    if not by_file:
        print("no matching bundles")
        return 0

    n_ok = n_skip = 0
    for e2e_file, group in sorted(by_file.items()):
        e2e_path = os.path.join(bake_library.DATA, e2e_file)
        if not os.path.exists(e2e_path):
            for r in group:
                print(f"  SKIP {bundle.slug_for(r['subject'], r['eye'])}: E2E missing")
                n_skip += len(group)
            continue
        raw = e2e_source.open_e2e(e2e_path)                       # one decode, both eyes
        for r in group:
            slug = bundle.slug_for(r["subject"], r["eye"])
            npz = os.path.join(bundle.LIBRARY_DIR, slug, "bundle.npz")
            if not os.path.isfile(npz):
                print(f"  SKIP {slug}: no bundle.npz")
                n_skip += 1
                continue
            idx = e2e_source.default_volume_index(raw, r["eye"])
            loc = locator.pick_localizer(raw, idx)
            if loc is None:
                print(f"  SKIP {slug}: no localizer")
                n_skip += 1
                continue
            L = locator.line_endpoints(raw, idx, loc.shape)
            if L is None:
                print(f"  SKIP {slug}: no per-B-scan records")
                n_skip += 1
                continue
            L = L.astype(np.float32)

            # materialize inside the context manager: an open NpzFile keeps a handle on the zip, and
            # Windows refuses to os.replace() a file that is still open.
            with np.load(npz) as zf:
                z = {k: zf[k] for k in zf.files}
            old = z.get("loc_lines")
            if old is not None and old.shape != L.shape:
                print(f"  SKIP {slug}: shape {old.shape} != {L.shape}")
                n_skip += 1
                continue

            # report the visible change: served order is L[::-1] (see bundle._BaseSource.loc_lines)
            H = float(loc.shape[0])
            oy, ny = (old[:, 1] + old[:, 3]) / 2, (L[:, 1] + L[:, 3]) / 2
            wf_old = (old[:, [0, 2]].max() - old[:, [0, 2]].min()) / loc.shape[1]
            wf_new = (L[:, [0, 2]].max() - L[:, [0, 2]].min()) / loc.shape[1]
            print(f"  {slug:24s} y-frac first/last {oy[::-1][0]/H:.3f}/{oy[::-1][-1]/H:.3f}"
                  f" -> {ny[::-1][0]/H:.3f}/{ny[::-1][-1]/H:.3f}   width {wf_old:.3f} -> {wf_new:.3f}")
            if dry_run:
                n_ok += 1
                continue

            z["loc_lines"] = L
            tmp = npz + ".tmp.npz"
            try:
                np.savez_compressed(tmp, **z)                      # atomic: write then replace
                os.replace(tmp, npz)
            except OSError as e:                                   # held by OneDrive / another process
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                print(f"    !! {slug}: could not write ({e}); bundle left untouched")
                n_skip += 1
                continue
            n_ok += 1

    print(f"\n{'would refresh' if dry_run else 'refreshed'}: {n_ok}   skipped: {n_skip}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="substring of the bundle slug")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    raise SystemExit(refresh(a.only, a.dry_run))
