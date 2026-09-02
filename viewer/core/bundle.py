"""Per-eye bundle: bake (write) + serve (read), and the ViewSource seam the routes call.

A library scan is served from a baked bundle on disk (numpy/PIL/cv2 ONLY — no oct-converter, no E2E
decode), so the offline app opens it instantly. An uploaded scan is served from a LiveSource that runs
the full pipeline once and holds the result in memory. Both expose the SAME ViewSource methods, so the
routes have one code path.

On-disk layout (viewer/data_store/library/<slug>/):
  meta.json       identity, FOV, areas (ours + PLEX), plex_polygons, slab, enface_out, …
  bundle.npz      vol(uint8 n,H,W) · bm/ilm(f32 n,W) · ga_enface(bool out,out) · ga_native(bool n,W) ·
                  loc_lines(f32 n,4) · field_invalid(bool n,W; saturated machine-fill cols, optional) ·
                  bm_dl(f32 n,W) + ga_native_dl(bool n,W)  (cached DL Bruch's-membrane variant, optional)
  localizer.png   raw IR (the viewer draws the locator lines on a canvas)
  projection.png  the f_trans en-face (windowed + 1 mm scale bar) — the right panel's base
  ga_overlay.png  predicted-GA translucent-green fill + contour (composited over projection.png)
  projection_dl.png / ga_overlay_dl.png   the same two, recomputed on the DL BM (optional)

The DL variant lets the "Use DL Bruch's-membrane" toggle (and the DL-default library view) serve a
cached, per-B-scan DL result with numpy/cv2 only — no E2E, no model, no onnxruntime — so it works in the
offline package too. The base arrays are the validated/device computation; only the BM-dependent layers
(bm, ga_native, projection, ga_overlay, area) have a `_dl` twin.
"""
import json
import os

import cv2
import numpy as np

from . import ga_native


def _to_png(arr):
    """Encode a uint8 array (HxW gray or HxWx3 RGB) as PNG bytes (cv2-only; no heavy imports)."""
    a = np.asarray(arr)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", a)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()

LIBRARY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data_store", "library")
SCHEMA = 3          # 2: + field_invalid (saturated machine-fill cols).  3: + cached DL BM variant
                    #    (bm_dl/ga_native_dl in npz, projection_dl.png/ga_overlay_dl.png, oac_area_dl_mm2)


def slug_for(subject, eye):
    return f"{subject}_{eye}"


def _round_rows(arr):
    """(n,W) float -> JSON rows, NaN -> None, 1-dp (compact BM transport for the overlay)."""
    a = np.asarray(arr, float)
    return [[None if not np.isfinite(v) else round(float(v), 1) for v in row] for row in a]


# --------------------------------------------------------------------------- writing (baker)
def write_bundle(slug, vm, meta, vm_dl=None):
    """Persist a computed view-model (viewmodel.compute output + PLEX/identity meta) as a bundle.

    `vm_dl` (optional): a SECOND viewmodel.compute output on the DL Bruch's-membrane. When given, the
    BM-dependent layers get a cached `_dl` twin (bm_dl/ga_native_dl + projection_dl.png/ga_overlay_dl.png)
    so the viewer can serve the DL view straight from the bundle. The base arrays stay the validated/
    device computation. Set meta['bm_dl_baked'] + meta['oac_area_dl_mm2'] alongside (the baker does)."""
    d = os.path.join(LIBRARY_DIR, slug)
    os.makedirs(d, exist_ok=True)
    n = int(meta["n_bscans"])
    loc = vm.get("loc_lines")
    fi = vm.get("field_invalid")
    W = int(meta["W"])
    arrays = dict(
        vol=vm["vol_u8"].astype(np.uint8),
        bm=np.asarray(vm["bm"], np.float32),
        ilm=np.asarray(vm["ilm"], np.float32),
        ga_enface=np.asarray(vm["ga_enface"], bool),
        ga_native=np.asarray(vm["ga_native"], bool),
        loc_lines=(np.zeros((n, 4), np.float32) if loc is None else np.asarray(loc, np.float32)),
        field_invalid=(np.zeros((n, W), bool) if fi is None else np.asarray(fi, bool)),
    )
    if vm_dl is not None:                                   # cached DL Bruch's-membrane variant (BM-dep only)
        arrays["bm_dl"] = np.asarray(vm_dl["bm"], np.float32)
        arrays["ga_native_dl"] = np.asarray(vm_dl["ga_native"], bool)
    np.savez_compressed(os.path.join(d, "bundle.npz"), **arrays)
    for name, key in (("localizer.png", "localizer_png"),
                      ("projection.png", "projection_png"),
                      ("ga_overlay.png", "ga_overlay_png")):
        if vm.get(key) is not None:
            with open(os.path.join(d, name), "wb") as f:
                f.write(vm[key])
    if vm_dl is not None:
        for name, key in (("projection_dl.png", "projection_png"), ("ga_overlay_dl.png", "ga_overlay_png")):
            if vm_dl.get(key) is not None:
                with open(os.path.join(d, name), "wb") as f:
                    f.write(vm_dl[key])
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return d


def write_index(rows):
    os.makedirs(LIBRARY_DIR, exist_ok=True)
    with open(os.path.join(LIBRARY_DIR, "index.json"), "w") as f:
        json.dump(rows, f, indent=2)


def read_index():
    p = os.path.join(LIBRARY_DIR, "index.json")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f)


# --------------------------------------------------------------------------- ViewSource (serving)
class _BaseSource:
    """Shared serving logic. Subclasses provide the raw arrays + meta + the three baked PNGs."""

    # -- subclass hooks --
    def meta_json(self):
        raise NotImplementedError

    def _vol(self):
        raise NotImplementedError

    def _bm(self):
        raise NotImplementedError

    def _ga_native(self):
        raise NotImplementedError

    def _field_invalid(self):
        """(n,W) bool saturated-machine-fill mask, or None if this source has none (old bundle)."""
        return None

    def _loc_lines(self):
        raise NotImplementedError

    def localizer_png(self):
        raise NotImplementedError

    def projection_png(self):
        raise NotImplementedError

    def ga_overlay_png(self):
        raise NotImplementedError

    def plex_correction_png(self):
        """The doctor's marked-up PLEX GA correction image (a reference, shown ABOVE the projection — not
        registered, not blended). None when this eye has no correction (most eyes / all uploads)."""
        return None

    # -- derived (identical for both backends) --
    def meta_public(self):
        m = self.meta_json()
        keys = ("slug", "subject", "visit", "eye", "patient_id", "acq_date", "n_bscans", "H", "W",
                "fov_mm", "axial_um_per_px", "enface_mmpp", "enface_out", "feature", "slab_um",
                "oac_area_mm2", "oac_area_dl_mm2", "plex_area_mm2", "plex_source", "is_control",
                "bm_source", "bm_dl_baked", "has_plex_correction",
                # absent on bundles baked before the reverse-scan fix -> the UI defaults it to true
                "enface_flip")
        out = {k: m.get(k) for k in keys}
        out["has_plex"] = bool(m.get("plex_polygons"))
        out["identity"] = self.identity()
        return out

    def identity(self):
        m = self.meta_json()
        who = m.get("subject") or m.get("patient_id") or "patient ?"
        return f"{who} · {m.get('eye', '—')}"

    def bscan_png(self, idx):
        v = self._vol()
        i = max(0, min(v.shape[0] - 1, int(idx)))
        return _to_png(v[i])

    def bm_rows(self):
        return _round_rows(self._bm())

    def ga_intervals(self):
        return [ga_native.intervals(row) for row in self._ga_native()]

    def field_invalid_runs(self):
        """Per-B-scan list of [start,end] inclusive saturated-machine-fill column runs (empty where the
        B-scan has no band). [] overall for bundles baked before this field existed (schema < 2)."""
        fi = self._field_invalid()
        if fi is None:
            return []
        out = []
        for row in np.asarray(fi, bool):
            runs, x, W = [], 0, len(row)
            while x < W:
                if row[x]:
                    s = x
                    while x < W and row[x]:
                        x += 1
                    runs.append([s, x - 1])
                else:
                    x += 1
            out.append(runs)
        return out

    def loc_lines(self):
        L = self._loc_lines()
        if L is None or len(L) == 0 or not np.any(L):
            return []
        # locator.line_endpoints returns the lines in REVERSED display order (its row j is the line of
        # vol[n-1-j]); this reversal restores display order, so served[i] is the line of B-scan i.
        # Reversing at serve time rather than at bake time keeps baked bundles and live uploads on one code
        # path -- and it is why a bundle baked before the locator scale fix still highlights the CORRECT
        # B-scan (only its extents are stale). Change locator's order and this line together, or neither.
        # A mirror bug here is invisible on a central/symmetric lesion -- both hypotheses fit -- so test any
        # change on an eccentric one (011 OD, 026 OD), and against the advRPE PLEX masks, which are the only
        # external witness to the fundus frame.
        return [[round(float(x), 2) for x in r] for r in L[::-1]]

    def plex_polygons(self):
        return self.meta_json().get("plex_polygons") or []

    def dashboard(self):
        m = self.meta_json()
        return {"oac_area_mm2": m.get("oac_area_mm2"), "plex_area_mm2": m.get("plex_area_mm2")}


class BundleSource(_BaseSource):
    """A baked library eye, served straight from disk (no oct-converter / no E2E decode).

    variant: "base" = the validated/device computation; "dl" = the cached DL Bruch's-membrane twin
    (bm_dl/ga_native_dl + projection_dl.png/ga_overlay_dl.png + oac_area_dl_mm2). The DL variant shares
    every BM-independent array (vol, ilm, loc_lines, localizer, PLEX, field_invalid) with the base."""

    def __init__(self, slug, variant="base", share_from=None):
        self.slug = slug
        self.variant = variant
        self.dir = os.path.join(LIBRARY_DIR, slug)
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(f"no library bundle '{slug}'")
        with open(os.path.join(self.dir, "meta.json")) as f:
            self._meta = json.load(f)
        if variant == "dl":
            if not self._meta.get("bm_dl_baked"):
                raise FileNotFoundError(f"bundle '{slug}' has no baked DL Bruch's-membrane")
            m = dict(self._meta)                            # report the DL area + source for this view
            if m.get("oac_area_dl_mm2") is not None:
                m["oac_area_mm2"] = m["oac_area_dl_mm2"]
            m["bm_source"] = "dl"
            self._meta = m
        self._npz = None
        # Share the decompressed-array + PNG caches with a sibling variant of the SAME bundle (e.g. the DL
        # twin shares the base source's caches), so the big BM-independent arrays (vol/ilm/loc_lines) and
        # PNGs decompress/read once. The cache is keyed by npz key / file name, and base vs dl read
        # different keys (bm/ga_native/projection.png vs bm_dl/ga_native_dl/projection_dl.png), so there is
        # no collision — only vol/ilm/loc/field_invalid/localizer are actually shared.
        self._cache = share_from._cache if share_from is not None else {}
        self._png_cache = share_from._png_cache if share_from is not None else {}

    def _arr(self, key):
        a = self._cache.get(key)
        if a is None:
            if self._npz is None:
                self._npz = np.load(os.path.join(self.dir, "bundle.npz"))
            a = self._cache[key] = self._npz[key]
        return a

    def _arr_opt(self, key):
        """Like _arr but returns None if the bundle predates the key (back-compat for schema < 2)."""
        if self._npz is None:
            self._npz = np.load(os.path.join(self.dir, "bundle.npz"))
        return self._arr(key) if key in self._npz.files else None

    def meta_json(self):
        return self._meta

    def _vol(self):
        return self._arr("vol")

    def _bm(self):
        return self._arr("bm_dl" if self.variant == "dl" else "bm")

    def _ga_native(self):
        return self._arr("ga_native_dl" if self.variant == "dl" else "ga_native")

    def _field_invalid(self):
        return self._arr_opt("field_invalid")

    def _loc_lines(self):
        return self._arr("loc_lines")

    def _png(self, name):
        if name not in self._png_cache:
            p = os.path.join(self.dir, name)
            self._png_cache[name] = open(p, "rb").read() if os.path.exists(p) else None
        return self._png_cache[name]

    def localizer_png(self):
        return self._png("localizer.png")

    def projection_png(self):
        return self._png("projection_dl.png" if self.variant == "dl" else "projection.png")

    def ga_overlay_png(self):
        return self._png("ga_overlay_dl.png" if self.variant == "dl" else "ga_overlay.png")

    def plex_correction_png(self):
        return self._png("plex_correction.png")    # BM-independent → same file for base + dl variants


class LiveSource(_BaseSource):
    """An uploaded E2E, processed once via viewmodel.compute and held in memory. No PLEX reference."""

    def __init__(self, vm, meta):
        self._vm = vm
        self._meta = meta

    def meta_json(self):
        return self._meta

    def _vol(self):
        return self._vm["vol_u8"]

    def _bm(self):
        return self._vm["bm"]

    def _ga_native(self):
        return self._vm["ga_native"]

    def _field_invalid(self):
        return self._vm.get("field_invalid")

    def _loc_lines(self):
        return self._vm.get("loc_lines")

    def localizer_png(self):
        return self._vm.get("localizer_png")

    def projection_png(self):
        return self._vm.get("projection_png")

    def ga_overlay_png(self):
        return self._vm.get("ga_overlay_png")
