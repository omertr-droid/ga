"""En-face projection adapter — speaks "features" to the API, reuses src/m3_projections for the math.

Features (native = per-A-scan (n,W), then destriped/normalised + resampled to a square en-face frame):
  raw      Spectralis-style sub-BM slab (mean intensity)   window [0.0, 1.0]     device-like view
  f_trans  transmission fraction (ILM-anchored)            window [0.18, 0.62]   GA bright
  f_gated  transmission gated by RPE-integrity (deliver.)  window [0.02, 0.40]   GA only where RPE gone
  f_rpe    shadow-invariant RPE-loss cue                   window [-0.85, 0.05]  high = RPE lost

The en-face uses the NATIVE 6x6 field with NO central crop: we resample the whole field to isotropic
mm pixels in a square frame sized to contain it (some 6x6 verticals exceed 6 mm), keeping the row-flip
to fundus orientation. mm/px is fixed at register_qc.ADV_MMPP (6/512) so scale bars stay consistent.

Native maps are split out (native_full / _native_one / finish) so a live layer edit can recompute ONLY
the edited B-scan's native row and patch it into a cached map — the basis for real-time edit preview
(preview_enface), instead of re-summing the whole volume on every drag.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

import m3_projections as mp
import register_qc as reg

ENFACE_MMPP = reg.ADV_MMPP            # 6/512 mm/px, isotropic
AX = mp.AX                           # axial um/px
RAW_SLAB = mp.SLAB_UM                # sub-BM slab (BM+10..+340um) — matches the B-scan 'slab' band

FEATURES = {
    "raw": (0.0, 1.0),
    "f_trans": (0.18, 0.62),
    "f_gated": (0.02, 0.40),
    "f_rpe": (-0.85, 0.05),
    "f_oac_rpe": (0.0, 0.6),     # BM-anchored OAC: max above BM -> bright=RPE present, dark=RPE gone=GA
    "f_oac_sub": (0.0, 0.4),     # sub-BM mean OAC -> bright=hypertransmission=GA
    "f_oac_elev": (0.0, 40.0),   # RPE->BM elevation (um) -> bright=drusen lift, dark=flat (atrophy/normal)
}
DEFAULT_FEATURE = "f_trans"
# features with a per-row-patchable native (fast live-edit preview)
SINGLE_NATIVE = ("raw", "f_trans", "f_rpe", "f_oac_rpe", "f_oac_sub", "f_oac_elev")


def default_window(feature):
    return FEATURES.get(feature, FEATURES[DEFAULT_FEATURE])


def to_enface(nat, fov, flip=True):
    """Flip to fundus orientation + isotropic-resample the WHOLE native field (no crop) into a
    square frame at ENFACE_MMPP, sized to contain the field.

    `flip` (= ov.enface_flip): rows are reversed only when the raster runs bottom->top in the fundus,
    which is every eye except the reverse-scanned 003-016 / 003-130. Default True = historical behaviour."""
    flipped = np.asarray(nat, np.float32)[::-1] if flip else np.asarray(nat, np.float32)
    src_mmpp = (fov[0] / flipped.shape[1], fov[1] / flipped.shape[0])
    out = max(64, int(round(max(fov) / ENFACE_MMPP)))
    return reg.resample(flipped, src_mmpp, out=out)


def to_enface_mask(nat_bool, fov, flip=True):
    """Like to_enface but for a (n,W) boolean field-validity mask: the SAME row-flip + isotropic
    resample (and the SAME `out` size, so it aligns 1:1 with to_enface output) but NEAREST interpolation
    so the result stays {0,1}. Off-frame pixels resample to 0 (borderValue=0) = invalid, which is correct.
    Returns a bool (out,out) en-face mask (True = valid in-field)."""
    nb = np.asarray(nat_bool, bool)
    flipped = (nb[::-1] if flip else nb).astype(np.float32)
    src_mmpp = (fov[0] / flipped.shape[1], fov[1] / flipped.shape[0])
    out = max(64, int(round(max(fov) / ENFACE_MMPP)))
    return reg.resample(flipped, src_mmpp, out=out, interp=cv2.INTER_NEAREST) > 0.5


def _transmit_native(vol, ilm, bm, slab=mp.SLAB_UM):
    """ILM-anchored transmission fraction with a tunable sub-BM slab (default == proj_transmit_ilm).
    num = sum(BM+slab_lo .. BM+slab_hi); den = sum(ILM .. BM+slab_hi)."""
    num = mp.band(vol, bm, slab[0], slab[1], "sum")
    den = mp.band_surfaces(vol, ilm, bm + slab[1] / AX, "sum")
    return np.clip(num / (den + 1e-3), 0, 1.5)


# --------------------------------------------------------------------------- native maps (pre-finish)
def native_full(ov, feature, surfaces, slab=None):
    """Pre-finish native (n,W) map for a SINGLE_NATIVE feature (raw/f_trans/f_rpe). `slab`=(lo,hi) µm
    tunes the sub-BM window for the slab-based features (raw/f_trans); None = the feature default."""
    ilm, bm = surfaces
    if feature == "raw":
        lo, hi = slab or RAW_SLAB
        return mp.band(ov.vol, bm, lo, hi, "mean")
    if feature == "f_trans":
        return _transmit_native(ov.vol, ilm, bm, slab or mp.SLAB_UM)
    if feature == "f_rpe":
        return mp.proj_rpe_loss_ilm(ov.vol, ilm, bm)
    if feature == "f_oac_rpe":
        return mp.proj_oac_rpe_above_bm(ov.vol, bm)
    if feature == "f_oac_sub":
        return mp.proj_oac_subbm(ov.vol, bm)
    if feature == "f_oac_elev":
        return mp.proj_oac_rpe_elevation(ov.vol, bm)
    raise ValueError(f"native_full: unsupported feature {feature!r}")


def _native_one(vol_i, ilm_i, bm_i, feature):
    """The native (W,) row for one B-scan from its own surfaces — used to patch a cached native map."""
    v = vol_i[None]
    nat = native_full(_OneVol(v), feature, (ilm_i[None], bm_i[None]))
    return nat[0]


class _OneVol:
    """Minimal volume shim so native_full can run on a single B-scan (it only reads .vol)."""
    __slots__ = ("vol",)

    def __init__(self, vol):
        self.vol = vol


def _norm01(m):
    """Robust [1,99] percentile stretch to [0,1] (the device-like auto-contrast for the raw view)."""
    f = np.isfinite(m)
    lo, hi = (np.percentile(m[f], [1, 99]) if f.any() else (0.0, 1.0))
    return np.clip((m - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def finish(ov, feature, nat):
    """Native (n,W) -> square en-face. raw = robust-normalise, no destripe (device-like); the cue
    features destripe (signed for the RPE-loss ratio that crosses zero); f_oac_elev is a geometric um
    surface so it skips destriping (striping correction would corrupt the um values)."""
    fl = getattr(ov, "enface_flip", True)
    if feature == "raw":
        return to_enface(_norm01(nat), ov.fov_mm, fl)
    if feature == "f_oac_elev":
        return to_enface(nat, ov.fov_mm, fl)
    return to_enface(mp.destripe2d(nat, signed=(feature == "f_rpe")), ov.fov_mm, fl)


# --------------------------------------------------------------------------- composed en-face
def _gated(ov, surfaces):
    """transmission gated by RPE-integrity (mirrors mp.gated_feature but with to_enface = no crop)."""
    ilm, bm = surfaces
    fl = getattr(ov, "enface_flip", True)
    t = mp.destripe2d(_transmit_native(ov.vol, ilm, bm), signed=False)
    p = mp.destripe2d(mp.proj_rpe_present_ilm(ov.vol, ilm, bm), signed=False)
    f_trans = to_enface(t, ov.fov_mm, fl)
    pres = to_enface(p, ov.fov_mm, fl)
    gate = mp.rpe_gone_gate(gaussian_filter(pres, mp.GATE_SMOOTH_PX))
    return np.clip(f_trans, 0.0, None) * gate


def enface(ov, feature, surfaces):
    """The en-face map (square float frame) for a feature, computed live from the volume + surfaces."""
    if feature == "f_gated":
        return _gated(ov, surfaces)
    return finish(ov, feature, native_full(ov, feature, surfaces))


def render_feature(ov, feature, surfaces, slab=None):
    """Full en-face recompute for any feature with an optional custom sub-BM slab (raw/f_trans). Used by
    the live preview when a whole-volume shift or a slab change means the cached native can't be patched."""
    if feature == "f_gated":
        return _gated(ov, surfaces)
    return finish(ov, feature, native_full(ov, feature, surfaces, slab))


def preview_enface(ov, feature, surfaces, bscan, layer, ys, nat=None):
    """En-face with a LIVE (unsaved) layer override on ONE B-scan. For SINGLE_NATIVE features only the
    edited B-scan's native row is recomputed and patched into `nat` (the cached full native), so the
    preview is cheap enough for real-time dragging/shifting. f_gated falls back to a full recompute."""
    ilm, bm = surfaces
    ilm_i = np.asarray(ys, float) if layer == "ilm" else ilm[bscan]
    bm_i = np.asarray(ys, float) if layer == "bm" else bm[bscan]
    if feature == "f_gated":
        ilm2, bm2 = ilm.copy(), bm.copy()
        ilm2[bscan], bm2[bscan] = ilm_i, bm_i
        return _gated(ov, (ilm2, bm2))
    nat = (native_full(ov, feature, surfaces) if nat is None else nat).copy()
    nat[bscan] = _native_one(ov.vol[bscan], ilm_i, bm_i, feature)
    return finish(ov, feature, nat)


def recompute_transmit(ov, surfaces, slab_lo_um, slab_hi_um):
    """Live band-tuned transmission for the projection-tuning sliders (numerator/denominator-bottom
    slab BM+slab_lo .. BM+slab_hi). Returns a square float en-face frame."""
    ilm, bm = surfaces
    nat = _transmit_native(ov.vol, ilm, bm, (float(slab_lo_um), float(slab_hi_um)))
    return to_enface(mp.destripe2d(nat, signed=False), ov.fov_mm)


def bands_meta():
    return {"slab_um": list(mp.SLAB_UM), "rpe_um": list(mp.RPEBAND_UM)}
