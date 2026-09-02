"""Real per-B-scan locator lines for the doctor viewer's IR panel.

Each Spectralis B-scan stores its TRUE endpoints on the SLO localizer in DEGREES about fixation
(posX1/posY1 -> posX2/posY2, slightly tilted: posY1 != posY2). We draw those real endpoints, mapped to
localizer pixels through the SLO's OWN angular field, and pair the IR localizer to the SAME acquisition
series as the volume (fundus image_id sid == volume_id sid), so the line frame matches the IR exactly.

This replaces the linear-over-image-height estimate in reader/core/render.localizer_png (whose own
comment flags it as approximate).

THE SCALE IS THE SLO's, NOT THE RASTER's. An earlier version min-max normalized each raster's own angular
extent to fill the whole localizer, on the premise that the IR was "captured at the raster field". That is
false: the paired IR is the wider ~30deg macular SLO (the optic disc is visible in-frame, ~15.5deg from the
fovea -- a 20deg raster never images it). The effect was that a 6x6 (+/-10deg) scan's lines were stretched
1.5x, pinning the first/last B-scan to the very top/bottom edge and running the lines across the disc. Only
the CENTRE line was right; the error grew toward the ends. Empirically, under FIELD_DEG the 20deg raster now
spans 0.667 of the IR width and the 30deg (121-line) raster spans ~1.00 -- the internal check that this is
the correct field. It also preserves a genuinely DE-CENTRED raster (e.g. 003-003, posY -9.6..+10.5), which
min-max silently re-centred.

No SLO scale exists in the E2E to read: oct-converter's fundus `pixel_spacing` is a hardcoded [0.01,0.01]
default, and eyepy hardcodes `30/width` deg/px and logs "The localizer scale is currently hardcoded and not
read from the E2E file." FIELD_DEG below is therefore a documented ASSUMPTION, not a file value.

Association: bscan_data concatenates EVERY volume's per-B-scan records (both eyes, interleaved in file
order). We filter to this volume's class (numImages, imgSizeX) -> 2*n records (two eyes), split into
consecutive n-record blocks (each eye is contiguous), pick the block for this volume's ordinal among
same-class volumes, and order it by aktImage (the in-volume B-scan index). The two eyes raster the
SAME +/-10deg field with the same stepping, so the geometry is eye-robust; the only eye-specific choice
that matters (the localizer image, and the vertical orientation) is fixed by the sid pairing + QC.
"""
import numpy as np

from reader.core import e2e_source

# Angular field of the co-acquired macular IR/SLO (degrees, square). NOT stored in the E2E -- see the module
# docstring. Cross-checked: the 121-line 30deg raster maps to ~1.00 of the IR width under this constant, and
# the 97-line 20deg raster to 0.667 (= 20/30). Wide-field outliers (measured raster spans 29.8-32.6deg) can
# overshoot the frame edge by a few percent; the canvas simply clips, which is honest.
FIELD_DEG = 30.0

# posY sign: POSITIVE posY is INFERIOR, i.e. a LARGER localizer row. Established, not assumed:
#   * vol[i] sits at raster-fraction (n-1-i)/(n-1) from the top  (cross-checked against the advRPE PLEX
#     masks on 5 decisive eccentric-lesion eyes, 0 mirrored);
#   * vol[i] == aktImage i  (fovea tie-break: the thinnest-central-retina B-scan lands on the aktImage whose
#     posY crosses 0, on 008 OD/OS where the raster is de-centred enough to distinguish the hypotheses);
#   * posY(aktImage 0) = +10 on every eye but 016/130.
# Together: B-scan 0 carries posY=+10 and lies at the BOTTOM of the field  =>  py = cy + posY*ky.
#
# An earlier version had this sign inverted (`FLIP_Y=True`, py = cy - posY*ky) and compensated with the
# serve-time reversal in bundle._BaseSource.loc_lines(). Those two cancel EXACTLY only when the raster is
# centred (mean posY == 0). On a de-centred raster they mirror the decentration: up to 113 px -- 14.7% of
# the IR height -- on 011 OD (mean posY +2.21 deg), 98 px on 014 OD, 94 px on 008 OS. Most cohort rasters
# are near-centred, which is why it went unseen.
#
# NOTE ON ORDER: bundle._BaseSource.loc_lines() serves this array REVERSED. We therefore return the lines in
# reversed display order (row j = the line of vol[n-1-j]) so the serve-time reversal restores display order.
# Keeping that contract means an un-refreshed baked bundle still highlights the correct B-scan (it just draws
# the old, stretched extents) -- nothing mirrors. Change both together or neither.

# Some eyes raster the slow axis the other way (posY INCREASES with aktImage): 003-016 and 003-130, both
# eyes. The mapping below is direction-agnostic -- it places each B-scan by its own posY -- so those eyes
# need no special case here.


def pick_localizer(raw, index):
    """The IR/NIR localizer co-acquired with volume `index` (the fundus whose series id == the
    volume's series id). Falls back to the eye's first IR/NIR, then any fundus for the eye, else None."""
    v = raw.vols[index]
    eye = e2e_source.eye_tag(getattr(v, "laterality", None))
    vsid = e2e_source._parse_sid(getattr(v, "volume_id", None))
    funduses = e2e_source._funduses(raw)
    cands = [f for f in funduses if e2e_source.eye_tag(getattr(f, "laterality", None)) == eye]
    pick = next((f for f in cands
                 if e2e_source._parse_sid(getattr(f, "image_id", None)) == vsid), None)
    if pick is None:                                   # fall back to an IR/NIR modality, then any
        def mod_of(f):
            return str(raw.modmap.get(e2e_source._parse_sid(getattr(f, "image_id", None)), "")).upper()
        ir = [f for f in cands if "IR" in mod_of(f) or "NIR" in mod_of(f)]
        pick = (ir or cands or [None])[0]
    if pick is None:
        return None
    try:
        return np.asarray(pick.image)
    except Exception:
        return None


def localizer_sid(raw, index):
    """The series id of the localizer paired with volume `index` (for the bundle's audit trail)."""
    v = raw.vols[index]
    eye = e2e_source.eye_tag(getattr(v, "laterality", None))
    vsid = e2e_source._parse_sid(getattr(v, "volume_id", None))
    for f in e2e_source._funduses(raw):
        if (e2e_source.eye_tag(getattr(f, "laterality", None)) == eye
                and e2e_source._parse_sid(getattr(f, "image_id", None)) == vsid):
            return vsid
    return None


def _volume_records(raw, index):
    """This volume's per-B-scan position records in DISPLAY order (B-scan 0..n-1), or None."""
    v = raw.vols[index]
    n = len(v.volume)
    W = int(np.asarray(v.volume[0]).shape[1])
    bd = raw.md.get("bscan_data") or []
    recs = [b for b in bd
            if b.get("numImages") == n and b.get("imgSizeX") == W and b.get("posX1") is not None]
    if len(recs) < n:
        return None
    same = [r for r in raw.refs if r.n_bscans == n and r.W == W]
    ordinal = next((k for k, r in enumerate(same) if r.index == index), 0)
    nblocks = max(1, len(recs) // n)
    b0 = min(ordinal, nblocks - 1) * n
    blk = recs[b0:b0 + n]
    if len(blk) != n:
        blk = recs[:n]
    akt = [b.get("aktImage") for b in blk]
    if sorted(a for a in akt if a is not None) == list(range(n)):
        order = sorted(range(n), key=lambda k: blk[k]["aktImage"])
    else:                                              # fallback: slow-axis monotonic across a raster
        order = sorted(range(n), key=lambda k: blk[k].get("centrePosY", 0.0))
    return [blk[k] for k in order]


def line_endpoints(raw, index, loc_shape):
    """(n,4) float32 [x1,y1,x2,y2] localizer-pixel endpoints, in REVERSED display order, or None.

    Real tilted endpoints from posX/posY (degrees about fixation), placed on the co-acquired localizer at
    the SLO's own FIELD_DEG scale — so the raster occupies its true fraction of the IR (a 6x6/20deg scan
    covers the central ~2/3) instead of being stretched to the frame, and a de-centred raster stays
    de-centred. Row j is the line of vol[n-1-j]: the caller (bundle._BaseSource.loc_lines) serves the array
    reversed, which restores display order. Returns None when the per-B-scan records are missing; callers
    already degrade to "draw no lines" (viewmodel.compute -> loc_lines=None)."""
    recs = _volume_records(raw, index)                      # aktImage order == vol order
    if not recs:
        return None
    H_loc, W_loc = int(loc_shape[0]), int(loc_shape[1])
    x1 = np.array([r["posX1"] for r in recs], float)
    x2 = np.array([r["posX2"] for r in recs], float)
    y1 = np.array([r["posY1"] for r in recs], float)
    y2 = np.array([r["posY2"] for r in recs], float)
    kx = W_loc / FIELD_DEG                                  # px per degree
    ky = H_loc / FIELD_DEG
    cx = (W_loc - 1) / 2.0                                  # fixation (0,0) deg -> the IR's centre
    cy = (H_loc - 1) / 2.0

    def px(x):
        return cx + np.asarray(x) * kx

    def py(y):
        return cy + np.asarray(y) * ky                      # +posY is inferior (see the note above)

    lines = np.stack([px(x1), py(y1), px(x2), py(y2)], axis=1).astype(np.float32)
    return lines[::-1].copy()                               # reversed: the serve-time [::-1] undoes it
