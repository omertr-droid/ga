"""En-face GA mask (out x out) -> native (n_bscans, W) GA-column flags.

Inverse of reader.core.projection.to_enface (centred resample about the image centre + slow-axis flip).
Lifted from reader/api/routes_segmentation._enface_to_native_mask + _center_extract so the doctor app
never imports the dev web router. The native flags drive the middle-panel "predicted GA A-scans"
highlight (a band over the sub-BM slab for the flagged columns).
"""
import cv2
import numpy as np

# Fixed en-face scale (register_qc.ADV_MMPP = reader.core.projection.ENFACE_MMPP = 6/512 mm/px). Kept as
# a literal so the library serve path (bundle.py) needs only numpy+cv2 — no heavy projection import.
MMPP = 6.0 / 512.0


def enface_out(fov_mm):
    """The square en-face frame size to.to_enface produces for this field (matches projection.to_enface)."""
    return max(64, int(round(max(fov_mm) / MMPP)))


def center_extract(a, fh, fw):
    """Pull the centred (fh, fw) field out of (or pad into) a square en-face frame. Inverse of the
    centre-pad in projection.to_enface / register_qc.resample. (== routes_segmentation._center_extract)"""
    H, W = a.shape
    out = np.zeros((fh, fw), a.dtype)
    oy, ox = (fh - H) // 2, (fw - W) // 2
    sy0, sx0 = max(0, -oy), max(0, -ox)
    dy0, dx0 = max(0, oy), max(0, ox)
    h, w = min(H - sy0, fh - dy0), min(W - sx0, fw - dx0)
    if h > 0 and w > 0:
        out[dy0:dy0 + h, dx0:dx0 + w] = a[sy0:sy0 + h, sx0:sx0 + w]
    return out


def enface_to_native(enf, fov_mm, n, W, flip=True):
    """(out, out) bool en-face GA mask -> (n, W) bool native GA-column map.

    `flip` must match the forward pass (projection.to_enface / ov.enface_flip): un-flip only if the
    en-face rows were reversed on the way out. Reverse-scanned rasters (003-016, 003-130) pass flip=False."""
    fh = max(1, int(round(fov_mm[1] / MMPP)))
    fw = max(1, int(round(fov_mm[0] / MMPP)))
    field = center_extract(np.asarray(enf, bool).astype(np.uint8), fh, fw)
    nat = cv2.resize(field, (W, n), interpolation=cv2.INTER_NEAREST)
    return (nat[::-1] if flip else nat) > 0


def intervals(row):
    """[start, end) column intervals where a boolean row is True (compact transport for the highlight)."""
    out, x, m = [], 0, len(row)
    while x < m:
        if row[x]:
            s = x
            while x < m and row[x]:
                x += 1
            out.append([int(s), int(x)])
        else:
            x += 1
    return out
