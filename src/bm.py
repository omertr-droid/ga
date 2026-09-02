#!/usr/bin/env python
"""M2 - Bruch's membrane (BM) self-segmentation from a Spectralis OCT volume.

Classical graph-search (Chiu/Garvin style): per B-scan, find the smooth continuous surface at
the bottom of the RPE/BM complex (the strongest bright->dark transition there), anchored to a
robust RPE guide so it survives GA (where the RPE is gone but BM persists). A 3D smoothing pass
regularises BM across the slow axis. BM persists under atrophy, so it is the right reference
floor for the sub-BM hypertransmission slab (M3).

Production interface (DL-ready - swap the body for a model later):
    segment_bm(bscan)            -> y[width]            BM row per A-scan, one B-scan
    segment_volume(volume)       -> bm[n_bscans, width] BM surface for the whole volume
"""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter, grey_closing
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve

EPS = 1e-3
AXIAL_UM_PER_PX = 3.8716699928045273  # Spectralis axial sampling (from the E2E)


def _norm(b):
    b = b.astype(np.float32)
    lo, hi = np.percentile(b, 1), np.percentile(b, 99.5)
    return np.clip((b - lo) / (hi - lo + 1e-6), 0, 1)


def _denoise(b):
    return gaussian_filter(median_filter(b, size=(3, 3)), sigma=0.8)


def _edge(b, sign):
    """Rectified vertical gradient, normalised to [0,1]. sign=+1 dark->bright, -1 bright->dark."""
    g = np.zeros_like(b)
    g[:-1] = sign * (b[1:] - b[:-1])
    g = np.clip(g, 0, None)
    return g / (g.max() + 1e-6)


def _first_above(b, frac=0.25):
    """Topmost row per column exceeding frac*column-max (rough ILM); column-median smoothed.
    LEGACY seed (fragile on vitreous speckle); superseded by _complex_anchor + _ilm_from_complex.
    Kept because reader/core/e2e_source imports it directly."""
    H, W = b.shape
    colmax = b.max(0)
    mask = b > (frac * colmax)[None, :]
    has = mask.any(0)
    first = np.argmax(mask, 0)
    med = int(np.median(first[has])) if has.any() else H // 3
    first[~has] = med
    return median_filter(first, size=31)


def _brightest(b, lo, hi):
    """Per-column argmax row within [lo,hi] (rough RPE/BM-complex centre); median smoothed.
    LEGACY (ILM-seeded); superseded by _complex_anchor's sustained-brightness anchor."""
    H, W = b.shape
    out = np.empty(W, int)
    for x in range(W):
        a = b[lo[x]:hi[x] + 1, x]
        out[x] = lo[x] + (int(np.argmax(a)) if a.size else 0)
    return median_filter(out, size=31)


def _band_sum(b, halfwin):
    """Vertical box-sum: out[y,x] = sum(b[y-halfwin .. y+halfwin, x]) (cumsum, vectorised).
    A 'sustained-brightness' transform: a lone bright pixel (vitreous speckle) contributes ~1x,
    a real layer (the ~15-20px RPE/BM complex) contributes ~Nx -- so argmax/threshold on the
    box-sum tracks sustained bright bands and ignores speckle. Used to anchor the complex (deep)
    and to gate the ILM onset (so a single speckle pixel can't pass as the inner-retina surface)."""
    b = np.asarray(b, np.float32)
    H, W = b.shape
    c = np.cumsum(np.vstack([np.zeros((1, W), np.float32), b]), axis=0)   # (H+1, W)
    rows = np.arange(H)
    lo = np.clip(rows - halfwin, 0, H)
    hi = np.clip(rows + halfwin + 1, 0, H)
    return c[hi] - c[lo]


def _banded_path(grad, center, K=80, depth_bias=0.0):
    """Min-cost L->R path through a K-tall band centred per-column on `center`.

    Vectorised graph build over a rectified K x W grid; scipy Dijkstra. Returns row per column.
    depth_bias>0 lowers the cost of deeper rows, so among competing edges the path prefers the
    deeper one (used for BM, to avoid riding the shallower ellipsoid-zone edge above the RPE).
    """
    H, W = grad.shape
    half = K // 2
    c = np.clip(np.round(center), half, H - 1 - half).astype(int)
    ks = np.arange(K)
    rows_idx = c[None, :] - half + ks[:, None]               # K x W absolute rows
    G = grad[rows_idx, np.arange(W)[None, :]]                # rectified gradient band
    N = K * W
    KK, XX = np.meshgrid(ks, np.arange(W - 1), indexing="ij")  # K x (W-1)
    S, D, Wt = [], [], []
    for dr in (-1, 0, 1):
        k2 = KK + dr
        valid = ((k2 >= 0) & (k2 < K)).ravel()
        src = (KK * W + XX).ravel()[valid]
        dst = (k2 * W + (XX + 1)).ravel()[valid]
        depth = depth_bias * (KK + np.clip(k2, 0, K - 1)) / (2.0 * (K - 1))
        w = (2.0 - G[KK, XX] - G[np.clip(k2, 0, K - 1), XX + 1] + EPS - depth).ravel()[valid]
        S.append(src); D.append(dst); Wt.append(w)
    START, END = N, N + 1
    k_all = np.arange(K)
    S.append(np.full(K, START)); D.append(k_all * W + 0); Wt.append(np.full(K, EPS))
    S.append(k_all * W + (W - 1)); D.append(np.full(K, END)); Wt.append(np.full(K, EPS))
    graph = csr_matrix((np.concatenate(Wt), (np.concatenate(S), np.concatenate(D))),
                       shape=(N + 2, N + 2))
    _, pred = dijkstra(graph, indices=START, return_predecessors=True)
    path = c.copy()
    node = END
    while node != START and node >= 0:
        p = pred[node]
        if node < N:
            x, k = node % W, node // W
            path[x] = c[x] - half + k
        node = p
    return path


def _deepest_strong(bs, search, rel=0.5):
    """Per column, the DEEPEST row in the search window whose box-sum exceeds rel*column-max-box-sum.
    Picks the RPE/BM complex over the shallower (also-bright) inner-retina/NFL without an ILM seed,
    while the box-sum (sustained brightness) keeps sparse vitreous speckle below threshold."""
    H, W = bs.shape
    lo = int(round(search[0] * H))
    hi = max(lo + 1, int(round(search[1] * H)))
    win = bs[lo:hi]
    strong = win >= (rel * win.max(0)[None, :])
    rows = np.arange(win.shape[0])[:, None]
    seed = lo + np.where(strong, rows, -1).max(0)        # deepest strong row per column
    seed[seed < lo] = lo + int(np.argmax(win.mean(1)))   # fallback: global brightest row
    return seed.astype(float)


def _complex_anchor(b, search=(0.05, 0.99), halfwin=8, K=90):
    """Robust RPE/BM-complex centre row per column, found WITHOUT an ILM seed (so a vitreous-speckle
    ILM latch can't drag it shallow). Seed = the deepest *sustained* bright band (box-sum, speckle-
    proof) per column, median-smoothed; then refine to the continuous brightest band exactly like the
    legacy RPE step (intensity graph, no depth bias) so good eyes are unchanged. Returns center[W]."""
    bs = _band_sum(b, halfwin)
    seed = median_filter(_deepest_strong(bs, search), size=41)
    rpe = _banded_path(b / (b.max() + 1e-6), seed, K=K)   # continuous brightest = RPE/BM complex
    return median_filter(rpe, size=31).astype(float)


def _ilm_from_complex(b, center, gap=(30, 400), min_run=6, frac=0.35):
    """ILM derived RELATIVE to the complex anchor, not top-down from the absolute top: per column,
    the first *sustained* bright onset within [center-gap_hi, center-gap_lo]. 'Sustained' = the row
    sits inside a run of >= min_run bright rows, so a single bright vitreous-speckle pixel cannot
    qualify (the dominant self-seg failure). Seeds the existing dark->bright edge graph."""
    H, W = b.shape
    bright = (b > frac * b.max(0)[None, :]).astype(np.float32)
    run = _band_sum(bright, min_run // 2) >= (min_run - 1)        # sustained bright (run-length)
    ceil_ = np.clip(center - gap[0], 0, H - 1).astype(int)
    floor_ = np.clip(center - gap[1], 0, H - 1).astype(int)
    seed = np.empty(W)
    for x in range(W):
        a, c = floor_[x], ceil_[x]
        col = run[a:c, x]
        seed[x] = (a + int(np.argmax(col))) if (c > a and col.any()) else (center[x] - 60)
    seed = median_filter(seed, size=41)
    return _banded_path(_edge(b, 1), seed, K=50)                  # snap to the real ILM edge


def _thickness_clamp(ilm, bm, min_px=12, max_px=150):
    """Enforce a plausible per-column ILM->BM retinal thickness (the projection's normalization runs
    ILM..BM+slab, so a stray shallow ILM would integrate vitreous). Implausible columns snap ILM to
    BM minus the column-median thickness."""
    thick = bm - ilm
    fin = thick[np.isfinite(thick)]
    med = float(np.clip(np.median(fin) if fin.size else 60.0, min_px, max_px))
    bad = ~np.isfinite(thick) | (thick < min_px) | (thick > max_px)
    out = ilm.copy()
    out[bad] = bm[bad] - med
    return out


def _rpe_peak(b, center, half=14):
    """Brightest row per column within +/-half of the complex anchor = the RPE/BM-complex PEAK. Stable
    under GA (the residual thin BM line is still the local brightest band), unlike the bright->dark edge,
    so it anchors the BM search at the complex without itself diving into the hypertransmission below."""
    H, W = b.shape
    out = np.empty(W)
    for x in range(W):
        c = int(np.clip(center[x], half, H - 1 - half))
        seg = b[c - half:c + half + 1, x]
        out[x] = c - half + int(np.argmax(seg))
    return out


def _robust_poly_baseline(y, deg=3, k=2.5, n_iter=16):
    """The smooth macular eye-wall trend BM would hold WITHOUT lesions: a robust low-order polynomial
    fit (iteratively reweighted least squares, Tukey biweight). Globally stiff — it rejects a dive/ride
    of ANY width as a departure from the trend (a morphological structuring element can only reroute an
    excursion narrower than itself, which left wide lesions — 008/016 — half-corrected), while a deg-3
    poly still bends to the real gentle bowl + tilt of the macula. Returns (baseline[W], inlier w[W])."""
    n = len(y)
    x = np.linspace(-1.0, 1.0, n)
    w = np.ones(n)
    z = np.full(n, float(np.median(y)))
    for _ in range(n_iter):
        coef = np.polyfit(x, y, deg, w=np.sqrt(np.clip(w, 1e-6, 1.0)))
        z = np.polyval(coef, x)
        r = y - z
        s = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-6
        u = np.clip(r / (k * s), -1.0, 1.0)
        w = (1.0 - u * u) ** 2                                       # Tukey biweight: 0 beyond k*MAD
    return z, w


def _whittaker(y, w, lam=1200.0, floor=0.02):
    """Weighted penalised least squares with a 2nd-difference roughness penalty:
        min_z  sum_x w_x (z_x - y_x)^2 + lam * sum_x (z_{x-1} - 2 z_x + z_{x+1})^2.
    Used here to lightly smooth the reroute blend (uniform weights); the banded sparse solve is fast."""
    n = len(y)
    w = np.clip(np.asarray(w, float), floor, 1.0)
    D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n)).tocsc()
    A = (diags(w) + lam * (D.T @ D)).tocsc()
    return spsolve(A, w * np.asarray(y, float))


def _bm_adaptive_core(b, center, lam=1500.0, offset=10.0, dr=12.0):
    """BM as a smooth membrane riding just under the RPE, that REROUTES across lesions (the production
    self-seg core; denoised `b` + complex `center` passed in so callers reuse them). The classical
    bright->dark edge marks the bottom of the RPE/BM complex, but it FOLLOWS pathology the wrong way:
    under GA / a bright choroid there is no dark floor, so it DIVES into the bright sub-RPE tissue
    (007/016 — BM ends up tens of px below the RPE, in the choroid). So the candidate is anchored to the
    **RPE peak + a small offset** (`_rpe_peak`, the bright complex; survives GA and a bright choroid) so
    BM rides just under the RPE and CANNOT dive -- EXCEPT under a drusen/PED dome, where BM must sit deep
    at the druse base, not up under the elevated RPE. Drusen vs a (concave) choroid bowl are told apart by
    the SIGN of the RPE deformation: a drusen/PED is the RPE pushed UP = a local convex bump = a black
    top-hat on a smoothed `peak` (`grey_closing(peak)-peak` > 0); a 016-style bowl is the RPE dipping
    DOWN (concave) -> top-hat ~0. So `candidate = w_dr*edge + (1-w_dr)*(peak+offset)` (deep edge under
    drusen, just-under-RPE elsewhere). Then the smooth eye-wall trend is fit robustly
    (`_robust_poly_baseline` = IRLS low-order polynomial — globally stiff, rejects a dive/ride/spike of
    ANY width) and BM is REROUTED across the divergences: `BM = w*candidate + (1-w)*trend`, lightly
    smoothed (`_whittaker`). Pure OCT segmentation; area-immune regardless. Validated one B-scan at a time:
    GA dives (007), bright-choroid dive (016 — now rides under the RPE), drusen/PED domes (017/011 — flat
    base), anchor spikes (003), large-GA undulation (008), normal bowl (005)."""
    W = b.shape[1]
    grad = _edge(b, -1)
    peak = _rpe_peak(b, center)
    edge = _banded_path(grad, peak + 22, K=70, depth_bias=0.2)        # bottom of complex (druse base / dives)
    se = max(31, (int(round(W * 0.43)) | 1))                          # bump-detect window > any drusen/PED
    peak_sm = median_filter(peak, 15)                                # de-noise so the top-hat fires on real bumps
    druse = grey_closing(peak_sm, size=se, mode="nearest") - peak_sm  # >0 where the RPE is pushed UP (drusen/PED)
    w_dr = np.clip(druse / dr, 0.0, 1.0)
    candidate = w_dr * edge + (1.0 - w_dr) * (peak + offset)          # deep base under drusen; just-under-RPE else
    trend, w = _robust_poly_baseline(candidate)                      # smooth eye-wall trend; w~0 where it diverges
    blend = w * candidate + (1.0 - w) * trend                        # reroute across dives/spikes/residual domes
    return _whittaker(blend, np.ones_like(blend), lam=lam)


def segment_surfaces(bscan):
    """One robust pass over a single B-scan -> (ilm[W], bm[W]) as float32. Anchor the RPE/BM complex
    FIRST (deep, sustained-brightness; speckle-proof), derive ILM relative to it (sustained onset), then
    BM context-adaptively (bright->dark edge where the choroid below is dark; clamped to the bright RPE/BM
    line under GA hypertransmission so it can't dive into the choroid). Validated strictly better than the
    old edge-only BM vs device (cohort median 15.3->10.7 um, 24/24 eyes improved, none regressed)."""
    b = _denoise(_norm(bscan))
    center = _complex_anchor(b)
    ilm = _ilm_from_complex(b, center)
    bm = _bm_adaptive_core(b, center)                              # dive-resistant BM (see _bm_adaptive_core)
    ilm = _thickness_clamp(ilm, bm)
    return ilm.astype(np.float32), bm.astype(np.float32)


def segment_bm(bscan):
    """BM row per A-scan for a single B-scan (H x W). Returns float array length W.
    Uses the DL model when opted in (env OCT_BM_DL + a model is available; see src/bm_dl.py), else the
    classical surfaces. Any DL failure falls back silently."""
    try:
        import bm_dl
        if bm_dl.active():
            return bm_dl.segment_bm(bscan).astype(np.float32)
    except Exception:
        pass
    return segment_surfaces(bscan)[1]


def _robust_slow_axis(surf, k_smooth=9, mad_k=4.0, max_jump=10.0):
    """Reject across-slow-axis outliers a small median window misses -- multi-B-scan latches AND
    isolated up-spikes -- and replace them by per-column interpolation along the slow axis. A median-9
    baseline survives a run of up to ~4 bad B-scans (vs the old kernel-3 median); residuals beyond
    mad_k*MAD or max_jump px are flagged (the fill_bm pattern, transposed to the slow axis)."""
    surf = np.asarray(surf, float)
    n, W = surf.shape
    base = median_filter(surf, size=(k_smooth, 1))
    resid = surf - base
    mad = np.median(np.abs(resid - np.median(resid, axis=0)), axis=0) * 1.4826 + 1e-6
    bad = (np.abs(resid) > mad_k * mad[None, :]) | (np.abs(resid) > max_jump)
    out = surf.copy()
    for x in range(W):
        m = ~bad[:, x]
        if m.sum() >= max(3, int(0.3 * n)):
            out[~m, x] = np.interp(np.flatnonzero(~m), np.flatnonzero(m), out[m, x])
        else:
            out[:, x] = base[:, x]
    return out


def _finish_surface(surf):
    """Slow-axis robust-outlier rejection + the spike-kill median + gentle bowl smoothing."""
    surf = _robust_slow_axis(surf)
    surf = median_filter(surf, size=(3, 41))       # kill spikes (across slow axis + along B-scan)
    return gaussian_filter(surf, sigma=(1.5, 3.0))  # gentle smoothing; the surfaces are smooth bowls


def _interp_invalid_cols(surf, invalid):
    """Replace each row's invalid (machine-fill) A-scans with a linear interpolation from its valid
    neighbours. Under a saturated 'white band' the B-scan has no vertical gradient, so the graph search
    has no signal there -- the surface must ride ACROSS the band, not dive. `invalid` is (n,W) bool."""
    out = np.asarray(surf, float).copy()
    for i in range(out.shape[0]):
        bad = np.asarray(invalid[i], bool)
        good = ~bad
        if bad.any() and good.sum() > 5:
            xs = np.arange(out.shape[1])
            out[i, bad] = np.interp(xs[bad], xs[good], out[i, good])
    return out


def segment_surfaces_volume(volume, invalid=None):
    """One robust pass over a whole volume -> (ilm[n,W], bm[n,W]). ILM is slow-axis robust-regularised.
    BM is taken PER-B-SCAN (each B-scan's own dive-resistant confidence-weighted surface) with NO slow-axis
    blend: the per-B-scan BM is already a smooth membrane (the Whittaker handles along-width continuity and
    interpolates GA gaps), so the volume just stacks it -- keeping 'Re-segment All' == 'Re-segment this
    B-scan' (a single B-scan can't be slow-axis smoothed, so matching it avoids a margin mismatch).

    `invalid` (n,W) bool: machine-fill / out-of-field A-scans (saturated 'white band'). The surfaces are
    interpolated ACROSS those columns from valid neighbours instead of being read off a zero-gradient band."""
    vol = np.asarray(volume, float)
    surf = [segment_surfaces(vol[i]) for i in range(len(vol))]
    ilm = _finish_surface(np.array([s[0] for s in surf], float))
    bm = np.array([s[1] for s in surf], np.float32)        # per-B-scan; no slow-axis averaging
    if invalid is not None:
        ilm = _interp_invalid_cols(ilm, invalid)
        bm = _interp_invalid_cols(bm, invalid).astype(np.float32)
    return ilm, bm


def segment_volume(volume, invalid=None):
    """BM surface bm[n_bscans, W] for a whole volume. Uses the DL model when opted in (env OCT_BM_DL +
    a model is available; see src/bm_dl.py), else the classical per-B-scan surfaces. `invalid` (n,W) bool:
    machine-fill columns the surface is interpolated across (see segment_surfaces_volume)."""
    try:
        import bm_dl
        if bm_dl.active():
            bm = bm_dl.segment_volume(volume).astype(np.float32)
            return _interp_invalid_cols(bm, invalid).astype(np.float32) if invalid is not None else bm
    except Exception:
        pass
    return segment_surfaces_volume(volume, invalid=invalid)[1]


# --------------------------------------------------------------------------- on-demand re-segmentation
def _deglitch_bm(bm, sigma=18.0, n_iter=6):
    """Robustly SMOOTH a BM row. BM is a gentle macular bowl with no high-frequency real features (drusen/
    PED elevate the RPE, not BM), so a robust low-pass — iteratively pull residual outliers toward the
    smooth trend, then smooth — removes residual jagged wander while preserving the real curve. Retained as
    a utility (validation scripts); the default BM now smooths inside _bm_adaptive_core via _whittaker."""
    bm = np.asarray(bm, float)
    z = bm.copy()
    for _ in range(n_iter):
        sm = gaussian_filter1d(z, sigma, mode="nearest")
        r = bm - sm
        s = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-6
        w = np.clip(1.0 - (np.abs(r) - 1.0 * s) / (3.0 * s), 0.0, 1.0)   # robust inlier weight (0 at >4*MAD)
        z = w * bm + (1.0 - w) * sm
    return gaussian_filter1d(z, sigma * 0.4, mode="nearest")


def _bm_adaptive(bscan):
    """Dive-resistant BM for one B-scan (the default per-B-scan BM) = denoise + anchor +
    _bm_adaptive_core (confidence-weighted smooth; no separate de-glitch needed)."""
    b = _denoise(_norm(bscan))
    return _bm_adaptive_core(b, _complex_anchor(b)).astype(np.float32)


def resegment_bm(bscan, invalid_row=None):
    """Re-segment BM for ONE B-scan. Re-runs the (now default) GA-robust context-adaptive method fresh,
    so the reader's 'Re-segment BM' button overrides a device layer / a stale cached surface / a prior
    edit on that B-scan. `invalid_row` (W,) bool: machine-fill cols the surface is interpolated across."""
    bm = _bm_adaptive(bscan)
    if invalid_row is not None:
        bm = _interp_invalid_cols(bm[None], np.asarray(invalid_row, bool)[None])[0]
    return bm


def resegment_bm_volume(volume, invalid=None):
    """Re-segment BM for a whole volume (the reader's 'Re-segment BM > All'). Same computation as
    segment_volume now (per-B-scan context-adaptive BM + light slow-axis smooth) — re-run fresh to
    override device/cached/edited BM across the volume. `invalid` (n,W) bool: machine-fill cols the
    surface is interpolated across."""
    return segment_volume(volume, invalid=invalid)
