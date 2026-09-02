"""WHY is prom high inside real GA? Decompose rpe_surface's prom = peak / inner-ref on GA-called columns
of the gold eye 005 OD: where does the peak land (distance above BM), and what is the inner-retina ref
level? If the inner ref COLLAPSES in GA (retina thinned/atrophic) faster than the residual outer signal,
the ratio stays >1 even with no RPE -> a structural false 'present'. Read-only, standalone."""
import os, sys
os.environ.setdefault("OCT_BM_DL", "1")
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import paths, bm_dl
import m3_projections as mp
from reader.core import e2e_source, oac_ga
from viewer.core import ga_native
import veto_rpe_prom as V

AX = mp.AX

for subj, eye, role in [("NHAMD-003-005-V3", "OD", "GOLD-GA"),
                        ("NHAMD-003-016-V2", "OD", "FP")]:
    path, plex = V.e2e_path(subj, eye)
    raw = e2e_source.open_e2e(path)
    ov = e2e_source.load_volume(raw, e2e_source.default_volume_index(raw, eye))
    vol = ov.vol
    bm = bm_dl.segment_volume(vol)
    row, prom = mp.rpe_surface(vol, bm)
    P = oac_ga.prep(ov, bm, baseline="radial2"); mask, _ = oac_ga.footprint(P, 0.5)
    ga_nat = ga_native.enface_to_native(mask, ov.fov_mm, ov.n_bscans, ov.W).astype(bool)

    # recompute the components of prom exactly as rpe_surface does, to read peak loc + ref on GA cols
    from scipy.ndimage import gaussian_filter1d
    n, H, W = vol.shape
    search_um, near_um, ref_lo_um, ref_hi_um, smooth = 110.0, 3.0, 220.0, 120.0, 2.0
    peak_above_bm_um = np.full((n, W), np.nan, np.float32)
    pv_arr = np.full((n, W), np.nan, np.float32)
    ref_arr = np.full((n, W), np.nan, np.float32)
    for i in range(n):
        c = gaussian_filter1d(vol[i], smooth, axis=0)
        for x in range(W):
            if not ga_nat[i, x]:
                continue
            bx = bm[i, x]
            a = int(np.clip(bx - search_um / AX, 0, H - 1)); z = int(np.clip(bx - near_um / AX, 1, H))
            if z <= a:
                continue
            seg = c[a:z, x]; k = a + int(np.argmax(seg)); pv = c[max(0, k-1):k+2, x].mean()
            ra = int(np.clip(bx - ref_lo_um / AX, 0, H - 1)); rz = int(np.clip(bx - ref_hi_um / AX, 1, H))
            ref = c[ra:rz, x].mean() if rz > ra else seg.mean()
            peak_above_bm_um[i, x] = (bx - k) * AX
            pv_arr[i, x] = pv; ref_arr[i, x] = ref
    m = ga_nat & np.isfinite(peak_above_bm_um)
    pk = peak_above_bm_um[m]; pv = pv_arr[m]; rf = ref_arr[m]; pr = prom[m]
    print(f"\n### {subj} {eye} [{role}] PLEX={plex:.2f}  GA cols n={m.sum()}")
    print(f"  peak distance above BM (um): p25={np.percentile(pk,25):.0f} p50={np.percentile(pk,50):.0f} "
          f"p75={np.percentile(pk,75):.0f}   (RPE/EZ normally ~20-50um above BM)")
    print(f"  peak value pv : p50={np.percentile(pv,50):.1f}")
    print(f"  inner ref     : p50={np.percentile(rf,50):.1f}")
    print(f"  prom=pv/ref   : p50={np.percentile(pr,50):.2f}")
    # how often is the 'peak' essentially AT BM (within ~12um) = residual sub-RPE / BM material, not RPE?
    nearbm = (pk <= 12).mean()
    print(f"  frac of GA-col peaks within 12um of BM (likely BM/residual, NOT a real RPE band): {nearbm*100:.0f}%")
    # is the high prom because ref is LOW (thinned retina) rather than peak being high?
    # compare ref on GA cols to ref on healthy core cols
    core = P["core"]
    # build healthy native-ish: easier to just report median pv & ref scale
    print(f"  peak/ref both attenuated? pv p50={np.percentile(pv,50):.1f}, ref p50={np.percentile(rf,50):.1f}")
