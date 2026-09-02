"""Diagnostics for the RPE-presence veto: (1) verify the native<->enface round-trip is faithful (so the
sweep's areas are trustworthy), (2) characterise prom INSIDE real GA vs the FP to explain the non-
separability, (3) compare prom over GA-called columns vs prom over the whole field (is the GA-called set
even where prom is low?). Read-only; standalone."""
import os, sys, csv
os.environ.setdefault("OCT_BM_DL", "1")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import paths, bm_dl
import m3_projections as mp
from reader.core import e2e_source, oac_ga, projection as proj
from viewer.core import ga_native
import veto_rpe_prom as V


def load(subject, eye):
    path, plex = V.e2e_path(subject, eye)
    raw = e2e_source.open_e2e(path)
    ov = e2e_source.load_volume(raw, e2e_source.default_volume_index(raw, eye))
    bm = bm_dl.segment_volume(ov.vol)
    row, prom = mp.rpe_surface(ov.vol, bm)
    P = oac_ga.prep(ov, bm, baseline="radial2")
    mask, area0 = oac_ga.footprint(P, 0.5)
    ga_nat = ga_native.enface_to_native(mask, ov.fov_mm, ov.n_bscans, ov.W).astype(bool)
    return ov, bm, prom, P, mask, area0, ga_nat, plex


def roundtrip_check(ov, mask):
    # forward to native then back to enface; how much area survives the round trip?
    nat = ga_native.enface_to_native(mask, ov.fov_mm, ov.n_bscans, ov.W).astype(bool)
    back = V.native_to_enface_mask(nat, ov.fov_mm, mask.shape) & mask
    a0 = float(mask.sum()) * V.MMPP2
    a1 = float(back.sum()) * V.MMPP2
    return a0, a1, a1 / (a0 + 1e-9)


for subj, eye, role in [("NHAMD-003-016-V2", "OD", "FP"),
                        ("NHAMD-003-005-V3", "OD", "GOLD"),
                        ("NHAMD-003-008-V1", "OD", "LARGE")]:
    ov, bm, prom, P, mask, area0, ga_nat, plex = load(subj, eye)
    a0, a1, frac = roundtrip_check(ov, mask)
    promG = prom[ga_nat]
    # prom over the whole in-field core vs over GA-called cols
    core = P["core"]
    print(f"\n### {subj} {eye} [{role}] PLEX={plex:.2f} area0={area0:.3f}")
    print(f"  round-trip area: {a0:.3f} -> {a1:.3f} (kept {frac*100:.1f}%)")
    print(f"  prom over GA-called cols: min={promG.min():.2f} "
          f"p5={np.percentile(promG,5):.2f} p50={np.percentile(promG,50):.2f} "
          f"p95={np.percentile(promG,95):.2f} max={promG.max():.2f}")
    # how many GA-called cols have prom <= 1.5 ('RPE gone') vs > 1.5 ('present')?
    for T in (1.5, 1.9):
        f_present = float((promG > T).mean())
        print(f"  frac of GA-called cols with prom>{T} ('present'): {f_present*100:.1f}%")
    # baseline-relative: is the OAC RPE-loss actually firing where prom is HIGH? correlate per-col.
    # bring rpe_nat (native OAC RPE-loss) and prom onto the same native frame and look at GA cols.
    rpe_nat = P["rpe_nat"]
    # rpe_nat is destriped OAC mean in the RPE band (LOW=GA). over GA-called cols:
    lossG = rpe_nat[ga_nat]
    g_base = P["g_base"]
    print(f"  OAC RPE-loss over GA cols (LOW=GA): p50={np.percentile(lossG,50):.3f} "
          f"vs g_base={g_base:.3f} (frac/g_base p50={np.percentile(lossG,50)/g_base:.2f})")
    # joint: of GA cols, how many are BOTH low-OAC (real RPE loss signal) AND high-prom (RPE present)?
    lowoac = lossG < 0.5 * g_base
    highprom = promG > 1.7
    print(f"  GA cols: low-OAC={lowoac.mean()*100:.0f}%  high-prom={highprom.mean()*100:.0f}%  "
          f"BOTH(low-OAC & high-prom)={ (lowoac & highprom).mean()*100:.0f}%")
