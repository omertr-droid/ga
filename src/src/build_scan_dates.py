#!/usr/bin/env python
"""Per-eye acquisition dates for the doctor-viewer library: Spectralis OCT vs the PLEX (advRPE) cube.

Answers "were the OCT and its PLEX reference taken at the same visit?" for every eye the viewer ships.
That matters because any area disagreement between the two is only attributable to the measurement if
the two scans see the same retina on the same day -- otherwise part of the gap is real GA growth.

The two dates come from independent sources, deliberately:
  * Spectralis: `e2e_acq_date` in results/spectralis_validation.csv, read from the E2E file itself
    (oct-converter `acquisition_date`).
  * PLEX:       `date` in results/ga_cohort_manifest.csv, which is the advRPE GA_vals CSV's own
    `acquisitionDateTime` field (a real cube timestamp, not a visit label).

Heads-up on a naming trap: the `date` column of results/spectralis_ga_pairing.csv -- and therefore
`acq_date` in the baked bundles and on the viewer's Library cards -- is the PLEX timestamp, NOT the
Spectralis one. This script reports both under unambiguous names.

Run (repo root):
  oct_env\\Scripts\\python.exe src\\build_scan_dates.py     -> results/viewer_scan_dates.csv
"""
import csv
import datetime as dt
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from paths import RESULTS_DIR  # noqa: E402
from viewer.core import bundle  # noqa: E402

OUT = os.path.join(RESULTS_DIR, "viewer_scan_dates.csv")


def _rows(name):
    with open(os.path.join(RESULTS_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _parse(s, fmts):
    s = (s or "").strip()
    for f in fmts:
        try:
            return dt.datetime.strptime(s, f)
        except ValueError:
            pass
    return None


def main():
    eyes = bundle.read_index()
    if not eyes:
        raise SystemExit("no baked library eyes — run src/bake_library.py first")

    val = {(r["subject"], r["eye"]): r for r in _rows("spectralis_validation.csv")}
    man = {(r["subject"], r["eye"]): r for r in _rows("ga_cohort_manifest.csv")}

    out = []
    for e in eyes:
        key = (e["subject"], e["eye"])
        v, m = val.get(key, {}), man.get(key, {})
        oct_dt = _parse(v.get("e2e_acq_date"), ("%Y-%m-%d", "%d-%b-%Y %H:%M:%S"))
        plex_dt = _parse(m.get("date"), ("%d-%b-%Y %H:%M:%S", "%Y-%m-%d"))

        gap = None
        if oct_dt and plex_dt:
            gap = (plex_dt.date() - oct_dt.date()).days
        out.append({
            "slug": e["slug"], "subject": e["subject"], "visit": e["visit"], "eye": e["eye"],
            "spectralis_acq_date": oct_dt.date().isoformat() if oct_dt else "",
            "plex_acq_datetime": (m.get("date") or "").strip(),
            "plex_acq_date": plex_dt.date().isoformat() if plex_dt else "",
            "days_plex_minus_oct": "" if gap is None else gap,
            "same_day": "" if gap is None else str(gap == 0),
            "our_ga_mm2": e.get("oac_area_dl_mm2", e.get("oac_area_mm2")),
            "plex_ga_mm2": e.get("plex_area_mm2"),
        })

    out.sort(key=lambda r: (r["subject"], r["eye"]))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    same = sum(1 for r in out if r["same_day"] == "True")
    unknown = sum(1 for r in out if r["same_day"] == "")
    print(f"wrote {OUT}")
    print(f"  {len(out)} eyes | same day: {same} | different day: {len(out) - same - unknown} | unknown: {unknown}")
    for r in out:
        if r["same_day"] != "True":
            print(f"    {r['slug']:24} OCT {r['spectralis_acq_date'] or '?':10} "
                  f"PLEX {r['plex_acq_date'] or '?':10} gap {r['days_plex_minus_oct']}")


if __name__ == "__main__":
    main()
