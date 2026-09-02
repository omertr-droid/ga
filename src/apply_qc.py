#!/usr/bin/env python
"""Apply QC decisions to the Spectralis<->GA pairing manifest.

Joins spectralis_validation.csv into spectralis_ga_pairing.csv and derives, per eye, a
qc_status the pipeline can filter on. Rules (in priority order):
  exclude_identity  - E2E's embedded patient number != folder (id_number_match == False).
                      003-013 is the confirmed case: the E2E is vessel-matched to patient 012,
                      so it is cross-patient vs the GA-013 annotation.
  exclude_timepoint - right patient but E2E acq date != GA visit date (date_match == False with a
                      real GA date). 003-007: Spectralis ~7 months before its GA V3 -> invalid
                      cross-sectional pairing for a growing lesion.
  ok                - passes identity + visit-date + 6x6 coverage.
Run: oct_env\\Scripts\\python.exe apply_qc.py
"""
import csv
import os
from datetime import datetime

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
VALID = os.path.join(RESULTS_DIR, "spectralis_validation.csv")

# Clinical / scope exclusions that CANNOT be derived from metadata: eyes outside the GA (dry-AMD) target.
# Keyed by subject (whole patient). 003-017: wet (neovascular) AMD -- the macular elevation is exudative
# /PED, NOT geographic atrophy (advRPE GA ~0 on both eyes); the OAC pipeline over-calls the dome, so it is
# not a valid GA cohort eye for training or validation.
SCOPE_EXCLUDE = {
    "NHAMD-003-017-V3": "wet (neovascular) AMD - exudative/PED elevation, not geographic atrophy; out of scope",
}


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_date(s):
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def main():
    pairing = load(PAIRING)
    valid = {(r["subject"], r["eye"]): r for r in load(VALID)}

    new_cols = ["qc_status", "qc_reason"]
    base_cols = [c for c in pairing[0].keys() if c not in new_cols]
    counts = {}
    for r in pairing:
        v = valid.get((r["subject"], r["eye"]), {})
        id_ok = v.get("id_number_match") == "True"
        date_ok = v.get("date_match") == "True"
        covers = v.get("covers_6x6") == "True"
        ga_date = parse_date(v.get("ga_visit_date", ""))
        acq = parse_date(v.get("e2e_acq_date", ""))

        if r["subject"] in SCOPE_EXCLUDE:
            status = "exclude_scope"
            reason = SCOPE_EXCLUDE[r["subject"]]
        elif not id_ok:
            status = "exclude_identity"
            reason = (f"E2E patient_id={v.get('e2e_patient_id','?')} != folder "
                      f"#{v.get('folder_pnum','?')}; vessel-matched to that other patient -> "
                      f"cross-patient vs GA annotation")
        elif ga_date is not None and not date_ok:
            gap = abs((ga_date - acq).days) if acq else "?"
            status = "exclude_timepoint"
            reason = (f"Spectralis acq {v.get('e2e_acq_date')} vs GA visit "
                      f"{v.get('ga_visit_date')} (~{gap} d gap); different timepoint")
        elif not covers:
            status = "review_coverage"
            reason = f"largest volume {v.get('fov_H_mm')}x{v.get('fov_V_mm')} mm < 6x6"
        else:
            status = "ok"
            reason = ""
        r["qc_status"], r["qc_reason"] = status, reason
        counts[status] = counts.get(status, 0) + 1

    with open(PAIRING, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_cols + new_cols)
        w.writeheader()
        w.writerows(pairing)

    print(f"Updated {PAIRING}  ({len(pairing)} eye rows)")
    for k in ("ok", "review_coverage", "exclude_timepoint", "exclude_identity", "exclude_scope"):
        if k in counts:
            print(f"  {k:18}: {counts[k]}")
    print("\nExcluded / flagged rows:")
    for r in pairing:
        if r["qc_status"] != "ok":
            print(f"  {r['subject']} {r['eye']}: {r['qc_status']} - {r['qc_reason']}")


if __name__ == "__main__":
    main()
