"""Patient/scan identity read from an opened E2E (``reader.core.e2e_source.RawE2E``).

Two sources of truth in a Spectralis E2E, used together:
  * ``raw.md["patient_data"]`` (oct-converter ``read_all_metadata``) — a 1-element list of dicts with
    ``patient_id``, ``first_name``, ``surname`` (the only place the patient NAME lives).
  * ``raw.vols[index]`` (the decoded volume) — carries ``patient_id`` and ``acquisition_date`` per scan.

We prefer the per-volume ``patient_id`` (it is the one tied to the scan being processed) and fall back
to the metadata block; the display NAME always comes from the metadata block. This mirrors the
identity helper the reader already uses (``reader/api/routes_fs.py:_patient_of``) so the clinic shows
the same identity the rest of the project does.
"""


def patient_data(raw) -> dict:
    """The E2E's embedded patient identity ``{patient_id, first_name, surname}`` (only non-empty keys),
    best-effort — returns ``{}`` if the metadata block is missing or malformed."""
    pd = (getattr(raw, "md", None) or {}).get("patient_data")
    if isinstance(pd, (list, tuple)):
        pd = pd[0] if pd else None
    if not isinstance(pd, dict):
        return {}
    out = {}
    for k in ("patient_id", "first_name", "surname"):
        v = pd.get(k)
        if v is not None and str(v).strip():
            out[k] = str(v).strip()
    return out


def patient_name(raw) -> str:
    """``"<first_name> <surname>"`` from the metadata block, or ``""`` when no name is recorded
    (de-identified files often carry only an id; the UI then falls back to the id)."""
    pd = patient_data(raw)
    return " ".join(p for p in (pd.get("first_name"), pd.get("surname")) if p).strip()


def patient_id(raw, index=None) -> str:
    """The patient id for a scan: the per-volume id (when ``index`` given) else the metadata id."""
    if index is not None:
        pid = getattr(raw.vols[index], "patient_id", None)
        if pid is not None and str(pid).strip():
            return str(pid).strip()
    return patient_data(raw).get("patient_id", "") or ""


def acq_date(raw, index) -> str:
    """The Spectralis acquisition-date string for volume ``index`` (e.g. ``'03-Apr-2022 09:14:02'``),
    or ``""`` if absent. Kept as the device string; the DB enriches it to a sortable ISO date."""
    return str(getattr(raw.vols[index], "acquisition_date", "") or "")
