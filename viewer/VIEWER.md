# Doctor viewer (`viewer/`) — standalone, offline, read-only

A simple clinician-facing viewer derived from `reader/`, reusing `reader.core.*` so the GA/projection
math has one source of truth. **1 menu, 3 screens** (a Library, a Patients tracking table, and a 3-panel
Viewer), packaged as a folder the user **double-clicks (`run.bat`) — no Python, no internet, no commands**.

## What it shows
- **Library** — a card grid of the built-in scans. Each card is an **IR thumbnail** + the scan date and a
  **GA comparison**: two stat tiles, **Our GA (DL, green)** vs **PLEX GA (blue)**, with a Δ/agreement line
  (controls — PLEX 0 — honestly show our over-call in amber). "Our GA" is computed on the DL Bruch's-
  membrane, the same number the viewer opens on. Click a card to open it. **"Add to library"** = paste a
  local `.E2E` path to process a new scan on this machine.
- **Patients** — the tracking table. Every scan loaded on this machine is persisted to a CSV registry
  (`viewer/data_store/registry.csv`) and grouped here by patient, so GA can be followed over time. Per
  eye, scans are ordered by acquisition date with the **Δ vs the previous scan** (GA growth shown in red,
  the clinically adverse direction). **Open** re-loads a scan (re-processing the uploaded E2E from its
  stored path, same eye); **Remove** deletes the row (the E2E on disk is untouched); **Export CSV**
  downloads the log. See *Patient GA tracking* below.
- **LEFT — IR localizer** with **real per-B-scan locator lines** (drawn from the file's true angular
  endpoints `posX1/posY1→posX2/posY2`, mapped to the co-acquired IR by series id — *not* an evenly-spaced
  estimate). The current B-scan's line is highlighted green; click a line or scroll to navigate.
- **MIDDLE — B-scan scroller** + 2 checkboxes: show **BM**, and **highlight predicted-GA A-scans**
  (a translucent band over the sub-BM slab where the OAC detector called GA).
- **RIGHT — en-face projection** (`f_trans`) with our **predicted GA (translucent green)** and the
  registered **PLEX (advRPE) GA (blue outline)**. Each is a **legend show/hide checkbox — both OFF by
  default** (the legend keeps the colored swatches). Click anywhere on the en-face to jump the B-scan
  there (a thin, non-occluding edge tick marks the column). For eyes that have one, the **doctor's
  marked-up PLEX GA correction** image is shown *literally above* the projection (a reference, not
  registered/overlaid; `plex_correction.png` in the bundle, `meta.has_plex_correction`).
- **Dashboard:** **"GA area (OCT)"** beside **"PLEX reference"**. The patient·eye identity sits in the
  **top bar only** (not duplicated in the dashboard).

**OCT-only constraint:** our number comes only from `reader.core.oac_ga`; PLEX is a *reference overlay and
a separate dashboard figure* — never blended into our computed area. Uploaded scans have no PLEX.

## Patient GA tracking (`viewer/core/registry.py`)
Without this, an uploaded scan lived only in an in-memory LRU (max 4) and vanished on restart — a patient
could not be followed across scans. Now every successful upload is persisted to **one CSV row** in
`viewer/data_store/registry.csv`, keyed by a stable `record_id = sha1(patient_id|eye|acq_date)`, so
**re-loading the same scan upserts its row** (no duplicates; `logged_at` refreshes). Columns:
`record_id, logged_at, source, patient_id, eye, visit, acq_date, n_bscans, fov_w_mm, fov_h_mm,
ga_area_mm2, bm_source, status, note, e2e_path, vid`. The Patients tab reads it back, groups by patient,
and shows per-eye GA over time.

- **Robust to the "file is open" gotcha (the explicit ask).** Writes are **atomic** (a temp sibling +
  `os.replace`); if Excel/OneDrive holds `registry.csv`, `os.replace` raises `PermissionError`, mapped to
  `RegistryLocked` — **the existing file is left intact** (no truncation). On upload this becomes a *soft*
  warning (`registry: {saved:false, warning:…}` in the response → a toast) so the scan still opens; the
  Patients tab shows a dismissable banner + Retry; the reads/deletes return **HTTP 423**. The same
  classifier (`is_lock_error`) maps a locked **E2E** on upload to a 423 ("close it and retry").
- **Endpoints:** `GET /api/registry` (`{records, locked, csv_path}`; `locked:true` instead of 500 when
  held), `DELETE /api/registry?record_id=…` (row only; the E2E file is never deleted),
  `GET /api/registry.csv` (download). All localhost, single-user.
- **Packaging:** `src/package_app.py` copies the new `core`/`web` files but **excludes `registry.csv` +
  `.registry-*.tmp`**, so the offline package ships a clean log that the doctor's own uploads populate.
  The library-only macOS build has no upload path, so the Patients tab is hidden there.

## Architecture
- `viewer/core/` (no web deps): `locator.py` (real locator lines + series-id IR pairing),
  `ga_native.py` (en-face GA mask → native A-scan flags), `plex.py` (advRPE registration → blue outline),
  `viewmodel.py` (compute all 3 panels from an E2E + volume + BM), `bundle.py` (bake/serve + the
  `ViewSource` seam: `BundleSource` for library eyes, `LiveSource` for uploads), `registry.py` (the
  persistent patient-tracking CSV; atomic writes, file-lock handling — see *Patient GA tracking*).
- `viewer/api/`: `app.py` (FastAPI factory), `deps.py` (resolves a `vid` → ViewSource; records uploads to
  the registry), `routes_viewer.py` (panel endpoints + `GET/DELETE /api/registry`, `GET /api/registry.csv`).
- `viewer/web/`: `index.html` + `js/{main,library_view,registry_view,viewer_view,api,lib}.js` + `css/app.css`.
- `viewer/data_store/library/<subject>_<eye>/`: the baked bundles (meta.json + bundle.npz + PNGs) +
  `index.json` (identity + areas the cards read). Each bundle also carries a **cached DL-BM twin**
  (`bm_dl`/`ga_native_dl` in the npz + `projection_dl.png`/`ga_overlay_dl.png` + `oac_area_dl_mm2`), served
  as `lib:<slug>|dl`. `viewer/data_store/registry.csv`: the tracking log (created on first upload; never
  shipped in the package).

Library scans are served from the bundle with **numpy + cv2 only** (no E2E decode; arrays cached after
first read, so scrolling is ~ms) — including the DL twin, so the DL view (the default for library eyes)
needs no E2E, no model and no onnxruntime even in the offline package. Uploading a new E2E runs the full
pipeline (`oct_converter` + `bm` + `oac_ga`) — the only path that needs the heavy scientific stack.

**Shipped library = the `qc_status==ok` eyes that have a PLEX GA value** (25 as of 2026-06-24 — controls
kept; a PLEX 0.0 is a real measurement; NaN/missing and any non-`ok` qc_status are dropped, e.g. 007/009/
013/017 plus 002 OD + 015 OS marked `exclude_manual`). Selection is defined in `library_rows()` in
`src/bake_library.py` (qc_ok + non-NaN `advRPE_area_mm2`); there is no hand-picked exclude list — to drop
an eye, set its `qc_status` in the pairing CSV to anything but `ok`, then re-bake (or `--reindex` to prune).

## Run (dev)
```
oct_env\Scripts\python.exe -m uvicorn viewer.api.app:app --host 127.0.0.1 --port 8011
# open http://127.0.0.1:8011/
```

## Bake / curate the library (one-time, dev machine)
Bakes one bundle per library eye (qc_ok + has a PLEX GA value), using the validated/device
(`effective_surfaces`) BM for the base view **plus a cached DL-BM twin** (`bm_dl.segment_volume`). A full
run is authoritative — it **prunes** any bundle no longer in the set:
```
oct_env\Scripts\python.exe src\bake_library.py            # the whole library set -> viewer/data_store/library
oct_env\Scripts\python.exe src\bake_library.py --only 005 # one eye (no prune)
oct_env\Scripts\python.exe src\bake_library.py --qc       # + locator QC overlays in outputs/viewer_bake_qc/
oct_env\Scripts\python.exe src\bake_library.py --reindex  # prune to the current set + rebuild index.json (no E2E)
oct_env\Scripts\python.exe src\bake_library.py --attach-plex-corrections "<dir>"  # add doctor PLEX-GA corrections
```
The library set follows the master pairing CSV's `qc_status`: change QC there, then re-bake (or `--reindex`
if only the set, not the pixels, changed). DL caching needs `outputs/bm_dl/bm_unet.onnx` at bake time; if
absent, bundles bake without the DL twin and the viewer falls back to the validated default.

**Doctor PLEX-GA corrections:** `--attach-plex-corrections <dir>` copies marked-up correction images named
`<patient>_<eye>.png` (e.g. `001_OD.png`) into the matching bundles as `plex_correction.png` (no bake/E2E,
just patches existing bundles + meta). Only eyes already in the library are matched (patient number + eye →
slug, one visit/patient); an image with no library eye is **skipped (no scan added)**; library eyes with no
image are left unchanged. The viewer shows it above the projection. Restart the viewer / re-package to pick
it up.

## Package for offline use (one-time, dev machine with internet)
`src/package_app.py` builds **BOTH** the Windows and the macOS package by default (Windows by
`package_app.py`, macOS delegated to `src/package_app_mac.py`):
```
oct_env\Scripts\python.exe src\package_app.py             # build BOTH (download runtimes once)
oct_env\Scripts\python.exe src\package_app.py --zip       # + Windows oct_ga_viewer.zip (mac always tars)
oct_env\Scripts\python.exe src\package_app.py --only windows   # Windows only
oct_env\Scripts\python.exe src\package_app.py --only mac       # macOS only
oct_env\Scripts\python.exe src\package_app.py --refresh-app    # fast: reuse runtimes, refresh app+launcher
set OCT_VIEWER_DIST=C:\path-outside-OneDrive               # build target (recommended, avoids OneDrive sync)
```

**Windows** (`dist/oct_ga_viewer/`): embeddable Python 3.11 + the copied scientific stack (pruning
`imageio_ffmpeg`/`pydicom`, verified unused) + app code + the baked library + `run.bat`. The doctor
copies/unzips the folder and **double-clicks `run.bat`** — it clears `PYTHONHOME`/`PYTHONPATH` (so it
ignores any Python already on the machine), sets `MPLBACKEND=Agg`, launches uvicorn on `127.0.0.1:8011`,
and **opens the browser only once `/api/health` answers** (the first start can take 20–40 s while Windows
scans the files; opening the browser earlier is what looked "stuck"). Windows x64 only.

**macOS** (`dist/oct_ga_viewer_mac.tar.gz`, Apple Silicon by default): **library-only** — only
`numpy`+`opencv-python-headless`+`fastapi`+`uvicorn` are vendored as arm64 cp311 wheels (no upload/heavy
stack), so the bundle is small and robust. A standalone arm64 CPython 3.11 (python-build-standalone) is
shipped **as its `runtime/python.tar.gz`, unextracted** (Windows never touches its unix symlinks); the
Mac unpacks it and offline-`pip install`s the wheels into `libs/` on first launch (`--no-index
--find-links wheels`), then runs the same uvicorn app. The Mac owner: double-click the tar.gz, run
`xattr -dr com.apple.quarantine <folder>` once (Gatekeeper), then double-click `run.command`. The
launcher sets `OCT_VIEWER_LIBRARY_ONLY=1` → `GET /api/config` → the JS hides the "Add to library" row.
For an Intel Mac: `--arch x86_64-apple-darwin`. To pin/override the interpreter: `--python <url-or-tarball>`.

Note: don't re-run the packager while a `run.bat`/`run.command` server is open — the running server locks
the files. To re-zip the already-built Windows folder without rebuilding, zip it directly
(`shutil.make_archive` on `dist/oct_ga_viewer`).
