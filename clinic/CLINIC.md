# GA Clinic

A lean, standalone OCT **Geographic-Atrophy reader**. It is a third app in this repo, a sibling of
`reader/` (the heavy BM-validation / GA-studio workbench) and `viewer/` (the doctor viewer with a
card-grid library and a PLEX reference overlay). The clinic is deliberately smaller and built around a
**patient database** instead of a card grid, with **PLEX removed entirely**.

It reuses the project's GA pipeline as the single source of truth: every area comes from
`viewer.core.viewmodel.compute(..., baseline="radial2")` — the exact call the proven viewer upload path
uses — so a scan opened in the clinic reports the **same `oac_area_mm2`** as the viewer and the CLI.

---

## 1. Quick start

### Run from source (dev)
```
oct_env\Scripts\python.exe -m uvicorn clinic.api.app:app --host 127.0.0.1 --port 8021
```
Then open <http://127.0.0.1:8021/>. There is no build step — the frontend is plain ES modules + CSS
served straight off the FastAPI static mount; edit a file and refresh.

### Run the packaged app (end user, offline)
1. Unzip `oct_ga_clinic` onto a Windows x64 machine.
2. Double-click `run.bat`. The first launch can take 20–40 s while Windows scans the files; the browser
   opens automatically once the server is ready.
3. To stop: close the black console window. No Python, no pip, no internet; it binds `127.0.0.1` only.

---

## 2. Screens

1. **Database (landing).** A searchable list of patients (name + id). Type in the search box to filter
   by id or name (client-side). Click a patient → their detail screen. The header has **Upload E2E** and
   **Export Excel**.
2. **Patient detail.** That patient's visits in a table: **Date · Eye (OD/OS) · GA area (mm²) ·
   Δ vs previous (same eye) · BM source**, with **Open** (loads the scan into the viewer) and **Remove**
   (drops the row; optionally removes an orphaned clinic-staged copy, never an original). Δ growth is shown in red (the adverse
   direction), shrinkage in green. (There is no *Visit* column: an uploaded E2E carries no visit number —
   `pipeline.process` sets `visit=None` — so the column was always blank. The CSV keeps the field.)
3. **Viewer.** The 3-panel reader:
   - **Left** — IR localizer with the real per-B-scan locator lines (current B-scan highlighted green);
     click a line or scroll to navigate.
   - **Middle** — B-scan scroller with **show BM** and **highlight predicted GA** checkboxes, a slider,
     prev/next, mouse wheel, and arrow keys.
   - **Right** — the `f_trans` en-face projection with a **predicted-GA** (translucent green) legend
     toggle; click anywhere to jump the B-scan to that location.
   - A single dashboard card shows **GA area (OCT)**. There is no PLEX card and no PLEX overlay.
   - The top bar names the scan on screen — **patient · eye · acquisition date · field size**. It is set by
     `route()` only on the Viewer, and reset to "no scan loaded" on the Database and Patient screens, so it
     never claims a scan is loaded when none is being viewed.

---

## 3. Upload workflow

`Upload E2E` opens a multi-step modal — a small finite state machine:

```
[staging →] choosing file → listing scans → choosing scans → choosing BM → processing → opens the viewer
```

- **Pick the file — three ways.**
  - **Drag and drop** an `.E2E` anywhere on the window. A dashed-accent overlay appears while a file is
    over the page; the drop enters the FSM at `staging` (see below) and the rest of the flow is identical.
  - Click **Browse…** to open a file navigator (drive/Home chips, an **editable address bar** — paste a
    folder or file path and press Enter to jump there, and a folder list with the `.E2E` files).
  - Paste an absolute path directly.

  Browse and paste are server-side: the server lists its own filesystem, which is the only way a local web
  app can hand the pipeline a real absolute path and the only approach that also works in the offline
  package. A **drop** cannot do that — the browser exposes a file's bytes and its name, never its path
  (`File.path` is an Electron extension, not a web API) — so the bytes are **staged** first (§3.1) and the
  flow continues on the staged path.
- **Choose the scan(s).** Only **6×6-measurable** volumes are listed (one per eye) — the fields wide enough
  to measure a 6×6 mm GA area. **One or both eyes** may be selected; both are pre-selected by default, and
  the rows toggle (a `listbox`/`option` multi-select). Whether a scan actually contains GA, and its area,
  are known only AFTER it is processed; the chooser shows geometry (eye · line pattern · field size ·
  B-scans) and whether a **device Bruch's-membrane** is present.
- **Choose the Bruch's-membrane source.** Decided **per scan** — one block per selected eye, because the
  two eyes can differ (one may ship a device BM and the other not). This is load-bearing, not cosmetic:
  `_resolve_bm` treats `"device"` and `"auto"` identically (both keep `ov.bm`), so sending one shared
  `"device"` choice for an eye that has no device contour would silently record `bm_choice="device"` while
  actually running the classical self-seg. The table below applies to each eye independently:

  | device BM present | DL model available | Prompt | Recorded `bm_source` |
  |---|---|---|---|
  | yes | yes | "DL BM segmentation (recommended)" vs "Device BM" | `dl` or `device` |
  | yes | no  | "Using the device Bruch's-membrane" | `device` |
  | no  | yes | "This will be DL BM segmented" | `dl` |
  | no  | no  | "Automatically segmented (classical)" | `auto` |

  `auto` is the classical self-segmentation built into the pipeline (`reader.core` / `bm.py`). The
  packaged app ships the DL model, so the first two rows are the usual cases.

Processing runs **once per selected eye, sequentially** (each is CPU-bound; the E2E is decoded only once
because `store.open` caches the `RawE2E`), records each to the database, and opens the **first eye that
succeeded** — the other is already saved and one click away on the patient screen. If one eye fails the
other still opens, with a per-eye `OD ✓ · OS ✗` status and a warning toast. If the DL model is chosen but
fails, the pipeline falls back to the device/auto BM and surfaces a warning (the scan still opens).

### 3.1 Staging a dropped file (`clinic/core/staging.py`)

`POST /upload/stage` streams the request's **raw body** (`application/octet-stream`) to
`<clinic-data>/uploads/<sha256 of the content>.E2E`, then returns that path; the client runs the ordinary
`/upload/open` → `/upload/process` on it. Three deliberate choices:

- **Raw bytes, not multipart.** FastAPI's `UploadFile`/`File()`/`Form()` need `python-multipart`, which is
  not installed (Starlette asserts on it at request time). Streaming needs nothing extra and never buffers
  the ~300 MB file in memory. Measured: ~1.2 s for 244 MB on loopback.
- **Content-addressed name.** `reader.core.ids.e2e_id` derives the eid from the *path*, and `vid` →
  `record_id` derive from the eid. A random temp name would therefore mint a fresh identity on every drop
  and **append a duplicate patient row** for the same physical scan. Hashing the content makes the path a
  pure function of the bytes, so a re-drop is a no-op on disk and an **upsert** in the database.
- **Validate, then bless.** The first four bytes must be the E2E magic `CMDb` (rejected as soon as they
  arrive, so a wrong file never costs a 300 MB write); over `MAX_STAGE_BYTES` → 413; empty → 400. Bytes go
  to a temp sibling and are `os.replace`d into the canonical name only after the whole stream is written and
  checked, so a cancelled or crashed upload can never leave a truncated `.E2E`. Closing the modal
  (`✕`/Escape/scrim) aborts the transfer and the temp file is unlinked.

The `name` query parameter is echoed for display only and **never touches the filesystem**, so a hostile
filename cannot escape the uploads directory.

**Staged files persist for reopen.** For a dropped scan this copy is the only server-side original (the
browser never disclosed its source path). When a staged record is removed, the UI offers to delete that
copy; deletion is reference-counted against `e2e_path`, so a shared OD/OS file is removed only after its
last record is gone. Browsed/original E2Es are never deleted.

One accepted limitation: **Browse**-ing a file at its original path *and* dropping the same file yields two
rows for one physical scan, because the eid is path-derived. It's a duplicate visit row, not corruption, and
either can be deleted from the patient screen.

---

## 4. Architecture

```
clinic/
  core/                     pure domain logic (no web deps)
    __init__.py             imports reader.core (puts repo src/ on sys.path → bm, bm_dl, m2_bm by name)
    identity.py             patient_id / patient_name / acq_date from E2E metadata
    scan_list.py            list 6x6-measurable scans + cheap per-scan device-BM detection
    pipeline.py             open → load → BM choice → viewmodel.compute → LiveSource  (the GA choke point)
    staging.py              stream a dropped .E2E to uploads/<sha256>.E2E (content-addressed, atomic)
    store.py                bounded session: 1 RawE2E + 2 LiveSources; batch-end raw/model release
    db.py                   the standalone patient database (patients.csv); atomic write + lock handling
    xlsx.py                 build the .xlsx export (openpyxl)
  api/
    app.py                  FastAPI factory: same-origin API, no-cache middleware, static frontend
    deps.py                 one process-wide ClinicStore + get_store()
    routes.py               all endpoints
    schemas.py              pydantic request bodies
  web/                      vanilla ES-module SPA (index.html, css/app.css, js/*)
  data_store/               patients.csv lives here (created on first write; shipped empty)
    uploads/                dropped E2Es, named by content hash; persists so /reopen works. Never shipped.
  CLINIC.md                 this file
```

### Reuse (single source of truth)
The clinic imports — never re-implements — the pipeline:
- `reader.core.e2e_source` — `open_e2e`, `load_volume`, `_device_layers` (device-BM detection).
- `reader.core.layers.effective_surfaces` — the filled ILM/BM surfaces.
- `viewer.core.viewmodel.compute(raw, ov, ilm, bm, baseline="radial2")` — **the GA number** + all overlays.
- `viewer.core.bundle.LiveSource` — the in-memory ViewSource the panel endpoints serve (same code path
  the viewer uses for uploads), minus PLEX.
- `bm_dl` — the optional DL Bruch's-membrane model.

`pipeline.process` reproduces the viewer's upload call sequence exactly (`load_volume` →
`effective_surfaces(ov, None)` → optional `bm_dl.segment_volume` swap → `viewmodel.compute`), with the
same defaults and **no post-processing of the area** — that is what keeps the clinic number identical to
the viewer/CLI.

### Endpoints (all under `/api`)
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` · `/config` | liveness; `{title, dl_available, dl_default}` |
| POST | `/upload/stage` | drag-and-drop only: stream raw bytes → `{path, size, dup}` (content-addressed). No GA. |
| POST | `/upload/open` | decode an E2E, list its 6×6 scans (+ device-BM + DL availability). No GA. |
| POST | `/upload/process` | process `{path, index, bm_choice}` → `{vid, meta, db, warning}`; records the scan. Called once per selected eye. |
| POST | `/upload/finish` | release the decoded E2E + ONNX session after all selected eyes finish; live viewer results stay cached. |
| POST | `/reopen` | re-open a database row by `record_id` (cache hit, else re-process) |
| GET | `/db` | patient list for Home (`locked:true` instead of 500 if the CSV is held) |
| GET | `/db/{patient_id}` | one patient's visits |
| GET | `/db.xlsx` · `/db.csv` | export; `DELETE /db?record_id=&delete_staged=` drops a row and optionally an orphaned staged copy. |
| GET | `/scan/{vid}/meta` `…/bscan/{i}.png` `…/localizer.png` `…/projection.png` `…/ga_overlay.png` | viewer images |
| GET | `/scan/{vid}/loc_lines` `…/bm` `…/ga_native` `…/dashboard` | viewer JSON |

There is no `/plex` route — the OCT-only invariant.

---

## 5. Data & the patient database

Packaged folders keep all mutable state under top-level `user_data/` (`OCT_CLINIC_DATA`); source runs
default to `clinic/data_store/`. `patients.csv` has one row per scan/visit, upserted so re-loading the same
scan updates its row rather than duplicating it. Key columns (see `db.FIELDS` for the full list):

`record_id, patient_id, patient_name, eye, visit, acq_date, n_bscans, fov_w_mm, fov_h_mm, ga_area_mm2,
bm_source, volume_index, bm_choice, e2e_path, vid`.

- **Upsert key:** `record_id = sha1(patient_id|eye|acq_date|vid)[:12]`. The `vid` keeps two date-less
  E2Es from colliding.
- **`patient_name`, `volume_index`, `bm_choice`** are the clinic's additions over the viewer's registry:
  the name powers the searchable list, and the index + choice make a re-open deterministic (the
  recomputed area matches the stored one).
- **Windows file-lock safety (the explicit ask).** patients.csv is often open in Excel or mid-OneDrive
  sync, which holds the file. Writes are atomic (temp file + `os.replace`); a held file raises `DbLocked`
  and is left intact. On **upload** that becomes a **soft warning** (the scan still opens and is shown);
  on **read / delete / export** it becomes **HTTP 423** so the UI can say "close it and try again".

### Excel export
`Export Excel` (Home header) downloads `ga_clinic.xlsx` via `/api/db.xlsx` — an `openpyxl` workbook with
one sheet ("GA measurements"), one row per scan, sorted by patient then acquisition date, a bold frozen
header, and GA area as a real number. `openpyxl` (pure-python) is the only dependency added for the
clinic.

---

## 6. DL Bruch's-membrane model

The packaged app bundles `outputs/bm_dl/bm_unet.onnx` (+ its `bm_unet.onnx.meta.json` sidecar). `bm_dl.py`
finds it under `OUT_DIR/bm_dl` (which resolves to `app/outputs/bm_dl/` in the package). `run.bat` sets
`OCT_BM_DL=1` so the upload BM prompt pre-selects DL when available. Inference runs on CPU via
onnxruntime. The clinic's config/chooser uses `bm_dl.discoverable()` (no model allocation), inference uses
two B-scans per batch, and `/upload/finish` releases the session after the selected eyes complete.

### Memory profile (measured Windows package, 286 MB E2E)

- Startup: 133 MB RSS.
- After opening/listing: 345 MB RSS (only clinic-eligible volumes retained as float32).
- After DL + GA: 880 MB RSS; peak 1.40 GB.
- After batch release, with the eye still viewable: 210 MB RSS.

The same eye remained exactly 2.1202 mm² before/after the memory changes; a second historical eye remained
exactly 0.1000 mm². `src/smoke_clinic.py` is the reusable packaged real-E2E check.

Relevant env vars: `OCT_BM_DL` (default the prompt to DL), `OCT_BM_DL_MODEL` (explicit model path),
`OCT_BM_DL_SMOOTH_SIGMA` (within-B-scan column smoothing).

---

## 7. Packaging & release

`src/package_clinic.py` builds the offline Windows package under `dist/oct_ga_clinic/`:
embeddable CPython 3.11 + the oct_env site-packages (pruning verified dev/optional entries) + the app
code (`clinic/`, `viewer/core`, `reader/core`, `src/`) + the DL model + `run.bat`. `patients.csv` and
`data_store/uploads/` are excluded, so the package ships a clean database and none of the dev box's staged
E2Es. Mutable data goes to top-level `user_data/`. A second double-click reopens the running instance.
Drag-and-drop needs **no** new site-package (raw bytes, not multipart).

```
oct_env\Scripts\python.exe src\package_clinic.py            # full build
oct_env\Scripts\python.exe src\package_clinic.py --zip      # + dist\oct_ga_clinic.zip
oct_env\Scripts\python.exe src\package_clinic.py --refresh-app   # reuse the runtime, refresh app + run.bat
```

Set `OCT_CLINIC_DIST` to build outside OneDrive (avoids syncing the large tree).

**Prune-list note (verified, do not regress):** `matplotlib` and `h5py` are pulled in transitively by
`eyepy`, so they **must** ship — confirm any change to `PRUNE_PREFIXES` by launching the packaged app and
running a real upload, not just a static import check.

### macOS build (`src/package_clinic_mac.py`)
Run on the **Windows dev box** (it cross-vendors macOS wheels with `pip download --platform macosx_…`):
```
oct_env\Scripts\python.exe src\package_clinic_mac.py                       # Apple Silicon (arm64)
oct_env\Scripts\python.exe src\package_clinic_mac.py --arch x86_64-apple-darwin   # Intel
```
It ships a standalone CPython (python-build-standalone, kept as `runtime/python.tar.gz`) and the explicit
closed wheel set in `requirements-clinic.txt` (`--no-deps`, so unused oct-converter export dependencies do
not sneak back in; headless OpenCV), plus the
app + DL model, and a `run.command` launcher. The Mac owner unpacks the `.tar.gz`, clears quarantine
once (`xattr -dr com.apple.quarantine <folder>`), and double-clicks `run.command` (offline; first run
unpacks Python and atomically installs the wheels). The wheel-vendoring step fails loudly if a pin has no
macOS cp311 wheel — adjust that pin and re-run, and **verify the bundle on a real Mac before release**.
The app code itself is cross-platform (audited): `os.path` everywhere, atomic CSV writes, a POSIX-aware
file browser (`/`, `/Volumes`, home), and no Windows-only APIs.

---

## 8. Troubleshooting

- **First launch looks stuck.** Normal: the embeddable Python scans files for 20–40 s on first run; the
  browser opens once `/api/health` answers.
- **"file not found on this machine".** The upload path must be an absolute path on the machine running
  the server.
- **"The E2E / database is open or locked".** Close it in Excel / wait for OneDrive to finish, then retry.
- **No 6×6 scan in this E2E.** The file has no macular volume wide enough for a 6×6 GA measurement.
- **Port already in use.** Another process holds 8021; stop it or run uvicorn on a different `--port`.

---

## 9. Development notes
- No build step. ES modules + one CSS file; the `_no_cache_assets` middleware makes edit-and-refresh work.
- `clinic.core` has no web dependencies; FastAPI/pydantic live only in `clinic.api`.
- The GA computation has exactly one entry point (`pipeline.process` → `viewmodel.compute`). Keep it that
  way — never post-process the area — so the clinic, viewer, and CLI stay in agreement.
