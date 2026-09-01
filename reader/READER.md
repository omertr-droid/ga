# oct_ga reader — full documentation

A local, modular **web app** to read a Heidelberg Spectralis **E2E**: scroll the **6×6 OCT volume**'s
B-scans HEYEX-style, overlay the **ILM/BM** layers and the projection **measurement bands**, watch a
**live en-face projection** under the B-scan, and **correct layers** by scrolling/dragging a segment up
or down with the projection updating in real time. It is the interactive workbench for getting the best
projection — and, later, for viewing and correcting **GA** segmentation.

It reuses the existing pipeline (`src/m2_bm`, `src/m3_projections`, `src/bm`, `src/qcviz`,
`src/register_qc`) in-process; it does **not** reimplement the science.

- **CURRENT FOCUS — BM-validation Library + workbench (built this session, backend-verified end-to-end).**
  A new **Library** tab is the landing page: the cohort's 6×6 eyes from
  `results/bm_worklist.csv` (precompute `src/build_bm_worklist.py`) with a red/orange/green progress pill;
  click a row → the **BM tab** opens that E2E+eye+6×6 as a streamlined **BM validation
  station** — **ILM dropped**, spline auto-on, **auto-persist (no Save)**, **validate-on-advance**
  ("Validate & next" / Space), a validated/pre-segmented/edited/device **filmstrip**, **per-image**
  Copy-BM-from-prev, Reset-to-device (this B-scan) + **Reset ALL to device** (whole eye → device, clears
  edits + validations), per-B-scan Re-segment. **No-device eyes auto-pre-segment with the DL model on open
  (cached, tagged `bm_src="model"` → purple "pre-segmented")** so there's no Segment click; device eyes
  seed from the device contour. The **topbar carries the patient·eye identity** on every tab. Validated flag →
  `bm_status.json` beside the corrections; new files `core/library.py`, `api/routes_library.py`,
  `web/js/library_view.js`; new routes `/api/library`, `…/corrections/copy_prev`, `…/corrections/all`,
  `…/corrections/presegment_bm`, `GET/PUT/DELETE …/bm_status`; `device_bm` + `corrected[].bm_src` in
  `/layers`. Goal: hand-validate device BM →
  a patient-split BM-surface training set → a **DL `segment_bm()`** (classical hand-tuning of `bm.py` was
  abandoned after it regressed volume-wide). Exporter + training not built yet. See §2.
- **Status:** MVP **+ Phase 2 done and verified end-to-end** (real E2Es, headless browser + live-server
  API round-trip). Done in Phase 2: **(B) hardened self-segmentation** (`src/bm.py`, validated by
  `src/m2_bm.py` — cohort central-6mm BM error 20.4→15.3 µm, worst-case 211→48 µm, no regressions);
  **(C) manual layer correction** (drag/scroll a segment, persisted as JSON sidecars, folded into the
  projection); and a **live projection panel** in the B-scan view with a **raw** (Spectralis-style slab)
  default + the cue features. NEW **B-scan GA annotation studio** (the Segment tab): per-B-scan **state**
  (todo/ga/ga_free/borderline/reviewed; `mask_store.py` status.json), B-scan **BM/RPE + slab + PLEX
  column-guide** cues, **status filmstrip** + GA-free/borderline/reviewed + keyboard + "Rest GA-free",
  **brush + SAM2-box assist** + threshold pre-fill → en-face footprint → cRORA **area mm² vs advRPE**
  (`api/routes_segmentation.py` + `core/{mask_store,footprint,segmenter_client}.py` +
  `web/js/segmentation_view.js`). **Export** `src/export_bscan_dataset.py` → patient-split B-scan dataset.
  **SAM2** box/point assist on Colab (`segmenter_service/sam2_serve.py` + `sam2_colab.ipynb`); MedSAM3-text
  is demoted to legacy (it segments the RPE band, not GA). The old `/ga.png` stub is superseded.

---

## 1. Run

```powershell
# first time only
oct_env\Scripts\python.exe -m pip install -r reader\requirements-reader.txt
# launch (from the repo root). add --reload during development to auto-pick-up code edits.
oct_env\Scripts\python.exe -m uvicorn reader.api.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/** → **Open E2E…** → browse to a `.E2E` (or paste an absolute path) →
pick an eye (the **6×6 / 97-line** volume is highlighted and default). Slow operations (loading a
volume, whole-volume BM re-segment, the PLEX registered lock, seeding, MedSAM3 labeling) close the
current menu and show a full-screen **loading overlay** (`js/loading.js`) — a spinner + label, and a
`done/total` progress bar for the MedSAM3 job (with a "Run in background" dismiss).

**Deep link** (bookmark a case; also used for headless screenshots):
```
http://127.0.0.1:8000/?open=<abs .E2E path>&eye=OD[&view=projection]
```

The only new dependencies beyond the pipeline are **fastapi** + **uvicorn[standard]**
(`reader/requirements-reader.txt`). Everything else (numpy, opencv, scipy, oct-converter, eyepy) is
already in `oct_env`.

---

## 2. What it does

### B-scan view (the workbench)
- **Centre, top:** the B-scan. Scroll with the slider, **↑/↓**, **PageUp/PageDown**, or the **mouse
  wheel**; HEYEX-style `B-scan i / n` counter. Toggle **ILM** (orange) and **BM** (yellow) lines, each
  tagged **`device`** / **`auto`** / **`edited`**, and the projection **bands** shaded on the B-scan
  (**slab** BM+10..+340 µm green = transmission numerator; **rpe** BM−45..−5 µm magenta = the band that
  vanishes in GA).
- **Centre, bottom — live projection panel.** The en-face projection of the *current* volume, with a
  feature dropdown (default **`raw`**). Updates live while you edit a layer.
- **Probe (cause ↔ effect).** Tick **Probe** in the projection header, then **click anywhere on the
  en-face map**: it jumps the slider to the B-scan that feeds that spot and draws a **cyan column** on
  the B-scan at the exact A-scan, with the **sub-BM slab segment** emphasised (the depth that produces
  the pixel); a crosshair marks the clicked spot on the map. Untick to hide the highlight and compare
  the B-scan clean vs. marked. The mapping is the exact inverse of `to_enface` (column = A-scan,
  row = B-scan with the fundus row-flip, so the map's bottom row is B-scan 1); front-end only.
- **Left:** the **localizer** (IR/NIR SLO) with a green current-position line.
- **Right — Layers & edit:** the toggles + the **layer-correction editor** (see below).

### Layer correction (Phase 2 — done)
Pick **Edit layer = ILM / BM**, choose **Apply to = All B-scans / This B-scan**, then move the layer:
**scroll**, **drag** vertically, or **▲/▼ / ↑/↓** (1 px). The overlay (and the slab band) track the
edit and the **projection updates in real time**.
- **All B-scans** = a rigid whole-volume shift (the projection-tuning primitive — one B-scan barely
  moves the en-face). **This B-scan** = a local fix; turning on **Draw mode** lets you brush-reshape the
  layer on that B-scan.
- **Slab depth (green zone)** — `lo/hi` µm below BM, applied to all B-scans; the green band redraws and
  the raw/f_trans projection recomputes. (A live tuning knob; not persisted.)
- **Save** persists (the whole-volume shift → `global.json`; a per-B-scan edit → `bscan_<idx>.json`).
  **Revert** drops it. Corrections fold into the projection via `effective_surfaces` and double as DL
  labels. Performance: a whole-volume shift or a slab change recomputes the en-face; a single-B-scan edit
  at the default slab patches only that B-scan's cached native row (sub-ms).

### BM-validation Library + workbench (current focus)
The **Library tab** (landing page) lists the cohort from `results/bm_worklist.csv` (precomputed by
`src/build_bm_worklist.py` from the master `spectralis_ga_pairing.csv`, `qc_status==ok`) with a
red/orange/green progress pill (`k/N` B-scans validated) and a **device / no-device** tag. **Both kinds are
clickable:** device-BM eyes seed from the device contour; **no-device eyes auto-pre-segment with the DL
model on open** (so they no longer open empty). Clicking a row opens that E2E + eye + 6×6 via the normal
open flow and lands on the **BM tab**, which (`bmFocus`) becomes a **BM validation station**:
- **ILM is dropped** — not shown or edited (only the transmission projection ever uses ILM, never BM).
- **already pre-segmented (no Segment click)** — for any B-scan that is still **unsegmented AND unvalidated
  AND has no existing correction**, BM is filled by the **DL model** (newest `outputs/bm_dl/bm_unet.onnx`,
  via `bm_dl.available()` — no `OCT_BM_DL` flag needed) and **cached** as a correction. It never overwrites
  device BM, a human edit, a prior fill, or a validated B-scan, and is idempotent. The DL pass is ~25s on
  CPU, so it is **precomputed OFFLINE** by `src/presegment_nodevice.py` (run once over the no-device
  worklist) → the eye opens already segmented + instantly. The BM tab also calls
  `POST …/corrections/presegment_bm` client-side as a fallback for any not-yet-precomputed eye (loading
  toast); both share `routes_corrections.presegment_eye`. Cold opens are fast because `load_volume`
  memoizes its surfaces to `reader/data_store/surfcache/` (the ~5-12s classical self-seg is paid once);
  `volume_open` runs no segmentation.
- the **control-point spline** is auto-enabled; drag points to fix BM, or just accept the seeded line.
- **auto-persist, no Save** — every spline edit debounce-writes a per-B-scan BM correction; navigating
  away flushes the pending write first, so nothing is lost.
- **validate-on-advance** — **"Validate & next ›"** (or **Space**) marks the current B-scan validated and
  advances; wheel / arrows / slider / filmstrip just navigate (no side effect). The last B-scan's button
  reads **"Validate (done)"**.
- a **filmstrip** colours each B-scan **validated** (green) / **pre-segmented (model)** (purple) /
  **edited** (cyan) / **device BM** (amber) / none; click to jump; a `k/N validated` counter sits beside
  it. The per-B-scan badge + layer-panel chip carry the same state — a DL pre-seg reads **"pre-segmented
  (model)"** until you edit it (→ cyan "edited") or validate it (→ green). The source tag is persisted as
  `bm_src` on the correction (`"model"` from DL pre-seg / Label-with-DL, `"user"` from a human edit).
- **Copy BM from prev** copies the *previous* B-scan's BM onto **this B-scan only** (per-image, never a
  batch) as a starting point you then tweak; **Reset to device** drops this B-scan's edit; **Re-segment**
  re-runs self-seg for this one B-scan.
- **Reset ALL to device** (with a confirm) resets the **whole eye** back to the device segmentation —
  discards every per-B-scan BM edit AND clears every validation mark, so labeling starts over from device.
The validated flag persists in `bm_status.json` next to the correction sidecars
(`reader/data_store/corrections/<eid>_<eye>/`). Purpose: hand-validate device BM scan-by-scan → export a
patient-split BM-surface training set → train a DL BM model behind `segment_bm()` (classical hand-tuning of
`bm.py` was abandoned after the peak-anchored reroute regressed volume-wide).

### Features (live projection panel + the Projection tab)
En-face float map → fixed cross-eye **display window** → grayscale + 1 mm scale bar:

  | feature | what | window | reads |
  |---|---|---|---|
  | `raw` | Spectralis-style sub-BM slab (mean intensity, robust-normalised, **not** destriped) | `[0.0, 1.0]` | device-like view |
  | `f_trans` | ILM-anchored transmission fraction | `[0.18, 0.62]` | GA bright |
  | `f_gated` | transmission gated by RPE-integrity (the deliverable) | `[0.02, 0.40]` | bright only where RPE is gone |
  | `f_rpe` | shadow-invariant RPE-loss cue | `[-0.85, 0.05]` | high = RPE lost |

The separate **Projection tab** additionally **live-tunes** the transmission slab bands (recomputes
from the volume).

### GA — OAC auto-detector (Segment tab)
The **OAC GA (auto)** button runs the BM-anchored OAC GA pipeline (`reader/core/oac_ga`, byte-identical to
the CLI `src/oac_area.py`) on the open eye using the **corrected BM**, and shows the RPE-loss en-face with
the green **cRORA footprint** + the area (and advRPE, in response headers). A **quadratic/linear** selector
picks the healthy-RPE baseline: **quadratic** (default — best on focal eyes; 005 OD gold-validated Dice
0.93) or **linear** (cleaner on large eccentric lesions; 008 OS). Detector = RPE-loss (mean OAC above BM)
vs the robust baseline **+ the 2nd cRORA criterion** (sub-BM hypertransmission fills heterogeneous GA
centres OAC misses, and drops no-transmission vignette corners) → cRORA ≥250 µm. Meaningful only on
BM-corrected eyes; on a huge eccentric lesion the quadratic baseline still leaves some corner FP (a
gold/learned baseline is the real fix — see `CLAUDE.md`).

### Stubbed (Phase 2, returns `501`)
- **GA overlay** (`/ga.png`) — old segmenter-outline stub (superseded by the OAC GA detector above).

---

## 3. Key decisions & data model

- **Target scan = the native 6×6, no crop.** Each E2E holds, per eye: a **121-line 30°** field
  (8.77×7.31 mm), a **97-line ~6×6** square field (512-wide, ≈5.8×5.8–6.6 mm), and a 1-line foveal.
  The reader targets the **97-line "6×6"** volume and isotropically resamples the **whole native field**
  to a square en-face frame — **no central 6×6 crop** (the user dropped the 30°-crop approach). Some
  6×6 verticals exceed 6 mm, so the frame is sized to contain the field (`projection.to_enface`).
- **Calibration is angular.** With no biometry, HEYEX uses the 24 mm model eye (**0.2924 mm/deg**). FOV
  is measured **per scan** from the B-scan angular endpoints (`calibration.class_fov_mm`, ported from
  `src/validate_spectralis.py`). Axial is metric: **3.8717 µm/px**. En-face frame mm/px = `6/512`
  (`register_qc.ADV_MMPP`).
- **Two layers, BM-anchored.** The E2E carries only **ILM (`contour0`) + BM (`contour1`)** — no device
  RPE/choroid line. BM is the anchor; "RPE loss" is captured by a band just above BM, not a traced line.
- **device vs auto.** Device contours are used where present; where absent, the reader **self-segments**
  (`bm.segment_volume` for BM + a self-seg ILM via `bm`'s graph helpers) and tags the layer `auto`.
  Self-seg eyes load in ~10 s (graph search); device-layer eyes in <1 s. Both are cached after first
  load.

---

## 4. Architecture

Three strictly separated layers. Dependency rule: `core/` imports `src/` + numpy/cv2 only; `api/`
imports `core/` + fastapi; `web/` calls the REST API only (no Python).

```
reader/
  READER.md                 # this document
  README-reader.md          # short quickstart (points here)
  requirements-reader.txt   # fastapi, uvicorn[standard]  (the only new deps)
  .gitignore                # __pycache__, data_store/, _smoke_*/_probe_* scratch
  __init__.py

  core/                     # DOMAIN LOGIC — no web deps
    __init__.py             # adds <repo>/src to sys.path so `import m2_bm` etc. resolve
    calibration.py          # AXIAL_UM_PER_PX, MM_PER_DEG, class_fov_mm, mm/px helpers
    e2e_source.py           # open E2E, enumerate volumes, load a volume (device|self-seg), localizer
    volume.py               # RawE2E / VolumeRef / OctVolume dataclasses
    layers.py               # LAYER_DEFS, BAND_DEFS, device_layers_json, LayerStore seam, effective_surfaces
    layer_store.py          # JsonSidecarLayerStore — manual corrections + BM-validation status (bm_status.json)
    projection.py           # FEATURES(+raw), native_full/finish/preview_enface, enface, recompute_transmit
    render.py               # numpy -> PNG bytes (bscan, projection, localizer)
    filesystem.py           # root-restricted directory listing for the file browser
    segmenter.py            # GaSegmenter seam + HeuristicSegmenter stub (Phase 2)
    ids.py                  # stable e2e_id / volume_id
    library.py              # BM-validation cohort listing (results/bm_worklist.csv + the layer store)

  api/                      # THIN HTTP LAYER — fastapi only
    __init__.py
    app.py                  # FastAPI factory: routers + StaticFiles(web) + CORS
    session.py              # SessionStore: decode-once, in-RAM LRU of RawE2E / OctVolume / projections
    deps.py                 # DI singletons (store, layer store, segmenter) — the only swap point
    schemas.py              # pydantic request models (OpenE2EIn, CorrectionIn, BmValidateIn, …)
    routes_fs.py            # /api/health, /api/fs/list, /api/e2e/open, /api/volumes/{id}/open
    routes_bscan.py         # bscan png, layers json (+ device_bm presence), localizer png
    routes_projection.py    # projection png, meta, recompute png
    routes_corrections.py   # layer corrections GET/PUT/DELETE + copy_prev + bm_status (BM validation)
    routes_ga.py            # Phase-2 GA overlay (501 stub)
    routes_library.py       # /api/library + /api/library/bm_status (BM-validation cohort)

  web/                      # STATIC FRONTEND — vanilla JS, no build step
    index.html
    css/app.css
    js/
      api.js                # the only place that knows endpoint URLs
      state.js              # tiny pub/sub store
      main.js               # bootstrap: open modal, file browser, eye/volume picker, tab router, deep link
      lib.js                # loadImage, colour/el helpers, debounce
      bscan_view.js         # two-canvas viewer + layer panel; `bmFocus` = the BM-validation workbench
      projection_view.js    # feature select, display window, live slab-band tuning
      library_view.js       # BM-validation Library: cohort table, progress pills, opens a scan
```

`reader/data_store/corrections/<eid>_<eye>/` (gitignored) holds the layer-correction sidecar JSON
(`bscan_<idx>.json`, `global.json`) and the BM-validation flag (`bm_status.json`).

---

## 5. Module reference (key functions)

### core
- **`calibration`** — `AXIAL_UM_PER_PX=3.8717`, `MM_PER_DEG=0.2924`; `class_fov_mm(bscan_data)` →
  `{(numImages,imgSizeX): (H_mm,V_mm)}`; `axial_mm_per_px()`, `lateral_mm_per_px(fov,W)`.
- **`e2e_source`** —
  - `open_e2e(path) → RawE2E` (`read_oct_volume` + `read_all_metadata`; builds `VolumeRef`s + a best-
    effort eyepy modality map for localizer labelling).
  - `default_volume_index(raw, eye=None)` → the 6×6 (most B-scans), else the largest volume.
  - `load_volume(raw, index) → OctVolume` — `_device_layers` where present (robust to **ragged
    contours**), else `bm.segment_volume` + `_self_ilm_volume`; both surfaces `m2_bm.fill_bm`'d.
  - `localizer_image(raw, eye)` — IR/NIR fundus from the E2E (prefers IR modality).
- **`volume`** — `OctVolume(vol[n,H,W], ilm_display/bm_display [may have NaN], ilm/bm [filled for
  projection], ilm_src/bm_src, fov_mm, …)`; `VolumeRef` (eye, n, H, W, fov_mm, is_6x6, kind); `RawE2E`.
- **`layers`** — `LAYER_DEFS` (ilm/bm colours), `BAND_DEFS` (slab/rpe in µm, from `m3.SLAB_UM`/
  `RPEBAND_UM`); `device_layers_json(ov)` (whole-volume payload: surfaces as `[[y|null]×W]×N`, band
  defs, `axial_um_per_px`, sources); **`effective_surfaces(ov, store)`** = the single choke point that
  merges device/self surfaces with (Phase-2) corrections; `LayerStore` Protocol + `NullLayerStore`.
- **`layer_store`** — `JsonSidecarLayerStore(root)` implementing the `LayerStore` Protocol:
  `get_corrected`/`put_corrected`/`delete_corrected`/`corrected_indices`, persisted one JSON sidecar per
  corrected B-scan with an in-RAM `(eid,eye)` cache (atomic writes).
- **`projection`** — `FEATURES` {name→window} (incl. **`raw`**); `SINGLE_NATIVE` (raw/f_trans/f_rpe have
  a per-row-patchable native); `native_full(ov,feature,surf)` + `_native_one` + `finish(ov,feature,nat)`
  (raw = robust-normalise, no destripe; cues destripe); `enface`; `preview_enface(…,bscan,layer,ys,nat)`
  (live one-B-scan edit, patches the cached native); `to_enface` (no-crop resample); `recompute_transmit`
  (live slab tuning). Reuses `m3.proj_transmit_ilm`/`band`/`proj_rpe_loss_ilm`/`proj_rpe_present_ilm`/
  `gated_feature`/`destripe2d`.
- **`render`** — `bscan_png(ov, idx)` (raw grayscale; client overlays layers), `projection_png(map,
  window)` (fixed window + 1 mm scale bar), `localizer_png(loc, idx, n)` (line; `idx<0` → clean base).
- **`filesystem`** — `list_dir(dir, root=DATA_DIR)` (confined to a root), `is_e2e(path)`.
- **`segmenter`** — `GaSegmenter` Protocol; `HeuristicSegmenter` threshold stub (NOT a validated area).
- **`ids`** — `e2e_id(path)` (stable sha1[:12]), `volume_id(eid, idx)`, `parse_volume_id`.

### api
- **`session.SessionStore`** — `open_e2e`, `get_raw`, `get_volume` (load-once + LRU), `get_projection`
  (caches the en-face float frame per `(volume_id, feature)`); thread-locked first loads.
- **`deps`** — singletons `get_store`, `get_layer_store` (→ `JsonSidecarLayerStore`, persisted layer
  corrections), `get_segmenter` (→ `HeuristicSegmenter`, still a stub). **The one place left to swap in
  the learned GA segmenter.** (`NullLayerStore` remains in `core.layers` as the no-op Protocol default.)
- **`app.create_app()`** — includes routers under `/api`, mounts `web/` at `/`, permissive CORS
  (localhost only). Handlers are sync `def` → FastAPI runs them in a threadpool so slow loads don't
  block the event loop.

### web
- **`bscan_view.js`** — two stacked canvases at the image's intrinsic size (image + transparent
  overlay), so toggling layers/bands is a pure client redraw (no refetch). Bands are filled between two
  BM-offset polylines computed from the BM array + `axial_um_per_px`. `hitTest` is the reserved seam for
  Phase-2 point dragging.
- **`projection_view.js`** — feature `<select>`, window lo/hi (re-fetches the cheap windowed PNG), slab
  lo/hi (debounced → `recompute.png`, "live/slower").
- **`main.js`** — modal + server-side folder browser + eye/volume picker + tab router; `?open=` deep
  link.

---

## 6. REST API

Base `http://127.0.0.1:8000`. `{vid}` = volume id from `/e2e/open`. Layers travel as JSON **separate**
from the B-scan PNG, so client toggling/editing never refetches the image.

| Method | Path | Params | Returns |
|---|---|---|---|
| GET | `/api/health` | — | `{ok, loaded:[e2e_id…]}` |
| GET | `/api/fs/list` | `?dir=` | `{dir, parent, root, entries:[{name,type:dir\|e2e,path,size_mb}]}` |
| POST | `/api/e2e/open` | body `{path}` | `{e2e_id, path, eyes:{OD:[{volume_id,kind,n_bscans,W,H,fov_mm,is_6x6}],…}, default:{OD:vid,…}}` |
| POST | `/api/volumes/{vid}/open` | — | `{volume_id,eye,n_bscans,H,W,fov_mm,axial_mm_per_px,lateral_mm_per_px,layers,sources,patient}` (`patient` = E2E's embedded `{patient_id,first_name,surname}` → topbar identity fallback) |
| GET | `/api/volumes/{vid}/bscan/{idx}.png` | — | `image/png` raw grayscale B-scan (404 out of range) |
| GET | `/api/volumes/{vid}/layers` | — | `{defs,bands,axial_um_per_px,sources,ilm,bm,device_bm?,corrected?}` (`device_bm` = per-B-scan device-BM presence; each `corrected[bi]` carries `bm`/`ilm` + a `bm_src` tag `"model"`\|`"user"`) |
| GET | `/api/volumes/{vid}/localizer.png` | `?bscan=` | `image/png` IR localizer (`bscan<0` = clean base) |
| GET | `/api/volumes/{vid}/projection.png` | `?feature=&lo=&hi=` | `image/png` en-face at the window |
| GET | `/api/volumes/{vid}/projection/meta` | `?feature=` | `{feature,features,default_window,windows,mmpp,bands,fov_mm}` |
| GET | `/api/volumes/{vid}/projection/recompute.png` | `?slab_lo=&slab_hi=&lo=&hi=` | `image/png` live band-tuned transmission |
| POST | `/api/volumes/{vid}/projection/preview.png` | body `{feature, global_ilm/global_bm, slab_lo/slab_hi, bscan,layer,shift\|ys, lo,hi}` | `image/png` projection with **live (unsaved)** whole-volume shift / slab / one-B-scan edit |
| GET | `/api/volumes/{vid}/corrections` | `?bscan=` | the corrected row(s) for a B-scan, or `{bscans:[…]}` |
| PUT | `/api/volumes/{vid}/corrections` | body `{layer, scope:"all"\|"bscan", bscan?, shift\|ys}` | save a whole-volume shift (`scope:"all"`) or a per-B-scan correction; invalidates the projection cache |
| DELETE | `/api/volumes/{vid}/corrections` | `?scope=all&layer=` or `?bscan=&layer=` | drop the whole-volume shift / a B-scan's correction |
| POST | `/api/volumes/{vid}/corrections/copy_prev` | `?bscan=&layer=bm` | copy the previous B-scan's BM onto **this** B-scan (per-image); returns the row |
| POST | `/api/volumes/{vid}/corrections/presegment_bm` | — | **auto pre-segment BM (DL)**: fill+cache every unsegmented, uncorrected, unvalidated B-scan, tagged `bm_src="model"`; soft no-op when the model is absent. Returns `{available,n,presegmented:[…]}` |
| DELETE | `/api/volumes/{vid}/corrections/all` | `?layer=bm` | **bulk reset**: drop EVERY per-B-scan correction for `layer` (+ its whole-volume shift) on this eye |
| GET / PUT / DELETE | `/api/volumes/{vid}/bm_status` | PUT body `{bscan,validated,by?}` | per-eye BM-validation progress / mark a B-scan validated / **DELETE** clears all validations for the eye |
| GET | `/api/library` | — | BM-validation cohort: device-BM 6×6 eyes + progress (red/orange/green) |
| GET | `/api/library/bm_status/{eid}/{eye}` | — | validated B-scan indices for one eye (single-row refresh) |
| GET | `/api/volumes/{vid}/segment/oac_ga.png` | `?frac=&order=` | **OAC GA auto-detector** (corrected BM): RPE-loss map + green cRORA footprint; area in `X-OAC-Area-mm2` / advRPE in `X-AdvRPE-Area-mm2` headers. `order` = 2 quadratic (default) / 1 linear |
| GET | `/api/volumes/{vid}/segment/export` | `?run=` | export this eye's labeled B-scans → `outputs/ga_bscan_dataset/` (merges this eye, keeps the rest) |
| GET | `/api/volumes/{vid}/ga.png`, `/ga/meta` | (Phase 2) | `501` (superseded by `segment/oac_ga.png`) |

---

## 7. Device-layer reality & the open decision

The E2E carries device ILM/BM on the **30° scan more often than the 6×6**. Sweep of all 20 E2Es
(40 eyes):

| where device layers exist | eyes | |
|---|---|---|
| **on the 6×6 already** | **16/40** | 002OD, 005, 006, 008, 009OD, 012, 015, 017, 108 — use directly |
| 6×6 missing, **but on the 30°** | **14/40** | 001OD, 002OS, 003, 004, 007OD, 009OS, 011OD, 013, 016OD, 026 |
| **nowhere** | **10/40** | 001OS, 007OS, 010, 011OS, 014, 016OS, 130 — must self-seg |

The 6×6 gap is largely an **HEYEX export/workflow artifact** (segmentation was run on the posterior-pole
30° scan, not the 20°/6×6 cube), not a fundamental limit.

**Resolved — (B) self-seg hardened (`src/bm.py`), (C) manual correction shipped.**
- **B (done).** The old top-down chain seeded by `_first_above` latched onto vitreous speckle (e.g.
  **003 OS B-scan 90**), dragging ILM→RPE→BM ~250 px shallow. Rewritten to **anchor the RPE/BM complex
  FIRST** (`_complex_anchor` = deepest *sustained* bright band via a box-sum, speckle-proof, ILM-free),
  **derive ILM relative to it** (`_ilm_from_complex`, sustained onset so a speckle pixel can't qualify),
  a per-column thickness clamp, and a slow-axis robust-outlier pass (`_robust_slow_axis`, fixes
  multi-B-scan latches a median-3 missed). `segment_surfaces[_volume]` return both surfaces in one pass;
  `segment_bm`/`segment_volume` signatures unchanged. Validated by `src/m2_bm.py`: **cohort central-6mm
  BM error 20.4→15.3 µm, worst 211→48 µm, eyes >40 µm 8→1, no regressions** (003 OS +0.4 µm = noise).
  The reader takes ILM+BM from this single pass (`e2e_source._self_ilm_volume` → `segment_surfaces_volume`).
- **C (done).** `JsonSidecarLayerStore` persists corrections; they fold into the projection through the
  existing `effective_surfaces` choke point and into the viewer through `device_layers_json`. See §8.
- **D — re-export device layers from HEYEX** remains a per-eye option for gold layers; **A — 30°→6×6
  transfer** stays deprioritised. Neither is needed now that self-seg is reliable + correctable.

---

## 8. Implementation seams

- **Layer correction (done).** `core.layer_store.JsonSidecarLayerStore` (lazy in-RAM cache) implements
  the `core.layers.LayerStore` Protocol + a whole-volume shift (`get_global`/`add_global`/`clear_global`
  → `global.json`); bound in `deps`, attached to the session via `SessionStore.attach_layer_store`. Both
  fold into the projection through **`effective_surfaces`** (global shift added to all rows, then per-B-scan
  absolute corrections) and into the viewer through **`device_layers_json`** (which also returns the
  current `global`). `routes_corrections` (GET/PUT/DELETE, `scope` = `all`|`bscan`) writes them and calls
  `SessionStore.invalidate_projection`. The real-time preview (`routes_projection.preview.png`) takes
  whole-volume shift / slab / one-B-scan edit; a pure single-B-scan edit at the default slab patches only
  that B-scan's cached native row (`projection.preview_enface`/`get_native`), else it recomputes the
  en-face (`projection.render_feature`, slab-aware via `native_full(..., slab)`).
- **GA overlay (still stubbed).** Implement `core.segmenter.GaSegmenter.segment(enface, fov) →
  (mask, area_mm2)` (the learned model), bind it in `deps.get_segmenter`, implement `routes_ga` (draw the
  outline with `qcviz.draw_contour`). Frontend: a toggleable overlay in `projection_view.js`.

---

## 9. Verification

- **Headless core** (no server): load a volume, compute `enface` for each feature, encode PNGs — all
  reuse paths exercised. (The build used throwaway `_smoke_*.py` scripts, since removed.)
- **API** (live-server round-trip, stdlib `urllib`): `health`; `e2e/open`; `volumes/{vid}/open`;
  `projection.png?feature=raw|f_trans`; `projection/preview.png` (shift); `PUT corrections` (shift +
  full-`ys`) → sidecar written + projection bytes change; `GET corrections`/`layers` show the edit;
  `DELETE` removes it + the sidecar; `ga → 501`.
- **B self-seg**: `src/m2_bm.py all` regenerates `results/m2_bm_errors.csv` — every device eye improved or
  held vs the pre-hardening baseline (cohort 20.4→15.3 µm; 003 OS +0.4 µm noise).
- **Browser** (headless Chrome via the `?open=` deep link, `outputs/reader_editor_shot.png`): the
  workbench renders — localizer + B-scan with ILM/BM + the live **raw** projection panel + the
  Correct-layer controls; a one-B-scan BM shift visibly changes the projection.

Confirmed working on a device-layer eye (008 OD, <1 s load) and a self-seg eye (130 OS / 001 OS, ~7 s load).

---

## 10. Known limitations

- **Slab depth is a live tuning knob, not persisted** — the projection panel always passes the current
  slab; the saved corrections (layer shifts/edits) persist, but the chosen slab resets on reload.
- **`raw` re-normalises globally**, so a one-B-scan edit slightly rebrightens the whole raw view (the
  edited band changes prominently); the destriped cue features don't. Cosmetic.
- **Self-seg residual tail**: a few eyes still have elevated `err_p95` (worst columns at the disc/scan
  edge), but medians are good and no eye regressed; the DL `segment_*` hook remains the long-term path.
- **Localizer position line is approximate** — linear in B-scan index (flipped to fundus orientation),
  not yet mapped from per-B-scan `centrePosY`.
- **Slab tuning exposes `slab_lo/slab_hi` only** — the ILM-anchored transmission's normalization top is
  the ILM by construction (the old BM-offset `REF_UM` doesn't apply).
- **No auto-reload by default** — Python (core/api) is loaded at server start; restart (or run with
  `--reload`) to pick up code edits. Static `web/` files are served live from disk.
- **Single local user** — bound to 127.0.0.1, permissive CORS, in-RAM session (no auth, no persistence
  beyond the process except the future correction sidecars).

---

## 11. Pointers

- Plan of record: `C:\Users\omerd\.claude\plans\help-me-build-a-misty-bubble.md`.
- Auto-memory: `project-reader-6x6.md` (the 6×6 pivot + device-layer sweep + this module).
- Reused pipeline code: `src/m2_bm.py`, `src/m3_projections.py`, `src/bm.py`, `src/qcviz.py`,
  `src/register_qc.py`, `src/validate_spectralis.py`, `src/paths.py`.
