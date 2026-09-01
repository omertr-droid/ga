# oct_ga reader

A local, modular web app to view a Spectralis **E2E**: scroll the **6×6 OCT volume**'s B-scans
HEYEX-style, overlay the **ILM/BM** layers (and the projection **bands**), and inspect the en-face
**transmission projection** — the workbench for getting the best projection (and, later, GA).

> **Full documentation:** see **[`READER.md`](READER.md)** — architecture, every module, the REST API,
> the device-layer findings, the open decisions, and the Phase-2 plan. This file is just a quickstart.

## Run
```
oct_env\Scripts\python.exe -m pip install -r reader\requirements-reader.txt   # first time only
oct_env\Scripts\python.exe -m uvicorn reader.api.app:app --host 127.0.0.1 --port 8000
```
Open **http://127.0.0.1:8000/** → *Open E2E…* → browse to a `.E2E` (or paste a path) → pick an eye
(the **6×6 / 97-line** volume is highlighted and default).

Deep link to a case: `http://127.0.0.1:8000/?open=<abs .E2E path>&eye=OD[&view=projection]`.

## What it does (MVP)
- **B-scans:** scroll (slider / ↑↓ / PageUp-Down / mouse wheel) with `B-scan i / n`; localizer with the
  current position line; toggle **ILM**, **BM**, and the **transmission slab** (BM+10..+340 µm) and
  **RPE/EZ band** (BM−45..−5 µm) shaded regions.
- **Projection:** `f_trans` (transmission), `f_gated` (gated by RPE-integrity), `f_rpe` (RPE-loss) at a
  fixed display window; live **slab band tuning** (recomputes from the volume).
- Layers are **device** (Spectralis's own contours) where present, else **auto** (self-segmented) —
  shown by a tag. On the 6×6 scan device layers are absent ~half the time, so auto is common.

## Architecture (modular by design)
```
reader/
  core/   pure domain logic, NO web deps  (e2e_source, volume, calibration, layers, projection,
          render, filesystem, segmenter, ids) — reuses src/{m2_bm,m3_projections,bm,qcviz,register_qc}
  api/    thin FastAPI layer (app, session, deps, schemas, routes_*) — no image math
  web/    static vanilla-JS frontend (index.html, css, js/*) — talks to the API only
```
All science lives in `core/` (Python — edit there). The frontend is dependency-free static files.

## Forward-compatible seams (Phase 2, already stubbed)
- **Layer correction:** `core.layers.LayerStore` + `effective_surfaces` (the single merge point);
  `/api/.../corrections` returns 501 until a `JsonSidecarLayerStore` is bound in `api/deps.py`.
- **GA overlay:** `core.segmenter.GaSegmenter`; `/api/.../ga.png` returns 501 until a model is bound in
  `api/deps.py`. Drop-in changes touch only `deps.py`, not the routes or the viewer.

## Notes
- Calibration is angular (24 mm model eye, 0.2924 mm/deg); FOV is read per-scan from B-scan geometry.
- The en-face uses the **native 6×6 field with no central crop** (isotropic resample to a square frame).
- Loading a self-segmented eye takes ~10 s (graph-search); device-layer eyes load in <1 s. Cached after.
