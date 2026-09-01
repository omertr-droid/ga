"""oct_ga reader — a local, modular web app to view Spectralis E2E B-scans, layer annotations,
and the en-face transmission projection.

Layers (strictly separated):
  reader.core  — domain logic, NO web deps (E2E load, layers, projection, render)
  reader.api   — thin FastAPI HTTP layer over core
  reader.web   — static vanilla-JS frontend (served by api)

Run the server from the repo root:
    oct_env\\Scripts\\python.exe -m uvicorn reader.api.app:app --host 127.0.0.1 --port 8000
"""
