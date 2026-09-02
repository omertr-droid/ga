"""FastAPI app factory for the GA Clinic.

Run from the repo root (dev)::

    oct_env\\Scripts\\python.exe -m uvicorn clinic.api.app:app --host 127.0.0.1 --port 8021

Then open http://127.0.0.1:8021/  (the packaged app's run.bat does the same with the embedded Python).

The API router is mounted under ``/api`` BEFORE the static frontend is mounted at ``/`` (route-first
order), so ``/api/*`` is matched before the catch-all static handler. ``StaticFiles(html=True)`` serves
``index.html`` for any path without a file extension (single-page app).
"""
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import clinic.core  # noqa: F401  (side effect: put repo src/ on sys.path before anything imports bm_dl)
from . import routes

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


def create_app() -> FastAPI:
    app = FastAPI(title="GA Clinic", version="1.0.0")
    # The SPA and API are served by this same localhost origin.  CORS is unnecessary, and allowing every
    # website to call a patient-data localhost API would be an avoidable prototype security risk.

    @app.middleware("http")
    async def _no_cache_assets(request, call_next):
        resp = await call_next(request)
        if request.url.path.endswith((".js", ".css", ".html")) or request.url.path == "/":
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    app.include_router(routes.router, prefix="/api")
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
