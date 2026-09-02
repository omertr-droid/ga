"""FastAPI app factory for the oct_ga reader.

Run from the repo root:
    oct_env\\Scripts\\python.exe -m uvicorn reader.api.app:app --host 127.0.0.1 --port 8000
Then open http://127.0.0.1:8000/
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import (routes_fs, routes_bscan, routes_projection, routes_corrections, routes_ga,
               routes_segmentation, routes_library)

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


def create_app() -> FastAPI:
    app = FastAPI(title="oct_ga reader", version="0.1.0")

    # Local single-user tool bound to 127.0.0.1; permissive CORS is fine and avoids dev friction.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    # Force the browser to revalidate JS/CSS/HTML so an edit always loads fresh (no stale-module
    # confusion during active development). Last-Modified makes the revalidation a cheap 304.
    @app.middleware("http")
    async def _no_cache_assets(request, call_next):
        resp = await call_next(request)
        if request.url.path.endswith((".js", ".css", ".html")) or request.url.path == "/":
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    # API routes first (matched before the catch-all static mount).
    for r in (routes_fs, routes_bscan, routes_projection, routes_corrections, routes_ga,
              routes_segmentation, routes_library):
        app.include_router(r.router, prefix="/api")

    # Static frontend at the root (html=True serves index.html for "/").
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
