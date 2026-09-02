"""Build the self-contained, OFFLINE doctor-viewer package(s). Default: BOTH Windows and macOS.

Run ONCE on the dev machine (Windows x64, the oct_env Python 3.11). It assembles, under dist/:

  oct_ga_viewer/            <- Windows: embeddable Python 3.11 + oct_env site-packages + app + run.bat
  oct_ga_viewer_mac/(.tar.gz)  <- macOS (Apple Silicon): see src/package_app_mac.py — library-only,
                                  double-click run.command, fully offline.

The Windows doctor copies/unzips the folder and double-clicks run.bat — no Python, no pip, no internet.
Library scans need only numpy+cv2 (instant); uploading a new E2E uses the bundled scientific stack.
The macOS half is built by package_app_mac.build_mac() (called from here unless --only windows).

The embeddable interpreter is the only Windows piece not already on disk. Provide it with --embed
<zip-or-dir>, or let the script download python-3.11.9-embed-amd64.zip from python.org (internet once).

Usage (from repo root):
  oct_env\\Scripts\\python.exe src\\package_app.py                       # build BOTH (download runtimes once)
  oct_env\\Scripts\\python.exe src\\package_app.py --zip                 # + dist\\oct_ga_viewer.zip (mac always tars)
  oct_env\\Scripts\\python.exe src\\package_app.py --only windows        # Windows only
  oct_env\\Scripts\\python.exe src\\package_app.py --only mac            # macOS only
  oct_env\\Scripts\\python.exe src\\package_app.py --no-site-packages    # fast structural dry-run (no libs/wheels)
"""
import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Build OUTSIDE OneDrive by setting OCT_VIEWER_DIST (avoids syncing a ~500 MB tree); default = dist/.
DIST = os.environ.get("OCT_VIEWER_DIST") or os.path.join(REPO, "dist")
OUT = os.path.join(DIST, "oct_ga_viewer")
PYVER = "311"
EMBED_VER = "3.11.9"
EMBED_URL = f"https://www.python.org/ftp/python/{EMBED_VER}/python-{EMBED_VER}-embed-amd64.zip"

# verified not loaded by the viewer/upload path (free ~100 MB); see src/bake notes
PRUNE_PKGS = {"imageio_ffmpeg", "pydicom"}
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def log(*a):
    print(*a, flush=True)


def ensure_embed(embed_arg):
    """Return a directory containing the embeddable python.exe (download/extract as needed)."""
    work = os.path.join(DIST, "_embed")
    if embed_arg and os.path.isdir(embed_arg) and os.path.exists(os.path.join(embed_arg, "python.exe")):
        return embed_arg
    os.makedirs(work, exist_ok=True)
    zpath = embed_arg if (embed_arg and embed_arg.lower().endswith(".zip")) else os.path.join(work, "embed.zip")
    if not (embed_arg and embed_arg.lower().endswith(".zip")):
        log(f"downloading {EMBED_URL} …")
        urllib.request.urlretrieve(EMBED_URL, zpath)
    ext = os.path.join(work, "py")
    if os.path.isdir(ext):
        shutil.rmtree(ext)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(ext)
    if not os.path.exists(os.path.join(ext, "python.exe")):
        raise SystemExit("embeddable zip did not contain python.exe")
    return ext


def write_pth(py_dir):
    """Enable site-packages + the app dir on the embeddable's import path."""
    with open(os.path.join(py_dir, f"python{PYVER}._pth"), "w") as f:
        f.write(f"python{PYVER}.zip\n.\nLib\\site-packages\n..\\app\nimport site\n")


def copy_site_packages(py_dir, prune=True):
    src = os.path.join(REPO, "oct_env", "Lib", "site-packages")
    dst = os.path.join(py_dir, "Lib", "site-packages")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    log("copying site-packages (this is the big step) …")
    shutil.copytree(src, dst, ignore=IGNORE, dirs_exist_ok=True)
    if prune:
        for name in PRUNE_PKGS:
            p = os.path.join(dst, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
                log(f"  pruned {name}")


def copy_app(app=None):
    app = app or os.path.join(OUT, "app")
    if os.path.isdir(app):
        shutil.rmtree(app)
    os.makedirs(app)
    # viewer/ (incl data_store/library = the baked bundles); ship a CLEAN tracking log (the doctor's
    # uploads on THIS machine populate it) — never carry the dev box's registry.csv or its temp files.
    viewer_ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "registry.csv", ".registry-*.tmp")
    shutil.copytree(os.path.join(REPO, "viewer"), os.path.join(app, "viewer"), ignore=viewer_ignore)
    # reader/ — only what viewer imports (reader.core). Drop the dev web app, stores, segmenter service.
    reader_ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "web", "data_store",
                                           "segmenter_service")
    shutil.copytree(os.path.join(REPO, "reader"), os.path.join(app, "reader"), ignore=reader_ignore)
    # src/ pipeline modules (drop archived one-off diagnostics)
    src_ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "exploration")
    shutil.copytree(os.path.join(REPO, "src"), os.path.join(app, "src"), ignore=src_ignore)
    return app


def write_runtime():
    # Open the browser ONLY once the server answers /api/health (the embeddable's first start can take
    # 20-40 s while Windows scans the files) — opening it early is what looks "stuck on connecting".
    waiter = ("$u='http://127.0.0.1:8011/'; "
              "for($i=0;$i -lt 150;$i++){ try{ $null=Invoke-WebRequest ($u+'api/health') "
              "-UseBasicParsing -TimeoutSec 2; Start-Process $u; break } catch{ Start-Sleep -Milliseconds 800 } }")
    bat = (
        "@echo off\r\n"
        "setlocal\r\n"
        "set HERE=%~dp0\r\n"
        "set MPLBACKEND=Agg\r\n"
        "set PYTHONDONTWRITEBYTECODE=1\r\n"
        "REM isolate from any Python the host machine may have installed (embeddable uses its own _pth)\r\n"
        "set PYTHONHOME=\r\n"
        "set PYTHONPATH=\r\n"
        "set PYTHONSTARTUP=\r\n"
        "echo ============================================================\r\n"
        "echo   GA Viewer - starting up\r\n"
        "echo   The FIRST launch can take 20-40 seconds while Windows\r\n"
        "echo   scans the files. Your browser opens automatically when\r\n"
        "echo   ready. Keep this window open while using the viewer.\r\n"
        "echo ============================================================\r\n"
        f"start \"\" /b powershell -NoProfile -WindowStyle Hidden -Command \"{waiter}\"\r\n"
        "\"%HERE%python\\python.exe\" -m uvicorn viewer.api.app:app --host 127.0.0.1 --port 8011\r\n"
        "echo.\r\n"
        "echo The viewer has stopped. You can close this window.\r\n"
        "pause\r\n"
    )
    with open(os.path.join(OUT, "run.bat"), "w", newline="") as f:
        f.write(bat)
    readme = (
        "GA Viewer (standalone)\r\n"
        "======================\r\n\r\n"
        "1. Double-click run.bat.\r\n"
        "2. A browser tab opens at http://127.0.0.1:8011/ . If not, open that address manually.\r\n"
        "3. Library = the built-in validated scans. Click one to open the 3-panel viewer.\r\n"
        "   Add to library = paste an absolute .E2E path on this machine to process a new scan.\r\n"
        "4. To stop, close the black console window (or press Ctrl+C in it).\r\n\r\n"
        "No internet or Python install is required. Windows x64 only.\r\n"
    )
    with open(os.path.join(OUT, "README.txt"), "w", newline="") as f:
        f.write(readme)


def make_zip():
    base = os.path.join(DIST, "oct_ga_viewer")
    log("zipping (slow for a ~500 MB tree) …")
    shutil.make_archive(base, "zip", root_dir=DIST, base_dir="oct_ga_viewer")
    return base + ".zip"


def build_windows(args):
    py_dir = os.path.join(OUT, "python")
    if args.refresh_app:
        if not os.path.exists(os.path.join(py_dir, "python.exe")):
            raise SystemExit("--refresh-app needs an existing build; run a full build first")
        log("win refresh-app: keeping python/, refreshing app code + run.bat …")
        copy_app()
        write_runtime()
    else:
        if os.path.isdir(OUT):
            shutil.rmtree(OUT)
        os.makedirs(OUT)
        log("win: resolving embeddable interpreter …")
        embed = ensure_embed(args.embed)
        shutil.copytree(embed, py_dir, dirs_exist_ok=True)
        write_pth(py_dir)
        if not args.no_site_packages:
            copy_site_packages(py_dir)
        else:
            log("(--no-site-packages) skipping libs")
        log("win: copying app code + baked library …")
        copy_app()
        write_runtime()

    sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) for f in fs)
    log(f"built {OUT}  ({sz / 1e6:.0f} MB)")
    if args.zip:
        log("created " + make_zip())
    log("win: done. Smoke-test: run dist\\oct_ga_viewer\\run.bat")


def main():
    ap = argparse.ArgumentParser(
        description="Build the offline doctor-viewer package(s). Default: BOTH Windows and macOS.")
    ap.add_argument("--embed", default=None, help="(win) path to python-3.11.x-embed-amd64.zip or an extracted dir")
    ap.add_argument("--zip", action="store_true", help="(win) also produce dist/oct_ga_viewer.zip")
    ap.add_argument("--no-site-packages", action="store_true",
                    help="skip copying libs / vendoring wheels (structural dry-run, both platforms)")
    ap.add_argument("--refresh-app", action="store_true",
                    help="reuse the existing runtime; only refresh app code + launcher (fast, both platforms)")
    ap.add_argument("--only", choices=["both", "windows", "mac"], default="both",
                    help="which package to build (default: both)")
    ap.add_argument("--python", default=None,
                    help="(mac) path/URL to a python-build-standalone install_only .tar.gz (else auto-resolve latest)")
    ap.add_argument("--arch", default="aarch64-apple-darwin",
                    help="(mac) target: aarch64-apple-darwin (Apple Silicon, default) or x86_64-apple-darwin")
    args = ap.parse_args()

    if args.only in ("both", "windows"):
        build_windows(args)
    if args.only in ("both", "mac"):
        from package_app_mac import build_mac      # lazy: keeps package_app importable on its own
        build_mac(args)
    log("all done.")


if __name__ == "__main__":
    main()
