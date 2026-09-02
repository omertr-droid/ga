"""Build the self-contained, OFFLINE GA Clinic package (Windows x64).

Run ONCE on the dev machine (Windows x64, the oct_env Python 3.11). It assembles, under dist/:

  oct_ga_clinic/   <- embeddable Python 3.11 + oct_env site-packages + app code + the DL BM model + run.bat

The end user copies/unzips the folder and double-clicks run.bat — no Python, no pip, no internet. The
clinic needs the live-upload + DL-Bruch's-membrane path, so it ships the scientific stack (numpy, opencv,
scipy, scikit-image, imagecodecs, eyepy, oct_converter, onnxruntime, openpyxl, fastapi/uvicorn) plus the
ONNX BM model.

The embeddable interpreter is the only piece not already on disk: provide it with --embed <zip-or-dir>,
or let the script download python-3.11.9-embed-amd64.zip from python.org (internet once).

Usage (from repo root):
  oct_env\\Scripts\\python.exe src\\package_clinic.py                 # full build (download runtime once)
  oct_env\\Scripts\\python.exe src\\package_clinic.py --zip           # + dist\\oct_ga_clinic.zip
  oct_env\\Scripts\\python.exe src\\package_clinic.py --refresh-app   # reuse runtime; refresh app + run.bat
  oct_env\\Scripts\\python.exe src\\package_clinic.py --no-site-packages   # fast structural dry-run

Set OCT_CLINIC_DIST to build outside OneDrive (avoids syncing the ~500 MB tree).
"""
import argparse
import os
import shutil
import urllib.request
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.environ.get("OCT_CLINIC_DIST") or os.path.join(REPO, "dist")
OUT = os.path.join(DIST, "oct_ga_clinic")
PORT = 8021
PYVER = "311"
EMBED_VER = "3.11.9"
EMBED_URL = f"https://www.python.org/ftp/python/{EMBED_VER}/python-{EMBED_VER}-embed-amd64.zip"

# Only entries verified unnecessary for the clinic upload/viewer path.  NOTE: matplotlib + h5py are
# pulled in transitively by eyepy, so they stay.  Confirm changes with a packaged real-E2E smoke test.
PRUNE_PREFIXES = {
    "_distutils_hack", "colorama", "distutils_precedence", "httptools", "imageio_ffmpeg", "pip", "pkg_resources",
    "pydicom", "pypdf", "python_dotenv", "pyyaml", "setuptools", "watchfiles", "websockets", "yaml",
}
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")

# The DL Bruch's-membrane model (+ its self-describing sidecar). bm_dl.py searches OUT_DIR/bm_dl, where
# OUT_DIR = REPO_ROOT/outputs and REPO_ROOT resolves to app/ in the package — so it lands at
# app/outputs/bm_dl/. We ship ONLY the live ONNX model + sidecar (not the .pt / variant weights).
MODEL_FILES = ["bm_unet.onnx", "bm_unet.onnx.meta.json"]


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
        if os.path.isfile(zpath) and os.path.getsize(zpath) > 0:
            log(f"reusing cached {zpath}")
        else:
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
        removed = []
        for name in os.listdir(dst):
            low = name.lower().replace("-", "_")
            if not any(low == p or low.startswith(p + "_") or low.startswith(p + ".")
                       for p in PRUNE_PREFIXES):
                continue
            target = os.path.join(dst, name)
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            else:
                try:
                    os.unlink(target)
                except OSError:
                    continue
            removed.append(name)
        log(f"  pruned {len(removed)} unused package entries")


def copy_app(app=None):
    app = app or os.path.join(OUT, "app")
    if os.path.isdir(app):
        shutil.rmtree(app)
    os.makedirs(app)
    # clinic/ — ship a CLEAN database (the user's uploads on THIS machine populate it); never carry the
    # dev box's patients.csv, its staged drag-and-drop E2Es (~300 MB each), or temp files.
    clinic_ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "patients.csv",
                                           ".patients-*.tmp", "uploads", ".stage-*.tmp")
    shutil.copytree(os.path.join(REPO, "clinic"), os.path.join(app, "clinic"), ignore=clinic_ignore)
    # viewer/ — only viewer.core is imported (viewmodel, bundle, ga_native, locator). Drop the viewer's
    # own web app, API, and its baked library bundles (the clinic has no library / no PLEX).
    viewer_ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "web", "api", "data_store")
    shutil.copytree(os.path.join(REPO, "viewer"), os.path.join(app, "viewer"), ignore=viewer_ignore)
    # reader/ — only reader.core is imported. Drop the dev web app, stores, segmenter service.
    reader_ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "web", "data_store", "segmenter_service")
    shutil.copytree(os.path.join(REPO, "reader"), os.path.join(app, "reader"), ignore=reader_ignore)
    # src/ pipeline modules (drop archived one-off diagnostics)
    src_ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "exploration")
    shutil.copytree(os.path.join(REPO, "src"), os.path.join(app, "src"), ignore=src_ignore)
    # the DL Bruch's-membrane model -> app/outputs/bm_dl/ (where bm_dl.py looks for it)
    mdir = os.path.join(app, "outputs", "bm_dl")
    os.makedirs(mdir, exist_ok=True)
    for fn in MODEL_FILES:
        src_m = os.path.join(REPO, "outputs", "bm_dl", fn)
        if os.path.exists(src_m):
            shutil.copy2(src_m, os.path.join(mdir, fn))
            log(f"  bundled model file {fn}")
        elif fn == "bm_unet.onnx":
            log("  WARNING: outputs/bm_dl/bm_unet.onnx not found — DL BM will be unavailable in the package")
    return app


def write_runtime():
    # Open the browser ONLY once the server answers /api/health (the embeddable's first start can take
    # 20-40 s while Windows scans the files) — opening it early is what looks "stuck on connecting".
    waiter = (f"$u='http://127.0.0.1:{PORT}/'; "
              "for($i=0;$i -lt 150;$i++){ try{ $null=Invoke-WebRequest ($u+'api/health') "
              "-UseBasicParsing -TimeoutSec 2; Start-Process $u; break } catch{ Start-Sleep -Milliseconds 800 } }")
    already_running = (f"$u='http://127.0.0.1:{PORT}/'; try {{ $r=Invoke-RestMethod ($u+'api/health') "
                       "-TimeoutSec 2; if($r.app -eq 'ga-clinic'){ Start-Process $u; exit 0 } } "
                       "catch {}; exit 1")
    bat = (
        "@echo off\r\n"
        "setlocal\r\n"
        "set HERE=%~dp0\r\n"
        f"powershell -NoProfile -Command \"{already_running}\"\r\n"
        "if not errorlevel 1 exit /b 0\r\n"
        "set MPLBACKEND=Agg\r\n"
        "set PYTHONDONTWRITEBYTECODE=1\r\n"
        "REM default the DL Bruch's-membrane to ON (the upload BM prompt pre-selects it)\r\n"
        "set OCT_BM_DL=1\r\n"
        "set OCT_CLINIC_DATA=%HERE%user_data\r\n"
        "if not exist \"%OCT_CLINIC_DATA%\" mkdir \"%OCT_CLINIC_DATA%\"\r\n"
        "REM isolate from any Python the host machine may have installed (embeddable uses its own _pth)\r\n"
        "set PYTHONHOME=\r\n"
        "set PYTHONPATH=\r\n"
        "set PYTHONSTARTUP=\r\n"
        "echo ============================================================\r\n"
        "echo   GA Clinic - starting up\r\n"
        "echo   The FIRST launch can take 20-40 seconds while Windows\r\n"
        "echo   scans the files. Your browser opens automatically when\r\n"
        "echo   ready. Keep this window open while using the app.\r\n"
        "echo ============================================================\r\n"
        f"start \"\" /b powershell -NoProfile -WindowStyle Hidden -Command \"{waiter}\"\r\n"
        f"\"%HERE%python\\python.exe\" -m uvicorn clinic.api.app:app --host 127.0.0.1 --port {PORT}\r\n"
        "echo.\r\n"
        "echo The GA Clinic has stopped. You can close this window.\r\n"
        "pause\r\n"
    )
    with open(os.path.join(OUT, "run.bat"), "w", newline="") as f:
        f.write(bat)
    readme = (
        "GA Clinic (standalone)\r\n"
        "======================\r\n\r\n"
        "1. Double-click run.bat.\r\n"
        f"2. A browser tab opens at http://127.0.0.1:{PORT}/ . If not, open that address manually.\r\n"
        "3. Database = the patients tracked on this machine. Search by id or name, click to open.\r\n"
        "   Browse/paste uses an E2E in place; drag/drop stores a managed copy under user_data.\r\n"
        "4. Export Excel saves the patient database as an .xlsx file.\r\n"
        "   All portable database, upload and cache files live under user_data.\r\n"
        "5. To stop, close the black console window (or press Ctrl+C in it).\r\n\r\n"
        "No internet or Python install is required. Windows x64 only.\r\n"
    )
    with open(os.path.join(OUT, "README.txt"), "w", newline="") as f:
        f.write(readme)


def make_zip():
    base = os.path.join(DIST, "oct_ga_clinic")
    log("zipping (slow for a large tree) …")
    shutil.make_archive(base, "zip", root_dir=DIST, base_dir="oct_ga_clinic")
    return base + ".zip"


def build(args):
    py_dir = os.path.join(OUT, "python")
    if args.refresh_app:
        if not os.path.exists(os.path.join(py_dir, "python.exe")):
            raise SystemExit("--refresh-app needs an existing build; run a full build first")
        log("refresh-app: keeping python/, refreshing app code + run.bat …")
        copy_app()
        os.makedirs(os.path.join(OUT, "user_data"), exist_ok=True)
        write_runtime()
    else:
        if os.path.isdir(OUT):
            shutil.rmtree(OUT)
        os.makedirs(OUT)
        log("resolving embeddable interpreter …")
        embed = ensure_embed(args.embed)
        shutil.copytree(embed, py_dir, dirs_exist_ok=True)
        write_pth(py_dir)
        if not args.no_site_packages:
            copy_site_packages(py_dir)
        else:
            log("(--no-site-packages) skipping libs")
        log("copying app code + DL model …")
        copy_app()
        os.makedirs(os.path.join(OUT, "user_data"), exist_ok=True)
        write_runtime()

    sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) for f in fs)
    log(f"built {OUT}  ({sz / 1e6:.0f} MB)")
    if args.zip:
        log("created " + make_zip())
    log("done. Smoke-test: run dist\\oct_ga_clinic\\run.bat (then open an E2E and process a scan).")


def main():
    ap = argparse.ArgumentParser(description="Build the offline GA Clinic package (Windows x64).")
    ap.add_argument("--embed", default=None, help="path to python-3.11.x-embed-amd64.zip or an extracted dir")
    ap.add_argument("--zip", action="store_true", help="also produce dist/oct_ga_clinic.zip")
    ap.add_argument("--no-site-packages", action="store_true", help="skip copying libs (structural dry-run)")
    ap.add_argument("--refresh-app", action="store_true",
                    help="reuse the existing runtime; only refresh app code + launcher (fast)")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
