# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the PharmAgent desktop app.

Build with desktop/build.sh (it builds the frontend first). Produces
desktop/dist/PharmAgent.app on macOS (windowed bundle), a PharmAgent/ onedir
folder elsewhere.

Layout inside the bundle (see desktop/pharmagent_desktop.py resource_root):
  frontend_dist/   built React app (served by FastAPI from the same origin)
  sample_data/     bundled example datasets; app.config.allowed_data_dirs
                   resolves backend/sample_data relative to the `app` package,
                   which lands at <root>/app, so this path keeps working.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

ROOT = Path(SPECPATH).resolve().parent          # repo root
BACKEND = ROOT / "backend"

hidden = (
    collect_submodules("app")                   # every backend module (lazy imports inside tools)
    + collect_submodules("uvicorn")
    + collect_submodules("scipy.special")
    + collect_submodules("scipy.optimize")
    + collect_submodules("scipy.stats")
    + ["webview", "webview.platforms.cocoa"]
)

datas = [
    (str(ROOT / "frontend" / "dist"), "frontend_dist"),
    (str(BACKEND / "sample_data"), "sample_data"),
]
datas += collect_data_files("docx")             # python-docx templates
datas += collect_data_files("scipy", includes=["**/*.npz", "**/*.npy"])

a = Analysis(
    [str(ROOT / "desktop" / "pharmagent_desktop.py")],
    pathex=[str(BACKEND)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib.tests", "scipy.tests", "numpy.tests", "pandas.tests",
              "pytest", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PharmAgent",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="PharmAgent")

app = BUNDLE(
    coll,
    name="PharmAgent.app",
    icon=None,
    bundle_identifier="ai.pmatrics.pharmagent",
    info_plist={
        "CFBundleDisplayName": "PharmAgent",
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": True,
        # WKWebView talks to the loopback backend over plain http
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
    },
)
