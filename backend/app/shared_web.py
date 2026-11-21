"""Serve the production frontend and the isolated API on one origin."""
from pathlib import Path

from fastapi.staticfiles import StaticFiles

from app.config import PROJECT_ROOT, PUBLIC_SHARED_MODE

if not PUBLIC_SHARED_MODE:
    raise RuntimeError("Start this entry point with IMAGE2EDITABLEPPT_PUBLIC_SHARED=1")

from app.main import app

dist = Path(PROJECT_ROOT) / "frontend" / "dist"
app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
