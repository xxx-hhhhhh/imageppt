from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import OUTPUTS_DIR, TEMP_DIR, UPLOADS_DIR, ensure_runtime_dirs


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectStore:
    def __init__(self) -> None:
        ensure_runtime_dirs()

    def project_dir(self, project_id: str) -> Path:
        path = TEMP_DIR / "projects" / project_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def project_file(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "project.json"

    def create(self, name: str) -> dict[str, Any]:
        project_id = uuid.uuid4().hex
        record = {"id": project_id, "name": name, "createdAt": _now(), "images": []}
        self.project_file(project_id).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        (UPLOADS_DIR / project_id).mkdir(parents=True, exist_ok=True)
        (OUTPUTS_DIR / project_id / "slides").mkdir(parents=True, exist_ok=True)
        (OUTPUTS_DIR / project_id / "assets").mkdir(parents=True, exist_ok=True)
        (OUTPUTS_DIR / project_id / "backgrounds").mkdir(parents=True, exist_ok=True)
        return record

    def get(self, project_id: str) -> dict[str, Any]:
        file = self.project_file(project_id)
        if not file.exists():
            raise FileNotFoundError(project_id)
        return json.loads(file.read_text(encoding="utf-8"))

    def save(self, record: dict[str, Any]) -> dict[str, Any]:
        self.project_file(record["id"]).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    def add_image(self, project_id: str, image: dict[str, Any]) -> dict[str, Any]:
        record = self.get(project_id)
        record.setdefault("images", []).append(image)
        return self.save(record)

    def slide_file(self, project_id: str, page: int) -> Path:
        return OUTPUTS_DIR / project_id / "slides" / f"page_{page}.json"

    def save_slide(self, project_id: str, page: int, layout: dict[str, Any]) -> None:
        from app.services.scene.ownership import canonicalize
        path = self.slide_file(project_id, page)
        path.parent.mkdir(parents=True, exist_ok=True)
        canonicalize(layout)
        if layout.get("sceneVersion") == "3.0":
            from app.services.scene.ownership import resolve_asset
            from app.services.visual_qa.analyzer import render_preview
            background = resolve_asset(layout.get("backgroundUrl"), OUTPUTS_DIR)
            if background and background.is_file():
                name = f"preview_page_{page}_{uuid.uuid4().hex[:12]}.png"
                render_preview(background, layout, OUTPUTS_DIR / project_id / "assets" / name)
                layout["previewUrl"] = f"/media/assets/{project_id}/{name}"
        pending = path.with_suffix(".pending")
        pending.write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(path)

    def get_slide(self, project_id: str, page: int) -> dict[str, Any]:
        file = self.slide_file(project_id, page)
        if not file.exists():
            raise FileNotFoundError(f"slide {page}")
        return json.loads(file.read_text(encoding="utf-8"))

    def list_slides(self, project_id: str) -> list[dict[str, Any]]:
        slides = []
        directory = OUTPUTS_DIR / project_id / "slides"
        if directory.exists():
            for file in sorted(directory.glob("page_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
                slides.append(json.loads(file.read_text(encoding="utf-8")))
        return slides
