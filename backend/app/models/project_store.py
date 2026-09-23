from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import OUTPUTS_DIR


class ProjectStore:
    """File backed project state shared by the API and reconstruction pipeline."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or OUTPUTS_DIR

    def _record_path(self, project_id: str) -> Path:
        if not project_id or any(char not in "0123456789abcdef" for char in project_id.lower()):
            raise FileNotFoundError(project_id)
        return self.root / project_id / "project.json"

    def _write(self, record: dict) -> None:
        path = self._record_path(record["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def create(self, name: str) -> dict:
        record = {"id": uuid.uuid4().hex, "name": name, "createdAt": datetime.now(timezone.utc).isoformat(), "images": []}
        self._write(record)
        return record

    def get(self, project_id: str) -> dict:
        path = self._record_path(project_id)
        if not path.is_file():
            raise FileNotFoundError(project_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def add_image(self, project_id: str, image: dict) -> None:
        record = self.get(project_id)
        record.setdefault("images", []).append(image)
        self._write(record)

    def approve_page(self, project_id: str, page: int) -> None:
        record = self.get(project_id)
        if page < 1 or page > len(record.get("images", [])):
            raise ValueError("Page does not exist")
        self.get_slide(project_id, page)
        approved = set(record.get("approvedPages", []))
        approved.add(page)
        record["approvedPages"] = sorted(approved)
        self._write(record)

    def save_slide(self, project_id: str, page: int, layout: dict) -> None:
        self.get(project_id)
        if page < 1:
            raise ValueError("Page must be positive")
        path = self.root / project_id / "slides" / f"page_{page}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def get_slide(self, project_id: str, page: int) -> dict:
        self.get(project_id)
        path = self.root / project_id / "slides" / f"page_{page}.json"
        if page < 1 or not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def list_slides(self, project_id: str) -> list[dict]:
        self.get(project_id)
        directory = self.root / project_id / "slides"
        if not directory.exists():
            return []
        paths = sorted(directory.glob("page_*.json"), key=lambda path: int(path.stem.split("_")[-1]))
        return [json.loads(path.read_text(encoding="utf-8")) for path in paths]
