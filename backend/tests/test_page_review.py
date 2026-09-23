from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.models.project_store import ProjectStore
from app.services.reconstruction.pipeline import AIUnavailableError


def test_pages_require_approval_before_next_page_and_export(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main_module, "store", ProjectStore(tmp_path / "projects"))

    class FakePipeline:
        def __init__(self, store: ProjectStore) -> None:
            self.scene_analyzer = SimpleNamespace(vision_routing={"usedProvider": "qwen", "aiUsed": True, "usedModel": "qwen3-vl-flash"})
            self.store = store

        def analyze_project(self, project_id: str, mode: str, page: int, allow_fallback: bool) -> tuple[list[dict], str, list[str]]:
            layout = {"version": "1.1", "slide": {"width": 100, "height": 60}, "elements": []}
            self.store.save_slide(project_id, page, layout)
            return [layout], "paddleocr", []

    monkeypatch.setattr(main_module, "ReconstructionPipeline", FakePipeline)
    client = TestClient(main_module.app)
    project_id = client.post("/api/projects", json={"name": "review"}).json()["id"]
    image = tmp_path / "source.png"
    Image.new("RGB", (100, 60), "white").save(image)
    for index in range(2):
        response = client.post(f"/api/projects/{project_id}/images", files={"files": (f"page{index}.png", image.read_bytes(), "image/png")})
        assert response.status_code == 200
    assert client.post(f"/api/projects/{project_id}/analyze", params={"page": 2}).status_code == 409
    assert client.post(f"/api/projects/{project_id}/analyze", params={"page": 1}).status_code == 200
    assert client.post(f"/api/projects/{project_id}/export/pptx").status_code == 409
    assert client.post(f"/api/projects/{project_id}/pages/1/approve").status_code == 200
    assert client.post(f"/api/projects/{project_id}/analyze", params={"page": 2}).status_code == 200


def test_ai_failure_pauses_without_silent_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main_module, "store", ProjectStore(tmp_path / "projects"))

    class FailingPipeline:
        def __init__(self, store: ProjectStore) -> None:
            pass

        def analyze_project(self, project_id: str, mode: str, page: int, allow_fallback: bool) -> tuple[list[dict], str, list[str]]:
            raise AIUnavailableError("AI planning failed")

    monkeypatch.setattr(main_module, "ReconstructionPipeline", FailingPipeline)
    client = TestClient(main_module.app)
    project_id = client.post("/api/projects", json={"name": "review"}).json()["id"]
    image = tmp_path / "source.png"
    Image.new("RGB", (100, 60), "white").save(image)
    client.post(f"/api/projects/{project_id}/images", files={"files": ("page.png", image.read_bytes(), "image/png")})
    response = client.post(f"/api/projects/{project_id}/analyze", params={"mode": "maximum"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "AI_UNAVAILABLE"
    assert main_module.store.list_slides(project_id) == []
