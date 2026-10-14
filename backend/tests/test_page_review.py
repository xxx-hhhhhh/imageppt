from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import app.main as main_module
from app.models.project_store import ProjectStore
from app.services.reconstruction.pipeline import AIUnavailableError
from fastapi.testclient import TestClient
from PIL import Image


def install_inline_tasks(monkeypatch, api, pipeline_class):
    """Endpoint contract fixture; process/deadline semantics tested separately."""
    from app.services.reconstruction.pipeline import ReconstructionQualityError
    from app.services.reconstruction.tasks import AnalysisTasks
    manager = AnalysisTasks(api.store)
    def submit(project_id, page, mode, fallback=False):
        pipeline = pipeline_class(api.store)
        task = {"id": "f" * 32, "status": "succeeded", "message": "test"}
        try:
            _, provider, warnings = pipeline.analyze_project(project_id, mode, page, fallback)
            (api.store.root/project_id/'conversion_report.json').write_text(json.dumps({
                'ocrProvider': provider, 'warnings': warnings, 'routing': pipeline.scene_analyzer.vision_routing}), encoding='utf8')
        except (AIUnavailableError, ReconstructionQualityError) as exc:
            task.update(status='failed', errorCode='AI_UNAVAILABLE' if isinstance(exc, AIUnavailableError) else 'QUALITY_REJECTED')
        return task
    monkeypatch.setattr(manager, 'submit', submit)
    monkeypatch.setattr(api, 'analysis_tasks', manager)


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

    install_inline_tasks(monkeypatch, main_module, FakePipeline)
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

    install_inline_tasks(monkeypatch, main_module, FailingPipeline)
    client = TestClient(main_module.app)
    project_id = client.post("/api/projects", json={"name": "review"}).json()["id"]
    image = tmp_path / "source.png"
    Image.new("RGB", (100, 60), "white").save(image)
    client.post(f"/api/projects/{project_id}/images", files={"files": ("page.png", image.read_bytes(), "image/png")})
    response = client.post(f"/api/projects/{project_id}/analyze", params={"mode": "maximum"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "AI_UNAVAILABLE"
    assert main_module.store.list_slides(project_id) == []
