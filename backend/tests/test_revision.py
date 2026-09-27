from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.models.project_store import ProjectStore
from app import main as api_main
from fastapi.testclient import TestClient
from app.services.reconstruction.revision import revise_problem_regions
from app.services.visual_qa.analyzer import render_preview, run_visual_qa


def _project(tmp_path: Path, *, suppressed: bool, duplicate: bool = False) -> tuple[ProjectStore, str, dict]:
    store = ProjectStore(tmp_path)
    project_id = store.create("revision-test")["id"]
    root = tmp_path / project_id
    (root / "backgrounds").mkdir()
    source = np.full((100, 240, 3), 255, np.uint8)
    cv2.putText(source, "HELLO", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite(str(root / "source.png"), source)
    cv2.imwrite(str(root / "backgrounds" / "page_1.png"), np.full_like(source, 255))
    line = {"id": "text_001", "type": "text", "text": "HELLO", "x": 20, "y": 28, "width": 130, "height": 40, "zIndex": 2, "style": {"fontSize": 28, "fontFamily": "Arial", "color": "#000000"}, "metadata": {"rawOCRBBox": [20, 28, 135, 60], **({"suppressRender": True} if suppressed else {})}}
    elements = [line]
    if duplicate:
        second = {**line, "id": "text_002", "metadata": {"rawOCRBBox": [20, 28, 135, 60]}}
        elements.append(second)
    layout = {"version": "1.1", "slide": {"width": 240, "height": 100}, "elements": elements}
    store.save_slide(project_id, 1, layout)
    store.add_image(project_id, {"id": "source", "name": "source.png", "path": str(root / "source.png")})
    preview = root / "reconstructed_preview.png"
    render_preview(root / "backgrounds" / "page_1.png", layout, preview)
    score = run_visual_qa(root / "source.png", preview, root, layout)
    score.update({"detectedTextCount": 2 if duplicate else 1, "editableTextCoverage": 0 if suppressed else 1})
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")
    return store, project_id, layout


def test_revision_restores_missing_editable_text_without_full_analysis(tmp_path: Path) -> None:
    store, project_id, _ = _project(tmp_path, suppressed=True)
    result = revise_problem_regions(store, project_id, 1)
    assert result["revisionRound"] == 1
    assert result["accepted"] is True
    assert result["editableCoverageAfter"] > result["editableCoverageBefore"]
    assert any(issue["problem"] == "missingEditableText" for issue in result["issuesBefore"])
    assert store.get_slide(project_id, 1)["elements"][0]["metadata"].get("suppressRender") is None


def test_revision_keeps_previous_page_when_metrics_do_not_improve(tmp_path: Path) -> None:
    store, project_id, original = _project(tmp_path, suppressed=False, duplicate=True)
    result = revise_problem_regions(store, project_id, 1)
    assert result["revisionRound"] == 1
    assert result["accepted"] is False
    assert store.get_slide(project_id, 1) == original
    history = json.loads((tmp_path / project_id / "revision_history_1.json").read_text(encoding="utf-8"))
    assert history[0]["accepted"] is False


def test_revision_endpoint_uses_saved_page_and_reports_result(tmp_path: Path, monkeypatch) -> None:
    store, project_id, _ = _project(tmp_path, suppressed=True)
    monkeypatch.setattr(api_main, "store", store)
    response = TestClient(api_main.app).post(f"/api/projects/{project_id}/pages/1/revise")
    assert response.status_code == 200
    payload = response.json()
    assert payload["revisionRound"] == 1
    assert payload["accepted"] is True
    assert payload["layout"]["elements"][0]["text"] == "HELLO"
    assert TestClient(api_main.app).post(f"/api/projects/{project_id}/pages/2/revise").status_code == 404
