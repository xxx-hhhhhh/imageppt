from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pptx import Presentation

from app import main as api_main
from app.models.project_store import ProjectStore
from app.services.pptx import renderer
from app.services.reconstruction.objectization_audit import audit_objectization
from app.services.reconstruction.revision import revise_problem_regions
from app.services.reconstruction.white_objectization import objectize_on_white
from app.services.visual_qa.analyzer import render_preview, run_visual_qa


FIXTURE = Path(__file__).parent / "fixtures" / "complex_modules.png"


def test_audit_detects_missing_pale_plate_and_writes_owner_debug(tmp_path):
    source = np.full((180, 300, 3), 255, np.uint8)
    cv2.rectangle(source, (30, 40), (210, 112), (246, 248, 250), -1)
    cv2.imwrite(str(tmp_path / "source.png"), source)
    cv2.imwrite(str(tmp_path / "background.png"), np.full_like(source, 255))
    cv2.imwrite(str(tmp_path / "preview.png"), np.full_like(source, 255))
    report = audit_objectization(tmp_path / "source.png", tmp_path / "background.png", tmp_path / "preview.png", {"elements": []}, tmp_path / "debug.png")
    assert report["whiteBackground"] is True
    assert report["missingBackplates"] == 1
    assert report["missingVisualObjects"] == 0
    assert report["issues"][0]["bbox"][0] <= 30
    assert (tmp_path / "debug.png").is_file()


def test_audit_flags_image_owner_that_renders_blank(tmp_path):
    source = np.full((150, 240, 3), 255, np.uint8)
    cv2.circle(source, (70, 70), 22, (40, 80, 180), -1)
    cv2.imwrite(str(tmp_path / "source.png"), source)
    cv2.imwrite(str(tmp_path / "background.png"), np.full_like(source, 255))
    cv2.imwrite(str(tmp_path / "preview.png"), np.full_like(source, 255))
    layout = {"elements": [{"id": "badge", "type": "image", "x": 45, "y": 45, "width": 50, "height": 50, "src": "broken.png"}]}
    report = audit_objectization(tmp_path / "source.png", tmp_path / "background.png", tmp_path / "preview.png", layout)
    assert report["blankVisualOwners"] == 1
    assert any(item["problem"] == "blankVisualOwner" and item["elementId"] == "badge" for item in report["issues"])


def test_transparent_image_bbox_does_not_hide_missing_visual_content(tmp_path):
    source = np.full((150, 300, 3), 255, np.uint8)
    cv2.rectangle(source, (30, 45), (75, 100), (40, 90, 180), -1)
    cv2.rectangle(source, (175, 45), (220, 100), (50, 120, 70), -1)
    background = np.full_like(source, 255)
    preview = background.copy()
    preview[45:101, 30:76] = source[45:101, 30:76]
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    asset = np.zeros((80, 220, 4), np.uint8)
    asset[:, :, :3] = 255
    asset[0:56, 0:46, 3] = 255
    cv2.imwrite(str(asset_dir / "partial.png"), asset)
    for name, image in (("source.png", source), ("background.png", background), ("preview.png", preview)):
        cv2.imwrite(str(tmp_path / name), image)
    layout = {"elements": [{"id": "partial", "type": "image", "x": 30, "y": 45, "width": 220, "height": 80, "src": "/media/assets/demo/partial.png"}]}
    report = audit_objectization(tmp_path / "source.png", tmp_path / "background.png", tmp_path / "preview.png", layout)
    assert report["missingVisualObjects"] >= 1
    assert any(issue["bbox"][0] <= 175 <= issue["bbox"][2] for issue in report["issues"] if issue["problem"] == "missingVisualObject")


def test_revision_repairs_only_missing_local_plate(tmp_path):
    store = ProjectStore(tmp_path)
    project_id = store.create("plate-revision")["id"]
    root = tmp_path / project_id
    (root / "backgrounds").mkdir()
    source = np.full((180, 300, 3), 255, np.uint8)
    cv2.rectangle(source, (30, 40), (210, 112), (246, 248, 250), -1)
    cv2.imwrite(str(root / "source.png"), source)
    cv2.imwrite(str(root / "backgrounds" / "page_1.png"), np.full_like(source, 255))
    layout = {"slide": {"width": 300, "height": 180}, "elements": []}
    store.save_slide(project_id, 1, layout)
    render_preview(root / "backgrounds" / "page_1.png", layout, root / "reconstructed_preview.png")
    score = run_visual_qa(root / "source.png", root / "reconstructed_preview.png", root, layout)
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")
    result = revise_problem_regions(store, project_id, 1)
    assert result["accepted"] is True
    assert any(issue["problem"] == "missingBackplate" for issue in result["issuesBefore"])
    assert not any(issue["problem"] == "missingBackplate" for issue in result["issuesAfter"])
    assert len(store.get_slide(project_id, 1)["elements"]) == 1
    assert np.all(cv2.imread(str(root / "backgrounds" / "page_1.png")) == 255)
    assert cv2.imread(str(root / "reconstructed_preview.png"))[60, 60, 0] < 255


def test_revision_recovers_complex_unowned_visual_as_movable_image(tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("image-revision")["id"]
    root = tmp_path / project_id
    (root / "backgrounds").mkdir()
    source = np.full((180, 300, 3), 255, np.uint8)
    cv2.circle(source, (80, 75), 27, (30, 90, 185), -1, cv2.LINE_AA)
    cv2.circle(source, (80, 75), 8, (220, 220, 40), -1, cv2.LINE_AA)
    cv2.imwrite(str(root / "source.png"), source)
    cv2.imwrite(str(root / "backgrounds" / "page_1.png"), np.full_like(source, 255))
    layout = {"slide": {"width": 300, "height": 180}, "elements": []}
    store.save_slide(project_id, 1, layout)
    render_preview(root / "backgrounds" / "page_1.png", layout, root / "reconstructed_preview.png")
    score = run_visual_qa(root / "source.png", root / "reconstructed_preview.png", root, layout)
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")
    result = revise_problem_regions(store, project_id, 1)
    assert result["accepted"] is True
    images = [item for item in store.get_slide(project_id, 1)["elements"] if item["type"] == "image"]
    assert len(images) == 1 and images[0]["src"].startswith(f"/media/assets/{project_id}/")
    assert (root / "assets" / Path(images[0]["src"]).name).is_file()
    assert np.all(cv2.imread(str(root / "backgrounds" / "page_1.png")) == 255)
    assert cv2.imread(str(root / "reconstructed_preview.png"))[75, 80, 0] < 255
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    renderer.PPTXRenderer()._add_element(slide, images[0], 0.02, 0.02)
    assert len(slide.shapes) == 1


def test_debug_artifact_is_retrievable_from_existing_project_route(tmp_path, monkeypatch):
    store = ProjectStore(tmp_path)
    project_id = store.create("debug-audit")["id"]
    root = tmp_path / project_id
    cv2.imwrite(str(root / "objectization_debug.png"), np.full((20, 30, 3), 255, np.uint8))
    (root / "objectization_audit.json").write_text(json.dumps({"whiteBackground": True}), encoding="utf-8")
    monkeypatch.setattr(api_main, "store", store)
    monkeypatch.setattr(api_main, "OUTPUTS_DIR", tmp_path)
    client = TestClient(api_main.app)
    assert client.get(f"/api/projects/{project_id}/artifacts/objectization_debug.png").status_code == 200
    assert client.get(f"/api/projects/{project_id}/artifacts/objectization_audit.json").json()["whiteBackground"] is True


@pytest.mark.parametrize("variant", ["original", "lightened", "small_badge"])
def test_complex_fixture_white_objectized_end_to_end(tmp_path, variant):
    source = cv2.imread(str(FIXTURE))
    assert source is not None
    if variant == "lightened":
        source = cv2.addWeighted(source, 0.65, np.full_like(source, 255), 0.35, 0)
    elif variant == "small_badge":
        cv2.circle(source, (570, 70), 22, (35, 100, 190), -1, cv2.LINE_AA)
        cv2.circle(source, (570, 70), 7, (255, 255, 255), -1, cv2.LINE_AA)
    source_path = tmp_path / "source.png"
    background = tmp_path / "background.png"
    preview = tmp_path / "preview.png"
    cv2.imwrite(str(source_path), source)
    layout = {"slide": {"width": source.shape[1], "height": source.shape[0]}, "elements": []}
    stats = objectize_on_white(source_path, background, layout, tmp_path / "assets", "fixture", 1)
    assert stats["whiteObjectAssets"] + stats["whiteObjectShapes"] > 0
    for item in layout["elements"]:
        if item.get("type") == "image":
            item["src"] = str(tmp_path / "assets" / Path(item["src"]).name)
    render_preview(background, layout, preview)
    report = audit_objectization(source_path, background, preview, layout, tmp_path / "debug.png")
    assert report["whiteBackground"] is True
    assert report["backgroundResidualRegions"] == 0
    assert all(item["owner"] in {"editable_text", "movable_image", "native_shape"} for item in report["ownerRegions"])
    assert (tmp_path / "debug.png").is_file()
