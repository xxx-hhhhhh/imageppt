from __future__ import annotations

from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

from app.services.reconstruction.layered_background import separate_foreground
from app.services.reconstruction.planner import AIReconstructionPlanner
from app.services.visual_qa.analyzer import render_preview


def test_cutout_removes_duplicate_pixels_from_background_and_stays_movable(tmp_path: Path) -> None:
    image = np.full((300, 400, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (200, 200), (10, 70, 210), -1)
    source = tmp_path / "source.png"
    background = tmp_path / "background.png"
    cv2.imwrite(str(source), image)
    cv2.imwrite(str(background), image)
    scene = {"canvas": {"width": 400, "height": 300}, "vision": {"aiUsed": True, "reconstructionPlan": {"modules": [{
        "id": "visual", "role": "illustration", "reconstructionStrategy": "cutout_image",
        "bbox": {"left": 0.25, "top": 1 / 3, "width": 0.25, "height": 1 / 3}, "confidence": 0.95,
    }]}}, "elements": []}
    stats = AIReconstructionPlanner().apply(scene, source, tmp_path / "assets", "project", 1)
    assert stats["cutoutImages"] == 1
    asset = next(item for item in scene["elements"] if item["type"] == "image")
    standalone = tmp_path / "cutout.png"
    shutil.copy2(tmp_path / "assets" / f"{asset['id']}.png", standalone)
    asset["src"] = str(standalone)
    layout_asset = {"id": asset["id"], "type": "image", "x": 100, "y": 100, "width": 100, "height": 100, "zIndex": 2, "src": asset["src"], "metadata": asset["metadata"]}
    assert separate_foreground(background, [layout_asset]) == 1
    cleaned = cv2.imread(str(background))
    assert not np.array_equal(cleaned[150, 150], image[150, 150])
    assert layout_asset["metadata"]["backgroundSeparated"] is True
    preview = tmp_path / "preview.png"
    render_preview(background, {"slide": {"width": 400, "height": 300}, "elements": [layout_asset]}, preview)
    composed = cv2.imread(str(preview))
    np.testing.assert_array_equal(composed[150, 150], image[150, 150])
    layout_asset["x"] = 240
    moved = tmp_path / "moved.png"
    render_preview(background, {"slide": {"width": 400, "height": 300}, "elements": [layout_asset]}, moved)
    moved_image = cv2.imread(str(moved))
    assert not np.array_equal(moved_image[150, 150], image[150, 150])
    np.testing.assert_array_equal(moved_image[150, 290], image[150, 150])


def test_plan_keeps_text_editable_and_accepts_background_and_native_shape(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (400, 300), "white").save(source)
    scene = {"canvas": {"width": 400, "height": 300}, "vision": {"aiUsed": True, "reconstructionPlan": {"modules": [
        {"id": "base", "reconstructionStrategy": "background", "bbox": {"left": 0, "top": 0, "width": 1, "height": 1}, "confidence": 0.9},
        {"id": "geometry", "reconstructionStrategy": "native_shape", "bbox": {"left": 0.1, "top": 0.3, "width": 0.4, "height": 0.4}, "confidence": 0.95},
    ]}}, "elements": [
        {"id": "shape", "type": "rectangle", "bbox": {"left": 50, "top": 100, "width": 100, "height": 50}, "confidence": 0.95, "metadata": {}},
        {"id": "text", "type": "text", "bbox": {"left": 60, "top": 110, "width": 80, "height": 20}, "text": "Edit me", "confidence": 0.9, "metadata": {}},
    ]}
    stats = AIReconstructionPlanner().apply(scene, source, tmp_path / "assets", "project", 1)
    assert stats["plannedModules"] == 2
    assert stats["nativeShapesPlanned"] == 1
    assert scene["elements"][0]["metadata"]["reconstructionStrategy"] == "native_shape"
    assert scene["elements"][1]["metadata"].get("suppressed") is not True
    assert not [item for item in scene["elements"] if item.get("type") == "image"]


def test_ignore_module_suppresses_its_declared_members(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (400, 300), "white").save(source)
    scene = {"canvas": {"width": 400, "height": 300}, "vision": {"aiUsed": True, "reconstructionPlan": {"modules": [{
        "id": "artifact", "reconstructionStrategy": "ignore", "bbox": {"left": 0.1, "top": 0.1, "width": 0.3, "height": 0.3},
        "memberIds": ["duplicate"], "confidence": 0.95,
    }]}}, "elements": [{"id": "duplicate", "type": "text", "text": "duplicate", "bbox": {"left": 50, "top": 50, "width": 80, "height": 20}, "metadata": {}}]}
    AIReconstructionPlanner().apply(scene, source, tmp_path / "assets", "project", 1)
    assert scene["elements"][0]["metadata"]["suppressRender"] is True
