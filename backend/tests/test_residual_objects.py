from __future__ import annotations

import cv2
import numpy as np
from pptx import Presentation

from app.services.pptx import renderer as renderer_module
from app.services.reconstruction.residual_objects import extract_residual_objects


def test_disconnected_visual_parts_group_into_movable_ppt_images(tmp_path, monkeypatch):
    image = np.full((300, 600, 3), 255, np.uint8)
    cv2.ellipse(image, (125, 245), (65, 24), 0, 0, 360, (30, 35, 210), -1)
    cv2.ellipse(image, (215, 245), (20, 18), 0, 0, 360, (32, 38, 207), -1)
    cv2.rectangle(image, (370, 75), (450, 185), (65, 65, 65), -1)
    cv2.rectangle(image, (456, 100), (478, 185), (68, 68, 68), -1)
    cv2.circle(image, (530, 40), 9, (190, 115, 20), -1)
    occupied = np.zeros(image.shape[:2], np.uint8)
    project_id = "demo"
    assets, stats = extract_residual_objects(image, occupied, tmp_path / project_id / "assets", project_id, 1)

    assert stats["residualObjectsCount"] == len(assets) == 3
    assert stats["residualCoverageArea"] > 0
    assert stats["residualObjectizationRate"] > 0.95
    assert all(item["width"] * item["height"] < image.shape[0] * image.shape[1] * 0.4 for item in assets)
    monkeypatch.setattr(renderer_module, "OUTPUTS_DIR", tmp_path)
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    for item in assets:
        renderer_module.PPTXRenderer()._add_element(slide, item, 0.01, 0.01)
    assert len(slide.shapes) == len(assets)
    assert all(shape.shape_type == 13 for shape in slide.shapes)  # separate pictures


def test_residual_detection_respects_owners_and_colored_background(tmp_path):
    image = np.full((180, 320, 3), (225, 235, 245), np.uint8)
    cv2.rectangle(image, (40, 45), (85, 90), (35, 60, 165), -1)
    cv2.rectangle(image, (190, 50), (245, 110), (25, 90, 160), -1)
    occupied = np.zeros(image.shape[:2], np.uint8)
    occupied[45:91, 40:86] = 255

    assets, stats = extract_residual_objects(image, occupied, tmp_path / "demo" / "assets", "demo", 1)

    assert stats["residualObjectsCount"] == 1
    assert assets[0]["x"] > 180
    assert stats["residualObjectizationRate"] > 0.95


def test_page_sized_environment_is_not_repackaged_as_background_image(tmp_path):
    image = np.full((180, 320, 3), 255, np.uint8)
    cv2.rectangle(image, (5, 5), (314, 174), (20, 70, 100), -1)

    assets, stats = extract_residual_objects(image, np.zeros(image.shape[:2], np.uint8), tmp_path / "demo" / "assets", "demo", 1)

    assert assets == []
    assert stats["residualObjectsCount"] == 0
    assert stats["residualObjectizationRate"] == 0
