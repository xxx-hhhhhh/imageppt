from __future__ import annotations

import cv2
import numpy as np

from app.services.reconstruction.white_objectization import objectize_on_white


def test_white_surface_extracts_unowned_visual_and_keeps_text_editable(tmp_path):
    source = np.full((240, 400, 3), 255, np.uint8)
    cv2.rectangle(source, (30, 40), (105, 115), (30, 70, 180), -1)
    cv2.putText(source, "Title", (180, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    source_path = tmp_path / "source.png"
    background_path = tmp_path / "backgrounds" / "page_1.png"
    cv2.imwrite(str(source_path), source)
    layout = {"slide": {"width": 400, "height": 240}, "elements": [{"id": "title", "type": "text", "x": 175, "y": 45, "width": 100, "height": 40, "text": "Title", "metadata": {"rawOCRBBox": [175, 45, 275, 85]}}]}

    stats = objectize_on_white(source_path, background_path, layout, tmp_path / "assets", "demo", 1)

    assert stats["whiteObjectAssets"] >= 1
    assert np.all(cv2.imread(str(background_path)) == 255)
    assert layout["metadata"]["reconstructionSurfaceMode"] == "white_objectized"
    assert layout["elements"][0]["type"] == "text"
    assets = [item for item in layout["elements"] if item["type"] == "image"]
    assert any(item["x"] <= 30 and item["x"] + item["width"] >= 105 for item in assets)
    assert all(not (item["x"] >= 170 and item["x"] < 280) for item in assets)
