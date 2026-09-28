from pathlib import Path

import cv2
import numpy as np

from app.services.reconstruction.text_erasure import count_text_ghosting, erase_editable_text_sources


def test_editable_line_erases_source_from_background_and_asset(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "backgrounds").mkdir(parents=True)
    (root / "assets").mkdir()
    background_path = root / "backgrounds" / "page_1.png"
    asset_path = root / "assets" / "module.png"
    source = np.full((80, 180, 3), 255, np.uint8)
    cv2.putText(source, "LABEL", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(str(background_path), source)
    cv2.imwrite(str(asset_path), source)
    layout = {"elements": [
        {"id": "text_001", "type": "text", "text": "LABEL", "metadata": {"rawOCRBBox": [16, 20, 105, 52]}},
        {"id": "visual", "type": "image", "x": 0, "y": 0, "width": 180, "height": 80, "src": str(asset_path), "metadata": {}},
    ]}
    source_path = root / "source.png"
    cv2.imwrite(str(source_path), source)
    assert count_text_ghosting(source_path, background_path, layout) == 1
    stats = erase_editable_text_sources(background_path, layout)
    assert stats == {"backgroundTextErased": 1, "assetTextErased": 1}
    assert layout["elements"][0]["metadata"]["textOwner"] == "text_001"
    assert layout["elements"][1]["metadata"]["editableTextIds"] == ["text_001"]
    for path in (background_path, asset_path):
        cleaned = cv2.imread(str(path))
        assert cleaned[30:48, 22:100].mean() > source[30:48, 22:100].mean() + 20
    assert count_text_ghosting(source_path, background_path, layout) == 0
    assert "ghostingDetected" not in layout["elements"][0]["metadata"]


def test_residual_image_clears_alpha_under_text_without_raw_ocr_box(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "backgrounds").mkdir(parents=True)
    (root / "assets").mkdir()
    source = np.full((100, 220, 3), 255, np.uint8)
    cv2.putText(source, "LEGEND", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    source_path = root / "source.png"
    background_path = root / "backgrounds" / "page_1.png"
    asset_path = root / "assets" / "residual.png"
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(background_path), np.full_like(source, 255))
    cv2.imwrite(str(asset_path), np.dstack((source, np.full(source.shape[:2], 255, np.uint8))))
    layout = {"elements": [
        {"id": "legend", "type": "text", "text": "LEGEND", "x": 20, "y": 38, "width": 120, "height": 35, "metadata": {}},
        {"id": "residual", "type": "image", "x": 0, "y": 0, "width": 220, "height": 100, "src": str(asset_path), "metadata": {"layerRole": "residual"}},
    ]}

    assert count_text_ghosting(source_path, background_path, layout) == 1
    stats = erase_editable_text_sources(background_path, layout)

    assert stats == {"backgroundTextErased": 1, "assetTextErased": 1}
    cleaned = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    assert np.max(cleaned[42:69, 25:135, 3]) == 0
    assert count_text_ghosting(source_path, background_path, layout) == 0
