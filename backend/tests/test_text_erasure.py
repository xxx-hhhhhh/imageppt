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


def test_residual_image_repairs_text_without_punching_alpha_hole(tmp_path: Path) -> None:
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
    assert np.min(cleaned[42:69, 25:135, 3]) > 220
    assert np.min(cleaned[42:69, 25:135, :3]) > 230
    assert count_text_ghosting(source_path, background_path, layout) == 0


def test_residual_ribbon_preserves_color_beneath_editable_label(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "backgrounds").mkdir(parents=True)
    (root / "assets").mkdir()
    source = np.full((130, 300, 3), 255, np.uint8)
    cv2.rectangle(source, (15, 45), (285, 105), (35, 50, 185), -1)
    cv2.putText(source, "LABEL", (92, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 220, 250), 2)
    source_path = root / "source.png"
    background_path = root / "backgrounds" / "page_1.png"
    asset_path = root / "assets" / "ribbon.png"
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(background_path), np.full_like(source, 255))
    cv2.imwrite(str(asset_path), np.dstack((source[45:106, 15:286], np.full((61, 271), 255, np.uint8))))
    layout = {"elements": [
        {"id": "label", "type": "text", "text": "LABEL", "x": 90, "y": 58, "width": 110, "height": 36, "metadata": {}},
        {"id": "ribbon", "type": "image", "x": 15, "y": 45, "width": 271, "height": 61, "src": str(asset_path), "metadata": {"layerRole": "residual"}},
    ]}

    erase_editable_text_sources(background_path, layout)

    cleaned = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    assert cleaned[35, 120, 3] > 220
    assert np.max(np.abs(cleaned[35, 120, :3].astype(int) - np.array([35, 50, 185]))) < 25
    assert count_text_ghosting(source_path, background_path, layout) == 0


def test_flat_card_text_repair_avoids_nearby_icon_color_bleed(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "backgrounds").mkdir(parents=True)
    (root / "assets").mkdir()
    source = np.full((170, 260, 3), 255, np.uint8)
    cv2.rectangle(source, (25, 20), (230, 145), (244, 246, 251), -1)
    cv2.circle(source, (128, 48), 24, (35, 60, 190), -1)
    cv2.putText(source, "TITLE", (82, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    background_path = root / "backgrounds" / "page_1.png"
    asset_path = root / "assets" / "card.png"
    cv2.imwrite(str(background_path), np.full_like(source, 255))
    cv2.imwrite(str(asset_path), np.dstack((source[20:146, 25:231], np.full((126, 206), 255, np.uint8))))
    layout = {"elements": [
        {"id": "title", "type": "text", "text": "TITLE", "x": 80, "y": 78, "width": 98, "height": 35, "metadata": {}},
        {"id": "card", "type": "image", "x": 25, "y": 20, "width": 206, "height": 126, "src": str(asset_path), "metadata": {"layerRole": "residual"}},
    ]}

    erase_editable_text_sources(background_path, layout)

    cleaned = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    assert cleaned[76, 103, 3] > 220
    assert np.max(np.abs(cleaned[76, 103, :3].astype(int) - np.array([244, 246, 251]))) < 15


def test_residual_text_repair_does_not_extend_transparent_backplate(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "backgrounds").mkdir(parents=True)
    (root / "assets").mkdir()
    background_path = root / "backgrounds" / "page_1.png"
    asset_path = root / "assets" / "strip.png"
    cv2.imwrite(str(background_path), np.full((100, 160, 3), 255, np.uint8))
    rgb = np.full((100, 160, 3), (235, 240, 248), np.uint8)
    alpha = np.zeros((100, 160), np.uint8)
    alpha[30:70, 20:140] = 255
    rgb[63:68, 64:69] = (30, 40, 60)
    alpha[63:68, 64:69] = 0
    cv2.imwrite(str(asset_path), np.dstack((rgb, alpha)))
    layout = {"elements": [
        {"id": "label", "type": "text", "text": "A", "x": 58, "y": 58, "width": 25, "height": 20},
        {"id": "strip", "type": "image", "x": 0, "y": 0, "width": 160, "height": 100,
         "src": str(asset_path), "metadata": {"layerRole": "residual"}},
    ]}

    erase_editable_text_sources(background_path, layout)

    cleaned = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    assert cleaned[65, 66, 3] > 220  # The glyph hole is repaired.
    assert cleaned[73, 66, 3] == 0  # The strip outline stays at y=70.
