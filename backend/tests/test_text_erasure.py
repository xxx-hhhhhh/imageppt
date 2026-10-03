from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.reconstruction.text_erasure import count_text_ghosting, erase_editable_text_sources


def test_dark_text_erasure_preserves_pale_gradient_inside_ocr_box(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "assets").mkdir(parents=True)
    (root / "backgrounds").mkdir()
    asset_path = root / "assets" / "gradient.png"
    background_path = root / "backgrounds" / "page_1.png"
    surface = np.full((100, 240, 3), 255, np.uint8)
    surface[:, :, 0] = np.linspace(225, 250, 240).astype(np.uint8)
    surface[:, :, 1] = np.linspace(235, 252, 240).astype(np.uint8)
    surface[:, :, 2] = np.linspace(240, 254, 240).astype(np.uint8)
    source = surface.copy()
    cv2.putText(source, "LABEL", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 20, 20), 2)
    cv2.imwrite(str(asset_path), np.dstack((source, np.full((100, 240), 255, np.uint8))))
    cv2.imwrite(str(background_path), np.full_like(surface, 255))
    layout = {"elements": [
        {"id": "label", "type": "text", "text": "LABEL", "style": {"color": "#141414"},
         "metadata": {"rawOCRBBox": [20, 30, 205, 78]}},
        {"id": "plate", "type": "image", "x": 0, "y": 0, "width": 240, "height": 100,
         "src": str(asset_path), "metadata": {"layerRole": "residual"}},
    ]}
    erase_editable_text_sources(background_path, layout, clean_background=False)
    cleaned = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    glyphs = np.max(np.abs(source.astype(int) - surface.astype(int)), axis=2) > 30
    protected = cv2.dilate(np.uint8(glyphs), np.ones((3, 3), np.uint8)) == 0
    assert np.array_equal(cleaned[:, :, :3][protected], source[protected])
    assert cleaned[:, :, :3][glyphs].mean() > 220
    assert np.all(cleaned[:, :, 3] == 255)


def test_transparent_glyph_rgb_does_not_trigger_destructive_reclean(tmp_path: Path) -> None:
    source = np.full((100, 240, 3), 255, np.uint8)
    cv2.putText(source, "LABEL", (40, 65), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
    alpha = np.full(source.shape[:2], 255, np.uint8)
    alpha[np.any(source < 250, axis=2)] = 0
    source_path, background_path, asset_path = tmp_path / "source.png", tmp_path / "background.png", tmp_path / "asset.png"
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(background_path), np.full_like(source, 255))
    cv2.imwrite(str(asset_path), np.dstack((source, alpha)))
    layout = {"elements": [
        {"id": "label", "type": "text", "text": "LABEL", "metadata": {"rawOCRBBox": [20, 30, 200, 80]}},
        {"id": "asset", "type": "image", "x": 0, "y": 0, "width": 240, "height": 100, "src": str(asset_path)},
    ]}
    assert count_text_ghosting(source_path, background_path, layout) == 0
    assert not layout["elements"][0]["metadata"].get("ghostingDetected")


def test_sparse_visible_glyph_is_detected_despite_low_average_alpha(tmp_path: Path) -> None:
    source = np.full((100, 240, 3), 255, np.uint8)
    cv2.putText(source, "LABEL", (40, 65), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
    alpha = np.uint8(np.any(source < 250, axis=2)) * 255
    source_path, background_path, asset_path = tmp_path / "source.png", tmp_path / "background.png", tmp_path / "asset.png"
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(background_path), np.full_like(source, 255))
    cv2.imwrite(str(asset_path), np.dstack((source, alpha)))
    layout = {"elements": [
        {"id": "label", "type": "text", "text": "LABEL", "metadata": {"rawOCRBBox": [20, 30, 200, 80]}},
        {"id": "asset", "type": "image", "x": 0, "y": 0, "width": 240, "height": 100, "src": str(asset_path)},
    ]}
    assert alpha.mean() < 128
    assert count_text_ghosting(source_path, background_path, layout) == 1


@pytest.mark.parametrize("transparent_glyphs", [False, True])
@pytest.mark.parametrize("narrow_fragments", [False, True])
def test_split_visual_assets_do_not_hide_partial_text_ghosts(tmp_path: Path, transparent_glyphs, narrow_fragments) -> None:
    source = np.full((100, 240, 3), 255, np.uint8)
    cv2.putText(source, "LEFT", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)
    cv2.putText(source, "RIGHT", (130, 65), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)
    source_path, background_path = tmp_path / "source.png", tmp_path / "background.png"
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(background_path), np.full_like(source, 255))
    layout = {"elements": [{"id": "line", "type": "text", "text": "LEFT RIGHT",
                             "metadata": {"rawOCRBBox": [10, 25, 230, 80]}}]}
    width = 40 if narrow_fragments else 120
    for index, x in enumerate((20, 130) if narrow_fragments else (0, 120)):
        rgb = source[:, x:x+width]
        alpha = np.full(rgb.shape[:2], 255, np.uint8)
        if transparent_glyphs:
            alpha[np.any(rgb < 250, axis=2)] = 0
        asset = tmp_path / f"part_{index}.png"
        cv2.imwrite(str(asset), np.dstack((rgb, alpha)))
        layout["elements"].append({"id": f"part_{index}", "type": "image", "x": x, "y": 0,
                                   "width": width, "height": 100, "src": str(asset)})
    # Neither image contains the complete OCR box. Count the line only once.
    assert count_text_ghosting(source_path, background_path, layout) == (0 if transparent_glyphs else 1)
    if not transparent_glyphs:
        assert layout["elements"][0]["metadata"]["ghostingAssetIds"] == ["part_0", "part_1"]
        erase_editable_text_sources(background_path, layout, clean_background=False)
        assert count_text_ghosting(source_path, background_path, layout) == 0


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


def test_residual_text_box_does_not_pull_dark_visual_into_pale_gap(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "backgrounds").mkdir(parents=True)
    (root / "assets").mkdir()
    background_path = root / "backgrounds" / "page_1.png"
    asset_path = root / "assets" / "samples.png"
    source = np.full((90, 180, 3), (250, 250, 250), np.uint8)
    source[20:43, 12:168] = (25, 40, 70)
    cv2.putText(source, "K=5", (35, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 55, 90), 1)
    cv2.imwrite(str(background_path), np.full_like(source, 255))
    cv2.imwrite(str(asset_path), np.dstack((source, np.full(source.shape[:2], 255, np.uint8))))
    layout = {"elements": [
        {"id": "label", "type": "text", "text": "K=5", "metadata": {"rawOCRBBox": [30, 40, 100, 70]}},
        {"id": "samples", "type": "image", "x": 0, "y": 0, "width": 180, "height": 90,
         "src": str(asset_path), "metadata": {"layerRole": "residual"}},
    ]}

    erase_editable_text_sources(background_path, layout, clean_background=False)

    cleaned = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    assert np.min(cleaned[46:50, 35:100, :3]) >= 240
    assert np.max(np.abs(cleaned[25, 120, :3].astype(int) - source[25, 120].astype(int))) <= 2
    assert np.mean(cleaned[51:65, 35:80, :3]) > np.mean(source[51:65, 35:80]) + 4


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


def test_residual_ribbon_uses_local_color_when_asset_has_large_white_exterior(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "backgrounds").mkdir(parents=True)
    (root / "assets").mkdir()
    source = np.full((140, 300, 3), 255, np.uint8)
    source[40:100, 20:280] = (35, 60, 185)
    cv2.putText(source, "LABEL", (90, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 248, 255), 2)
    background_path = root / "backgrounds" / "page_1.png"
    asset_path = root / "assets" / "wide_asset.png"
    cv2.imwrite(str(background_path), np.full_like(source, 255))
    cv2.imwrite(str(asset_path), np.dstack((source, np.full(source.shape[:2], 255, np.uint8))))
    layout = {"elements": [
        {"id": "label", "type": "text", "text": "LABEL", "metadata": {"rawOCRBBox": [82, 56, 194, 86]}},
        {"id": "ribbon", "type": "image", "x": 0, "y": 0, "width": 300, "height": 140,
         "src": str(asset_path), "metadata": {"layerRole": "residual"}},
    ]}

    erase_editable_text_sources(background_path, layout, clean_background=False)

    cleaned = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    assert np.max(np.abs(cleaned[70, 110, :3].astype(int) - np.array([35, 60, 185]))) < 25
    assert np.max(np.abs(cleaned[70, 150, :3].astype(int) - np.array([35, 60, 185]))) < 25
    assert np.all(cleaned[60:85, 85:195, 3] > 220)


def test_residual_asset_repairs_each_text_line_on_its_own_colored_surface(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "backgrounds").mkdir(parents=True)
    (root / "assets").mkdir()
    source = np.full((150, 300, 3), 255, np.uint8)
    source[20:65, 20:280] = (35, 60, 185)
    source[65:130, 20:280] = (240, 245, 250)
    cv2.putText(source, "TITLE", (90, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 255), 2)
    cv2.putText(source, "BODY", (90, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 25), 2)
    background_path = root / "backgrounds" / "page_1.png"
    asset_path = root / "assets" / "card.png"
    cv2.imwrite(str(background_path), np.full_like(source, 255))
    cv2.imwrite(str(asset_path), np.dstack((source, np.full(source.shape[:2], 255, np.uint8))))
    layout = {"elements": [
        {"id": "title", "type": "text", "text": "TITLE", "metadata": {"rawOCRBBox": [82, 29, 170, 59]}},
        {"id": "body", "type": "text", "text": "BODY", "metadata": {"rawOCRBBox": [82, 81, 170, 111]}},
        {"id": "card", "type": "image", "x": 0, "y": 0, "width": 300, "height": 150,
         "src": str(asset_path), "metadata": {"layerRole": "residual"}},
    ]}

    erase_editable_text_sources(background_path, layout, clean_background=False)

    cleaned = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    assert np.max(np.abs(cleaned[42, 105, :3].astype(int) - np.array([35, 60, 185]))) < 25
    assert np.max(np.abs(cleaned[94, 105, :3].astype(int) - np.array([240, 245, 250]))) < 20


def test_residual_text_erasure_preserves_gradient_between_glyphs(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "backgrounds").mkdir(parents=True)
    (root / "assets").mkdir()
    base = np.full((110, 320, 3), 255, np.uint8)
    for x in range(20, 300):
        base[30:80, x] = (30 + (x - 20) // 5, 70 + (x - 20) // 4, 175 + (x - 20) // 7)
    source = base.copy()
    cv2.putText(source, "GRADIENT", (78, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    background_path = root / "backgrounds" / "page_1.png"
    asset_path = root / "assets" / "gradient.png"
    cv2.imwrite(str(background_path), np.full_like(source, 255))
    cv2.imwrite(str(asset_path), np.dstack((source, np.full(source.shape[:2], 255, np.uint8))))
    layout = {"elements": [
        {"id": "title", "type": "text", "text": "GRADIENT",
         "style": {"color": "#FFFFFF"}, "metadata": {"rawOCRBBox": [72, 40, 220, 72]}},
        {"id": "gradient", "type": "image", "x": 0, "y": 0, "width": 320, "height": 110,
         "src": str(asset_path), "metadata": {"layerRole": "residual"}},
    ]}

    erase_editable_text_sources(background_path, layout, clean_background=False)

    cleaned = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)[:, :, :3]
    region = np.zeros(base.shape[:2], np.bool_)
    region[42:70, 76:218] = True
    original_glyphs = np.any(source != base, axis=2) & region
    undamaged_plate = region & ~original_glyphs
    difference = np.max(np.abs(cleaned.astype(int) - base.astype(int)), axis=2)
    assert np.mean(difference[undamaged_plate]) < 5
    assert np.mean(difference[original_glyphs]) < 35


def test_residual_gradient_support_fills_ocr_hole_without_flat_patch(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "backgrounds").mkdir(parents=True)
    (root / "assets").mkdir()
    base = np.full((110, 320, 3), 255, np.uint8)
    for x in range(20, 300):
        base[30:80, x] = (30 + (x - 20) // 5, 70 + (x - 20) // 4, 175 + (x - 20) // 7)
    source = base.copy()
    cv2.putText(source, "GRADIENT", (78, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    alpha = np.full(source.shape[:2], 255, np.uint8)
    alpha[38:75, 70:224] = 0  # OCR ownership left a transparent hole in the movable plate.
    background_path = root / "backgrounds" / "page_1.png"
    asset_path = root / "assets" / "gradient.png"
    cv2.imwrite(str(background_path), np.full_like(source, 255))
    cv2.imwrite(str(asset_path), np.dstack((source, alpha)))
    layout = {"elements": [
        {"id": "title", "type": "text", "text": "GRADIENT",
         "style": {"color": "#FFFFFF"}, "metadata": {"rawOCRBBox": [72, 40, 220, 72]}},
        {"id": "gradient", "type": "image", "x": 0, "y": 0, "width": 320, "height": 110,
         "src": str(asset_path), "metadata": {"layerRole": "residual"}},
    ]}

    erase_editable_text_sources(background_path, layout, clean_background=False)

    cleaned = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    region = np.zeros(base.shape[:2], np.bool_)
    region[42:70, 76:218] = True
    original_glyphs = np.any(source != base, axis=2) & region
    undamaged_plate = region & ~original_glyphs
    difference = np.max(np.abs(cleaned[:, :, :3].astype(int) - base.astype(int)), axis=2)
    assert np.min(cleaned[region, 3]) > 220
    assert np.mean(difference[undamaged_plate]) < 5
    assert np.mean(difference[original_glyphs]) < 35


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
    alpha[25:75, 20:140] = 255
    rgb[55:68, 60:73] = (30, 40, 60)
    alpha[55:68, 60:73] = 0
    cv2.imwrite(str(asset_path), np.dstack((rgb, alpha)))
    layout = {"elements": [
        {"id": "label", "type": "text", "text": "A", "x": 55, "y": 50, "width": 30, "height": 23},
        {"id": "strip", "type": "image", "x": 0, "y": 0, "width": 160, "height": 100,
         "src": str(asset_path), "metadata": {"layerRole": "residual"}},
    ]}

    erase_editable_text_sources(background_path, layout)

    cleaned = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    assert cleaned[61, 66, 3] > 220  # A glyph hole wider than the closing kernel is repaired.
    assert cleaned[78, 66, 3] == 0  # The strip outline stays at y=75.
