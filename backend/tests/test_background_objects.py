from pathlib import Path

import cv2
import numpy as np

from app.services.reconstruction.background_objects import objectize_background_regions
from app.services.visual_qa.analyzer import render_preview


def test_background_cutouts_preserve_pixels_and_existing_transparent_owner(tmp_path):
    source = np.full((180, 320, 3), 255, np.uint8)
    cv2.rectangle(source, (5, 5), (314, 174), (50, 100, 190), 3)
    cv2.circle(source, (160, 90), 20, (60, 180, 90), -1)
    background = tmp_path / "background.png"
    original = tmp_path / "source.png"
    cv2.imwrite(str(original), source)
    cv2.imwrite(str(background), source)
    assets = tmp_path / "assets"
    assets.mkdir()
    icon = np.zeros((40, 40, 4), np.uint8)
    cv2.circle(icon, (20, 20), 18, (60, 180, 90, 255), -1)
    icon_path = assets / "icon.png"
    cv2.imwrite(str(icon_path), icon)
    before = icon_path.read_bytes()
    layout = {"slide": {"width": 320, "height": 180}, "elements": [
        {"id": "icon", "type": "image", "x": 140, "y": 70, "width": 40, "height": 40,
         "src": str(icon_path), "zIndex": 5}]}
    preview = tmp_path / "before.png"
    render_preview(background, layout, preview)
    shown = cv2.imread(str(preview))
    created = objectize_background_regions(original, background, layout,
                                           [{"bbox": [0, 0, 320, 180]}], assets, "demo", 1, 1)
    assert created
    assert icon_path.read_bytes() == before
    for item in created:
        item["src"] = str(assets / Path(item["src"]).name)
        assert item["metadata"]["backgroundSeparated"]
    render_preview(background, layout, tmp_path / "after.png")
    assert np.array_equal(cv2.imread(str(tmp_path / "after.png")), shown)
    assert np.all(cv2.imread(str(background))[5, 25] == 255)
    assert np.all(cv2.imread(str(background))[90, 160] == 255)
    assert layout["elements"][0]["metadata"]["backgroundDuplicateClearedPixels"] > 0


def test_missing_staged_asset_leaves_scene_and_background_unchanged(tmp_path, monkeypatch):
    from app.services.reconstruction import background_objects

    source = np.full((80, 150, 3), 255, np.uint8)
    source[20:40, 30:50] = (70, 110, 210)
    path = tmp_path / "background.png"
    cv2.imwrite(str(path), source)
    before = path.read_bytes()
    layout = {"elements": []}
    fake = {"id": "missing", "src": "/media/assets/demo/missing.png", "x": 30, "y": 20,
            "width": 20, "height": 20, "metadata": {}}
    monkeypatch.setattr(background_objects, "extract_residual_objects", lambda *a, **k: ([fake], {}))
    assert objectize_background_regions(path, path, layout, [{"bbox": [0, 0, 150, 80]}],
                                       tmp_path, "demo", 1, 1) == []
    assert path.read_bytes() == before
    assert not layout["elements"]


def test_partial_old_text_is_cleaned_before_plate_and_decor_are_objectized(tmp_path):
    source = np.full((160, 220, 3), 255, np.uint8)
    source[15:145, 20:200] = (247, 242, 249)
    ink = (85, 30, 10)
    cv2.putText(source, "A", (42, 62), cv2.FONT_HERSHEY_SIMPLEX, .6, ink, 2, cv2.LINE_AA)
    cv2.putText(source, "B", (42, 103), cv2.FONT_HERSHEY_SIMPLEX, .6, ink, 2, cv2.LINE_AA)
    cv2.circle(source, (140, 75), 16, (50, 170, 70), -1)
    path = tmp_path / "background.png"
    cv2.imwrite(str(path), source)
    text = {"id": "vertical", "type": "text", "text": "AB", "x": 37, "y": 42,
            "width": 28, "height": 67, "style": {"color": "#0A1E55"},
            "metadata": {"rawOCRBBox": [37, 42, 65, 109]}}
    layout = {"elements": [text]}
    created = objectize_background_regions(path, path, layout, [{"bbox": [15, 10, 205, 150]}],
                                           tmp_path / "assets", "demo", 1, 1)
    images = [item for item in created if item.get("type") == "image"]
    assert images
    retained_plate = retained_green = retained_ink = 0
    for item in images:
        cutout = cv2.imread(str(tmp_path / "assets" / Path(item["src"]).name), -1)
        visible = cutout[:, :, 3] > 0
        colors = cutout[:, :, :3].astype(np.int16)
        retained_plate += np.count_nonzero(visible & (np.max(np.abs(colors - [247, 242, 249]), axis=2) <= 2))
        retained_green += np.count_nonzero(visible & np.all(colors == [50, 170, 70], axis=2))
        retained_ink += np.count_nonzero(visible & (np.max(np.abs(colors - ink), axis=2) <= 38))
    assert retained_plate > 15000
    assert retained_green > 500
    assert retained_ink == 0
    assert layout["elements"][0] == text


def test_background_write_failure_keeps_original_and_discards_new_assets(tmp_path, monkeypatch):
    source = np.full((80, 150, 3), 255, np.uint8)
    source[20:40, 30:50] = (70, 110, 210)
    path = tmp_path / "background.png"
    cv2.imwrite(str(path), source)
    before = path.read_bytes()
    write = cv2.imwrite
    monkeypatch.setattr(cv2, "imwrite", lambda filename, pixels: False if Path(filename).name.startswith(".") else write(filename, pixels))
    layout = {"elements": []}
    assets = tmp_path / "assets"
    assert not objectize_background_regions(path, path, layout, [{"bbox": [0, 0, 150, 80]}], assets, "demo", 1, 1)
    assert path.read_bytes() == before
    assert not layout["elements"]
    assert not list(assets.glob("*.png"))
