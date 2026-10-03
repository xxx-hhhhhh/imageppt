import cv2
import numpy as np
import pytest

from app.services.reconstruction.asset_metrics import measure_movable_assets


def _files(tmp_path):
    source = np.full((100, 200, 3), 255, np.uint8)
    source[20:60, 20:60] = (60, 100, 190)
    source[20:60, 130:170] = (50, 170, 70)
    src = tmp_path / "source.png"
    bg = tmp_path / "background.png"
    cv2.imwrite(str(src), source)
    cv2.imwrite(str(bg), np.full_like(source, 255))
    return source, src, bg


def test_no_plan_does_not_claim_missing_visuals_are_fully_movable(tmp_path):
    _, source, background = _files(tmp_path)
    assert measure_movable_assets(source, background, {"elements": []}, {})["movableVisualCoverage"] == 0


def test_transparent_holes_are_not_covered_by_image_bounding_box(tmp_path):
    pixels, source, background = _files(tmp_path)
    alpha = np.zeros(pixels.shape[:2], np.uint8)
    alpha[20:60, 20:60] = 255
    path = tmp_path / "asset.png"
    cv2.imwrite(str(path), np.dstack((pixels, alpha)))
    layout = {"elements": [{"type": "image", "x": 0, "y": 0, "width": 200, "height": 100, "src": str(path)}]}
    plan = {"modules": [{"moduleId": "detected_visual_both", "bboxPixels": [0, 0, 200, 100]}]}
    assert measure_movable_assets(source, background, layout, plan)["movableVisualCoverage"] == .5


def test_opaque_white_asset_does_not_cover_erased_pale_plate(tmp_path):
    source = np.full((100, 200, 3), 255, np.uint8)
    source[20:80, 20:180] = (247, 245, 249)
    src, bg, asset = [tmp_path / name for name in ("source.png", "background.png", "asset.png")]
    cv2.imwrite(str(src), source)
    for path in (bg, asset):
        cv2.imwrite(str(path), np.full_like(source, 255))
    layout = {"elements": [{"type": "image", "x": 0, "y": 0, "width": 200, "height": 100, "src": str(asset)}]}
    assert measure_movable_assets(src, bg, layout, {})["movableVisualCoverage"] == 0


def test_native_plate_does_not_claim_missing_chart_inside_it(tmp_path):
    pixels, source, background = _files(tmp_path)
    layout = {"elements": [{"type": "rectangle", "x": 0, "y": 0, "width": 200, "height": 100,
                            "style": {"fill": "#FFFFFF", "stroke": "#BE643C"}}]}
    assert measure_movable_assets(source, background, layout, {})["movableVisualCoverage"] == 0


def test_partial_plan_does_not_hide_unplanned_lost_artwork(tmp_path):
    pixels, source, background = _files(tmp_path)
    path = tmp_path / "asset.png"
    cv2.imwrite(str(path), pixels[20:60, 20:60])
    layout = {"elements": [{"type": "image", "x": 20, "y": 20, "width": 40, "height": 40, "src": str(path)}]}
    plan = {"modules": [{"moduleId": "detected_visual_left_only", "bboxPixels": [20, 20, 60, 60]}]}
    assert measure_movable_assets(source, background, layout, plan)["movableVisualCoverage"] == .5


@pytest.mark.parametrize("with_plate", [False, True])
def test_bold_editable_ink_does_not_hide_missing_support(tmp_path, with_plate):
    source = np.full((100, 200, 3), 255, np.uint8)
    if with_plate:
        source[25:80, 20:150] = (247, 245, 249)
    cv2.putText(source, "M", (55, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (30, 60, 90), 8, cv2.LINE_AA)
    src, bg = tmp_path / "source.png", tmp_path / "background.png"
    cv2.imwrite(str(src), source)
    cv2.imwrite(str(bg), np.full_like(source, 255))
    layout = {"elements": [{"type": "text", "text": "M", "x": 45, "y": 22, "width": 64, "height": 55,
                            "metadata": {"rawOCRBBox": [45, 22, 109, 77]}, "style": {"color": "#5A3C1E"}}]}
    metrics = measure_movable_assets(src, bg, layout, {})
    assert metrics["movableVisualCoverage"] == (0 if with_plate else 1)
    if with_plate:
        layout["elements"].append({"type": "rectangle", "x": 20, "y": 25, "width": 130, "height": 55,
                                   "style": {"fill": "#F9F5F7", "strokeWidth": 0}})
        assert measure_movable_assets(src, bg, layout, {})["movableVisualCoverage"] == 1
