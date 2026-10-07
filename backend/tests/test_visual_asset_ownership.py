from pathlib import Path

import cv2
import numpy as np

from app.services.reconstruction.visual_asset_ownership import (
    count_duplicate_planned_visual_pixels, resolve_duplicate_contour_assets, transfer_planned_visual_pixels,
)
from app.services.reconstruction.white_objectization import layer_objectized_elements


def test_planned_visual_can_move_without_leaving_a_copy_in_residual(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    residual = np.full((100, 160, 4), 255, np.uint8)
    residual[:, :, :3] = (246, 245, 244)  # Retain the pale card surface.
    residual[30:65, 35:85, :3] = (35, 75, 120)
    planned = np.full((45, 60, 3), 255, np.uint8)
    planned[5:40, 5:55] = (35, 75, 120)
    original_path = assets / "residual.png"
    cv2.imwrite(str(original_path), residual)
    original_bytes = original_path.read_bytes()
    cv2.imwrite(str(assets / "planned.png"), planned)
    layout = {"elements": [
        {"id": "residual", "type": "image", "x": 0, "y": 0, "width": 160, "height": 100,
         "zIndex": 2, "src": "/media/assets/test/residual.png",
         "metadata": {"layerRole": "residual", "reconstructionStrategySource": "residual_detection"}},
        {"id": "planned", "type": "image", "x": 30, "y": 25, "width": 60, "height": 45,
         "zIndex": 2, "src": "/media/assets/test/planned.png",
         "metadata": {"reconstructionStrategySource": "planner"}},
    ]}

    assert count_duplicate_planned_visual_pixels(layout, assets) >= 35 * 50
    report = transfer_planned_visual_pixels(layout, assets, "test")
    assert report["trimmedOverlappingAssets"] == 1
    assert report["plannedVisualPixelsClearedFromOtherAssets"] >= 35 * 50
    assert original_path.read_bytes() == original_bytes
    new_residual = cv2.imread(str(assets / Path(layout["elements"][0]["src"]).name), cv2.IMREAD_UNCHANGED)
    assert new_residual[45, 50, 3] == 0  # No satellite remains under the moved object.
    assert new_residual[10, 10, 3] == 255  # The surrounding pale card is preserved.
    assert count_duplicate_planned_visual_pixels(layout, assets) == 0
    layer_objectized_elements(layout["elements"])
    assert layout["elements"][1]["zIndex"] > layout["elements"][0]["zIndex"]
    assert transfer_planned_visual_pixels(layout, assets, "test")["trimmedOverlappingAssets"] == 0


def test_duplicate_circle_prefers_transparent_contour(tmp_path: Path) -> None:
    source = np.full((100, 100, 3), 255, np.uint8)
    cv2.circle(source, (50, 50), 24, (30, 90, 190), -1)
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)
    assets = tmp_path / "assets"
    assets.mkdir()
    planned = source[20:80, 20:80]
    alpha = np.zeros((60, 60), np.uint8)
    cv2.circle(alpha, (30, 30), 24, 255, -1)
    cv2.imwrite(str(assets / "planned.png"), planned)
    cv2.imwrite(str(assets / "contour.png"), np.dstack((planned, alpha)))
    layout = {"elements": [
        {"id": "planned", "type": "image", "x": 20, "y": 20, "width": 60, "height": 60,
         "src": "/media/assets/test/planned.png", "metadata": {"reconstructionStrategySource": "planner"}},
        {"id": "contour", "type": "image", "x": 20, "y": 20, "width": 60, "height": 60,
         "src": "/media/assets/test/contour.png", "metadata": {"reconstructionStrategySource": "round_contour"}},
    ]}

    assert resolve_duplicate_contour_assets(source_path, layout, assets, "test") == 1
    merged = layout["elements"][2]
    assert layout["elements"][0]["metadata"]["ownedBy"] == merged["id"]
    assert layout["elements"][1]["metadata"]["ownedBy"] == merged["id"]
    image = cv2.imread(str(assets / Path(merged["src"]).name), cv2.IMREAD_UNCHANGED)
    assert image[0, 0, 3] == 0
    assert image[30, 30, 3] == 255


def test_incomplete_contour_prefers_complete_planned_icon(tmp_path: Path) -> None:
    source = np.full((90, 90, 3), 255, np.uint8)
    source[20:70, 20:70] = (30, 90, 190)
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)
    assets = tmp_path / "assets"
    assets.mkdir()
    planned = source[20:70, 20:70]
    alpha = np.full((50, 50), 255, np.uint8)
    alpha[:2, :] = 0  # A lost edge makes this contour visually worse.
    cv2.imwrite(str(assets / "planned.png"), planned)
    cv2.imwrite(str(assets / "contour.png"), np.dstack((planned, alpha)))
    layout = {"elements": [
        {"id": "planned", "type": "image", "x": 20, "y": 20, "width": 50, "height": 50,
         "src": "/media/assets/test/planned.png", "metadata": {"reconstructionStrategySource": "planner"}},
        {"id": "contour", "type": "image", "x": 20, "y": 20, "width": 50, "height": 50,
         "src": "/media/assets/test/contour.png", "metadata": {"reconstructionStrategySource": "round_contour"}},
    ]}

    assert resolve_duplicate_contour_assets(source_path, layout, assets, "test") == 1
    merged = layout["elements"][2]
    assert all(item["metadata"]["ownedBy"] == merged["id"] for item in layout["elements"][:2])
    image = cv2.imread(str(assets / Path(merged["src"]).name), cv2.IMREAD_UNCHANGED)
    assert image[0, 25, 3] == 255


def test_merged_icon_clears_underlying_residual_when_moved(tmp_path: Path) -> None:
    source = np.full((90, 90, 3), 255, np.uint8)
    cv2.circle(source, (45, 45), 17, (25, 75, 160), -1)
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)
    assets = tmp_path / "assets"
    assets.mkdir()
    crop = source[25:65, 25:65]
    mask = np.zeros((40, 40), np.uint8)
    cv2.circle(mask, (20, 20), 17, 255, -1)
    cv2.imwrite(str(assets / "planned.png"), crop)
    cv2.imwrite(str(assets / "contour.png"), np.dstack((crop, mask)))
    cv2.imwrite(str(assets / "residual.png"), np.dstack((source, np.full((90, 90), 255, np.uint8))))
    layout = {"elements": [
        {"id": "residual", "type": "image", "x": 0, "y": 0, "width": 90, "height": 90,
         "src": "/media/assets/test/residual.png", "metadata": {"reconstructionStrategySource": "residual_detection"}},
        {"id": "planned", "type": "image", "x": 25, "y": 25, "width": 40, "height": 40,
         "src": "/media/assets/test/planned.png", "metadata": {"reconstructionStrategySource": "planner"}},
        {"id": "contour", "type": "image", "x": 25, "y": 25, "width": 40, "height": 40,
         "src": "/media/assets/test/contour.png", "metadata": {"reconstructionStrategySource": "round_contour"}},
    ]}
    assert resolve_duplicate_contour_assets(source_path, layout, assets, "test") == 1
    assert transfer_planned_visual_pixels(layout, assets, "test")["trimmedOverlappingAssets"] == 1
    residual = cv2.imread(str(assets / Path(layout["elements"][0]["src"]).name), cv2.IMREAD_UNCHANGED)
    assert residual[45, 45, 3] == 0
    assert count_duplicate_planned_visual_pixels(layout, assets) == 0
