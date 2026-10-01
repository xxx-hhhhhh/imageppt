from pathlib import Path

import cv2
import numpy as np

from app.services.reconstruction.visual_asset_ownership import count_duplicate_planned_visual_pixels, transfer_planned_visual_pixels
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
