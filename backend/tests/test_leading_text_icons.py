from pathlib import Path

import cv2
import numpy as np

from app.services.reconstruction.leading_text_icons import extract_leading_text_icons
from app.services.reconstruction.visual_asset_ownership import transfer_planned_visual_pixels


def test_colored_leading_icon_becomes_independent_without_erasing_editable_text(tmp_path: Path) -> None:
    source = np.full((100, 250, 3), 255, np.uint8)
    cv2.rectangle(source, (33, 37), (49, 53), (30, 145, 30), 2)
    cv2.line(source, (37, 45), (41, 49), (30, 145, 30), 2)
    cv2.line(source, (41, 49), (46, 40), (30, 145, 30), 2)
    cv2.putText(source, "LABEL DATA", (61, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 65, 25), 1)
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)
    assets = tmp_path / "assets"
    assets.mkdir()
    cv2.imwrite(str(assets / "residual.png"), np.dstack((source, np.full((100, 250), 255, np.uint8))))
    layout = {"elements": [
        {"id": "residual", "type": "image", "x": 0, "y": 0, "width": 250, "height": 100,
         "src": "/media/assets/test/residual.png",
         "metadata": {"reconstructionStrategySource": "residual_detection"}},
        {"id": "label", "type": "text", "text": "LABEL DATA", "x": 30, "y": 34,
         "width": 160, "height": 26, "zIndex": 20,
         "metadata": {"rawOCRBBox": [30, 34, 190, 58]}},
    ]}

    assert extract_leading_text_icons(source_path, layout, assets, "test", 1) == 1
    icon = layout["elements"][-1]
    assert icon["metadata"]["reconstructionStrategySource"] == "leading_text_icon"
    assert layout["elements"][1]["metadata"]["rawOCRBBox"][0] > icon["x"] + icon["width"]
    assert layout["elements"][1]["text"] == "LABEL DATA"
    report = transfer_planned_visual_pixels(layout, assets, "test")
    assert report["trimmedOverlappingAssets"] == 1
    residual = cv2.imread(str(assets / Path(layout["elements"][0]["src"]).name), cv2.IMREAD_UNCHANGED)
    assert residual[45, 34, 3] == 0
    extracted = cv2.imread(str(assets / Path(icon["src"]).name), cv2.IMREAD_UNCHANGED)
    assert np.count_nonzero(extracted[:, :, 3]) > 60
    assert extract_leading_text_icons(source_path, layout, assets, "test", 1) == 0


def test_same_color_first_letter_is_not_misclassified_as_icon(tmp_path: Path) -> None:
    source = np.full((100, 250, 3), 255, np.uint8)
    cv2.putText(source, "LABEL DATA", (30, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 65, 25), 1)
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)
    layout = {"elements": [{"id": "text", "type": "text", "text": "LABEL DATA", "x": 30, "y": 34,
                            "width": 160, "height": 26,
                            "metadata": {"rawOCRBBox": [30, 34, 190, 58]}}]}
    assert extract_leading_text_icons(source_path, layout, tmp_path / "assets", "test", 1) == 0
    assert len(layout["elements"]) == 1
