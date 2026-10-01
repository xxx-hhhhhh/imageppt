import copy
from pathlib import Path

import cv2
import numpy as np

from app.services.reconstruction.colored_text_support import extract_colored_text_supports


def test_colored_title_strip_becomes_single_movable_owner_with_editable_text(tmp_path: Path) -> None:
    source = np.full((150, 320, 3), 255, np.uint8)
    cv2.rectangle(source, (25, 35), (285, 84), (220, 110, 20), -1)
    cv2.putText(source, "TITLE", (75, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.circle(source, (260, 60), 9, (40, 180, 55), -1)
    source_path = tmp_path / "source.png"
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    cv2.imwrite(str(source_path), source)
    old = np.dstack((source, np.full(source.shape[:2], 255, np.uint8)))
    cv2.imwrite(str(asset_dir / "old.png"), old)
    layout = {"slide": {"width": 320, "height": 150}, "elements": [
        {"id": "bar_shape", "type": "roundedRectangle", "x": 25, "y": 35, "width": 261, "height": 50,
         "zIndex": 5, "style": {"fill": "#146EDC"}, "metadata": {"reconstructionStrategy": "native_shape"}},
        {"id": "old", "type": "image", "x": 0, "y": 0, "width": 320, "height": 150,
         "zIndex": 10, "src": "/media/assets/test/old.png"},
        {"id": "title", "type": "text", "text": "TITLE", "x": 70, "y": 45, "width": 110, "height": 32,
         "zIndex": 20, "style": {"color": "#FFFFFF"}, "metadata": {"rawOCRBBox": [70, 45, 180, 77]}},
    ]}
    second_run_layout = copy.deepcopy(layout)

    stats = extract_colored_text_supports(source_path, layout, asset_dir, "test", 1)

    assert stats["coloredTextSupports"] == 1
    new = next(item for item in layout["elements"] if item["id"].startswith("colored_support"))
    assert new["type"] == "image" and new["width"] < 320
    assert next(item for item in layout["elements"] if item["id"] == "title")["zIndex"] > new["zIndex"]
    original = cv2.imread(str(asset_dir / "old.png"), cv2.IMREAD_UNCHANGED)
    assert original[60, 100, 3] == 255  # The previous valid asset stays intact.
    previous_owner = next(item for item in layout["elements"] if item["id"] == "old")
    assert previous_owner["src"] != "/media/assets/test/old.png"
    current = cv2.imread(str(asset_dir / Path(previous_owner["src"]).name), cv2.IMREAD_UNCHANGED)
    assert current[60, 100, 3] == 0
    assert current[110, 100, 3] == 255
    replacement = cv2.imread(str(asset_dir / Path(new["src"]).name), cv2.IMREAD_UNCHANGED)
    assert replacement is not None and replacement.shape[2] == 4
    assert np.max(replacement[:, :, 3]) == 255
    source_letters = np.all(source[45:77, 70:180] == 255, axis=2)
    rx, ry = 70 - new["x"], 45 - new["y"]
    cleaned_letters = replacement[ry:ry + 32, rx:rx + 110, :3][source_letters]
    assert np.mean(np.all(cleaned_letters >= 245, axis=1)) < 0.05
    assert next(item for item in layout["elements"] if item["id"] == "bar_shape")["metadata"]["suppressed"] is True
    first_asset_path = asset_dir / Path(new["src"]).name
    first_asset_bytes = first_asset_path.read_bytes()
    assert extract_colored_text_supports(source_path, second_run_layout, asset_dir, "test", 1)["coloredTextSupports"] == 1
    second = next(item for item in second_run_layout["elements"] if item["id"].startswith("colored_support"))
    assert second["src"] != new["src"]
    assert first_asset_path.read_bytes() == first_asset_bytes


def test_large_colored_module_is_not_mistaken_for_title_strip(tmp_path: Path) -> None:
    source = np.full((240, 360, 3), 255, np.uint8)
    source[30:200, 25:335] = (45, 150, 45)
    cv2.putText(source, "HEADING", (45, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)
    layout = {"elements": [{"id": "heading", "type": "text", "text": "HEADING", "x": 42, "y": 48,
                            "width": 150, "height": 35, "metadata": {"rawOCRBBox": [42, 48, 192, 83]}}]}

    stats = extract_colored_text_supports(source_path, layout, tmp_path, "test", 1)

    assert stats["coloredTextSupports"] == 0
