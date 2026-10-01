from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.reconstruction.asset_ownership import restore_image_owned_text
from app.services.pptx import renderer


def test_hidden_badge_letters_restore_complete_movable_image(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    project_id = "a" * 32
    root = tmp_path / project_id
    asset_dir = root / "assets"
    asset_dir.mkdir(parents=True)
    source = np.full((180, 240, 3), 255, np.uint8)
    cv2.circle(source, (110, 90), 50, (30, 40, 190), -1)
    cv2.putText(source, "AI", (90, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    source_path = root / "source.png"
    cv2.imwrite(str(source_path), source)
    damaged = source[40:140, 60:160].copy()
    damaged[39:65, 31:71] = 255
    alpha = np.zeros((100, 100), np.uint8)
    cv2.circle(alpha, (50, 50), 50, 255, -1)
    old_path = asset_dir / "badge.png"
    cv2.imwrite(str(old_path), np.dstack((damaged, alpha)))
    old_bytes = old_path.read_bytes()
    layout = {"elements": [
        {"id": "badge", "type": "image", "x": 60, "y": 40, "width": 100, "height": 100,
         "src": f"/media/assets/{project_id}/badge.png",
         "metadata": {"textCleaned": True, "editableTextIds": ["letters"], "preserveWholeAsset": True}},
        {"id": "letters", "type": "text", "x": 90, "y": 79, "width": 40, "height": 26,
         "text": "AI", "metadata": {"suppressed": True, "ownedBy": "badge", "rawOCRBBox": [90, 79, 130, 105]}},
    ]}

    restored = restore_image_owned_text(source_path, layout, asset_dir, project_id)

    assert restored == ["badge"]
    assert old_path.read_bytes() == old_bytes
    new_path = asset_dir / Path(layout["elements"][0]["src"]).name
    image = cv2.imread(str(new_path), cv2.IMREAD_UNCHANGED)
    assert image.shape[2] == 4 and image[0, 0, 3] == 0
    assert np.array_equal(image[50, 50, :3], source[90, 110])
    assert layout["elements"][0]["metadata"]["editableTextIds"] == []


def test_visible_editable_letters_prevent_source_text_restoration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    project_id = "b" * 32
    root = tmp_path / project_id
    asset_dir = root / "assets"
    asset_dir.mkdir(parents=True)
    source = np.full((80, 80, 3), 255, np.uint8)
    cv2.imwrite(str(root / "source.png"), source)
    cv2.imwrite(str(asset_dir / "badge.png"), source)
    layout = {"elements": [
        {"id": "badge", "type": "image", "x": 0, "y": 0, "width": 80, "height": 80,
         "src": f"/media/assets/{project_id}/badge.png", "metadata": {"textCleaned": True, "editableTextIds": ["label"]}},
        {"id": "label", "type": "text", "x": 10, "y": 10, "width": 30, "height": 20, "text": "AI"},
    ]}

    assert restore_image_owned_text(root / "source.png", layout, asset_dir, project_id) == []
    assert layout["elements"][0]["src"].endswith("/badge.png")
