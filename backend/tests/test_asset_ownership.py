from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.reconstruction.asset_ownership import is_badge_owned_text, is_uncertain_image_owned_text, restore_image_owned_text
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


def test_generic_image_does_not_claim_suppressed_normal_text_as_badge() -> None:
    text = {"id": "label", "type": "text", "x": 10, "y": 10, "width": 40, "height": 20,
            "text": "Normal label", "metadata": {"suppressed": True, "ownedBy": "photo"}}
    layout = {"elements": [
        {"id": "photo", "type": "image", "x": 0, "y": 0, "width": 100, "height": 80,
         "metadata": {"preserveWholeAsset": True, "editableTextIds": ["label"]}},
        text,
    ]}

    assert not is_badge_owned_text(text, layout)
    layout["elements"][0]["metadata"]["wholeBadgeAsset"] = True
    assert is_badge_owned_text(text, layout)


def test_uncertain_ocr_requires_complete_visible_image_owner() -> None:
    text = {"id": "ocr", "type": "text", "text": "noise", "confidence": .3,
            "metadata": {"suppressed": True, "ownedBy": "photo", "rawOCRBBox": [10, 10, 40, 30]}}
    asset = {"id": "photo", "type": "image", "x": 0, "y": 0, "width": 80, "height": 60,
             "metadata": {"preserveWholeAsset": True}}
    layout = {"elements": [text, asset]}
    assert is_uncertain_image_owned_text(text, layout)
    text["confidence"] = .9
    assert not is_uncertain_image_owned_text(text, layout)
    text.pop("confidence")
    assert not is_uncertain_image_owned_text(text, layout)
    text["confidence"] = .3
    asset["metadata"]["editableTextIds"] = ["ocr"]
    assert not is_uncertain_image_owned_text(text, layout)
    asset["metadata"].pop("editableTextIds")
    asset["metadata"]["suppressed"] = True
    assert not is_uncertain_image_owned_text(text, layout)
    asset["metadata"].pop("suppressed")
    text["metadata"]["rawOCRBBox"] = [70, 50, 100, 80]
    assert not is_uncertain_image_owned_text(text, layout)
    layout["elements"].remove(asset)
    assert not is_uncertain_image_owned_text(text, layout)


@pytest.mark.parametrize("scale", [1, .5])
def test_mixed_asset_restores_hidden_text_without_reintroducing_editable_glyphs(tmp_path: Path, monkeypatch, scale) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    project_id = "mixed-owner"
    root = tmp_path / project_id
    assets = root / "assets"
    assets.mkdir(parents=True)
    source = np.full((100, 200, 3), (210, 220, 235), np.uint8)
    source[20:35, 15:65] = (15, 25, 35)
    source[60:75, 115:165] = (15, 25, 35)
    source_path = root / "source.png"
    cv2.imwrite(str(source_path), source)
    clean = np.full_like(source, (210, 220, 235))
    rgb = cv2.resize(clean, (round(200*scale), round(100*scale)))
    alpha = np.full(rgb.shape[:2], 255, np.uint8)
    alpha[:3] = 0
    old_path = assets / "module.png"
    cv2.imwrite(str(old_path), np.dstack((rgb, alpha)))
    old_bytes = old_path.read_bytes()
    layout = {"elements": [
        {"id": "module", "type": "image", "x": 0, "y": 0, "width": 200, "height": 100,
         "src": f"/media/assets/{project_id}/module.png",
         "metadata": {"textCleaned": True, "editableTextIds": ["hidden", "editable"]}},
        {"id": "hidden", "type": "text", "text": "HIDDEN",
         "metadata": {"suppressed": True, "ownedBy": "module", "rawOCRBBox": [10, 15, 180, 80]}},
        {"id": "editable", "type": "text", "text": "EDITABLE",
         "metadata": {"rawOCRBBox": [110, 55, 170, 80]}},
    ]}

    assert restore_image_owned_text(source_path, layout, assets, project_id) == ["module"]
    updated = cv2.imread(str(assets / Path(layout["elements"][0]["src"]).name), cv2.IMREAD_UNCHANGED)
    source_scaled = cv2.resize(source, (rgb.shape[1], rgb.shape[0]))
    assert np.array_equal(updated[round(20*scale):round(35*scale), round(15*scale):round(65*scale), :3],
                          source_scaled[round(20*scale):round(35*scale), round(15*scale):round(65*scale)])
    assert np.array_equal(updated[round(55*scale):round(80*scale), :, :3], rgb[round(55*scale):round(80*scale)])
    assert np.array_equal(updated[:, :, 3], alpha)
    assert old_path.read_bytes() == old_bytes
    metadata = layout["elements"][0]["metadata"]
    assert metadata["textCleaned"] is True
    assert metadata["editableTextIds"] == ["editable"]
    assert metadata["restoredImageOwnedTextIds"] == ["hidden"]
    assert restore_image_owned_text(source_path, layout, assets, project_id) == []
