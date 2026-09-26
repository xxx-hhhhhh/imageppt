from __future__ import annotations

import cv2
import numpy as np

from app.services.background.analyzer import analyze_background
from app.services.background.strategy import restore_background
from app.services.ocr.provider import OCRResult


def test_solid_background_uses_ring_median_without_full_inpaint(tmp_path):
    image = np.full((80, 120, 3), 230, dtype=np.uint8)
    cv2.putText(image, "A", (42, 46), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    cv2.imwrite(str(source), image)
    profile = analyze_background(image, [40, 25, 62, 50])
    assert profile["category"] in {"solid", "texture"}
    restored, strategies = restore_background(source, [OCRResult("A", [40, 25, 62, 50], 0.9, {})], output)
    assert restored == output and output.exists()
    assert strategies[0]["reconstructionStrategy"] in {"native_fill", "local_inpaint"}
    assert strategies[0]["cleanBBox"][0] < 40
    assert strategies[0]["cleanBBox"][1] < 25
    cleaned = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
    assert int(cleaned[25:50, 40:62].min()) > 180


def test_preserved_asset_text_is_not_inpainted(tmp_path):
    image = np.full((80, 120, 3), 240, dtype=np.uint8)
    cv2.putText(image, "A", (42, 46), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    cv2.imwrite(str(source), image)
    _, strategies = restore_background(source, [OCRResult("A", [40, 25, 62, 50], 0.9, {})], output, [[30, 15, 75, 60]])
    assert strategies[0]["sourceContentPreserved"] is True
    assert np.array_equal(cv2.imread(str(source)), cv2.imread(str(output)))


def test_long_text_removal_keeps_adjacent_icon(tmp_path):
    image = np.full((120, 800, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (40, 45), (85, 85), (0, 0, 220), -1)
    cv2.putText(image, "long text", (100, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (15, 15, 15), 2)
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    cv2.imwrite(str(source), image)
    _, strategies = restore_background(source, [OCRResult("long text", [100, 50, 700, 80], 0.9, {})], output)
    cleaned = cv2.imread(str(output))
    assert strategies[0]["cleanBBox"][0] >= 88
    assert np.array_equal(cleaned[45:86, 40:86], image[45:86, 40:86])
