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
