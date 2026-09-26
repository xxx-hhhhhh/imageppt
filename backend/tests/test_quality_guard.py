from pathlib import Path

import cv2
import numpy as np

from app.services.reconstruction.quality_guard import preserve_bad_text_regions


def test_preserves_source_text_when_rendered_line_moves(tmp_path: Path) -> None:
    source = np.full((80, 200, 3), 255, dtype=np.uint8)
    cv2.putText(source, "HELLO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    background = np.full_like(source, 255)
    preview = background.copy()
    cv2.putText(preview, "HELLO", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    source_path, background_path, preview_path = (tmp_path / name for name in ("source.png", "background.png", "preview.png"))
    for path, image in ((source_path, source), (background_path, background), (preview_path, preview)):
        cv2.imwrite(str(path), image)
    layout = {"elements": [{"id": "text-1", "type": "text", "text": "HELLO", "metadata": {"rawOCRBBox": [18, 20, 100, 48]}}]}

    result = preserve_bad_text_regions(source_path, background_path, preview_path, layout, tmp_path / "assets", 1)

    assert result == {"preservedTextRegions": 1, "restoredModules": 0}
    assert layout["elements"][0]["metadata"]["suppressRender"] is True
    restored = cv2.imread(str(background_path))
    np.testing.assert_array_equal(restored[20:48, 18:100], source[20:48, 18:100])


def test_suppresses_duplicate_when_original_text_remains_under_overlay(tmp_path: Path) -> None:
    source = np.full((60, 140, 3), 255, dtype=np.uint8)
    cv2.putText(source, "TEXT", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    source_path, background_path, preview_path = (tmp_path / name for name in ("source.png", "background.png", "preview.png"))
    for path in (source_path, background_path, preview_path):
        cv2.imwrite(str(path), source)
    layout = {"elements": [{"id": "text-1", "type": "text", "text": "TEXT", "metadata": {"rawOCRBBox": [8, 15, 90, 45]}}]}

    result = preserve_bad_text_regions(source_path, background_path, preview_path, layout, tmp_path / "assets", 1)

    assert result["preservedTextRegions"] == 1
    assert layout["elements"][0]["metadata"]["suppressRender"] is True
