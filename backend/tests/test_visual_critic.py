from __future__ import annotations

from app.services.fusion.adjustment_validator import apply_safe_adjustments
from app.services.visual_qa.analyzer import enrich_quality_score, run_visual_qa

import cv2
import numpy as np


def test_critic_adjustments_are_bounded():
    scene = {"canvas": {"width": 1000, "height": 500}, "elements": [{"id": "title", "bbox": {"left": 100, "top": 100, "width": 200, "height": 40}, "style": {"fontSize": 40}}]}
    critic = {"issues": [{"elementId": "title", "adjustment": {"moveX": 9999, "moveY": -9999, "widthScale": 2, "fontSizeScale": 0.1}}]}
    result = apply_safe_adjustments(scene, critic)
    item = result["elements"][0]
    assert item["bbox"]["left"] == 200
    assert item["bbox"]["top"] == 50
    assert item["bbox"]["width"] == 240
    assert item["style"]["fontSize"] == 28


def test_visual_qa_weights_regions_and_reports_penalties(tmp_path):
    original = np.full((120, 240, 3), 255, dtype=np.uint8)
    preview = original.copy()
    cv2.putText(original, "A", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(preview, "A", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(preview, "A", (24, 59), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    original_path, preview_path = tmp_path / "original.png", tmp_path / "preview.png"
    cv2.imwrite(str(original_path), original)
    cv2.imwrite(str(preview_path), preview)
    layout = {"slide": {"width": 240, "height": 120}, "elements": [{"id": "text", "type": "text", "x": 10, "y": 20, "width": 60, "height": 50, "metadata": {}}]}
    score = run_visual_qa(original_path, preview_path, tmp_path, layout)
    assert {"textRegionScore", "layoutScore", "componentScore", "backgroundScore", "ghostingPenalty", "duplicatePenalty"}.issubset(score)
    assert score["ghostingPenalty"] > 0


def test_structural_qa_flags_misplaced_text_duplicate_and_distorted_image(tmp_path):
    source = np.full((120, 240, 3), 255, dtype=np.uint8)
    original_path, preview_path = tmp_path / "original.png", tmp_path / "preview.png"
    cv2.imwrite(str(original_path), source)
    cv2.imwrite(str(preview_path), source)
    asset_path = tmp_path / "asset.png"
    cv2.imwrite(str(asset_path), np.full((20, 100, 3), 100, dtype=np.uint8))
    text = {"id": "title", "type": "text", "text": "Title", "x": 70, "y": 50, "width": 60, "height": 20, "metadata": {"rawOCRBBox": [10, 10, 70, 30]}}
    layout = {"slide": {"width": 240, "height": 120}, "elements": [text, {**text, "id": "copy"}, {"id": "chart", "type": "image", "src": str(asset_path), "x": 130, "y": 10, "width": 50, "height": 50, "role": "chart"}]}
    score = run_visual_qa(original_path, preview_path, tmp_path, layout)
    problems = {issue["problem"] for issue in score["issues"]}
    assert {"wrongBBox", "duplicateText", "imageDistortion"}.issubset(problems)
    assert score["textPlacementScore"] < 1
    assert score["imageAspectScore"] < 1
    enriched = enrich_quality_score(score, editable_coverage=0.5, movable_coverage=0.4, ghosting_count=1, background_residual_count=1)
    assert enriched["overall"] < enriched["visualSceneScore"]
