from __future__ import annotations

from app.services.fusion.adjustment_validator import apply_safe_adjustments


def test_critic_adjustments_are_bounded():
    scene = {"canvas": {"width": 1000, "height": 500}, "elements": [{"id": "title", "bbox": {"left": 100, "top": 100, "width": 200, "height": 40}, "style": {"fontSize": 40}}]}
    critic = {"issues": [{"elementId": "title", "adjustment": {"moveX": 9999, "moveY": -9999, "widthScale": 2, "fontSizeScale": 0.1}}]}
    result = apply_safe_adjustments(scene, critic)
    item = result["elements"][0]
    assert item["bbox"]["left"] == 200
    assert item["bbox"]["top"] == 50
    assert item["bbox"]["width"] == 240
    assert item["style"]["fontSize"] == 28
