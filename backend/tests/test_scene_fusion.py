from __future__ import annotations

from app.services.fusion.scene_fusion import fuse_scene


def test_fusion_keeps_cv_bbox_and_applies_qwen_semantics():
    layout = {"elements": [
        {"id": "text_001", "type": "text", "x": 10, "y": 20, "width": 100, "height": 24, "confidence": 0.95, "style": {"fontSize": 30}},
        {"id": "shape_001", "type": "ellipse", "x": 5, "y": 10, "width": 32, "height": 32, "confidence": 0.8, "style": {}},
    ], "groups": [], "relations": []}
    vision = {"confidence": 0.9, "elements": [
        {"ocrId": "ocr_1", "role": "main_title", "semanticType": "text", "groupId": "header_1", "fontClass": "display", "fontWeight": 700, "alignment": "center", "reconstructionStrategy": "editable_text", "visionConfidence": 0.9},
        {"id": "shape_001", "role": "icon", "semanticType": "icon", "groupId": "header_1", "reconstructionStrategy": "transparent_image", "doNotVectorize": True, "visualComplexity": 0.9, "visionConfidence": 0.8},
    ], "groups": [{"id": "header_1", "type": "card", "members": ["text_001", "shape_001"]}]}
    result = fuse_scene(layout, [], vision)
    text = next(item for item in result["elements"] if item["id"] == "text_001")
    assert text["role"] == "main_title"
    assert text["x"] == 10 and text["y"] == 20
    assert text["groupId"] == "header_1"
    assert text["finalConfidence"] > 0.65
    assert text["style"]["fontClass"] == "display"
    assert text["style"]["fontWeight"] == 700
    assert text["style"]["align"] == "center"
    assert text["metadata"]["reconstructionStrategy"] == "editable_text"
    assert text["metadata"]["reconstructionStrategySource"] == "vision"
    icon = next(item for item in result["elements"] if item["id"] == "shape_001")
    assert icon["metadata"]["doNotVectorize"] is True
    assert icon["metadata"]["visualComplexity"] == 0.9
