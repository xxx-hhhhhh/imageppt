from __future__ import annotations

import base64

import pytest
from PIL import Image

from app.services.vision.qwen_provider import QwenProvider, image_to_data_url
from app.services.vision.schemas import extract_json, normalize_scene_payload, validate_json


def test_image_to_data_url_supports_local_png(tmp_path):
    path = tmp_path / "tiny.png"
    Image.new("RGB", (2, 2), "white").save(path)
    value = image_to_data_url(path)
    assert value.startswith("data:image/png;base64,")
    assert base64.b64decode(value.split(",", 1)[1]).startswith(b"\x89PNG")


def test_json_parser_strips_fence_and_validates_scene():
    payload = extract_json("```json\n{\"elements\":[{\"ocrId\":\"ocr_1\",\"role\":\"main_title\"}]}\n```")
    scene = validate_json(payload)
    assert scene["elements"][0]["role"] == "main_title"


def test_scene_schema_preserves_reconstruction_semantics():
    scene = validate_json({"elements": [{
        "id": "icon_1",
        "role": "icon",
        "semanticType": "ornament",
        "groupId": "hero",
        "zLayer": "foreground",
        "fontClass": "display",
        "fontWeight": 700,
        "alignment": "center",
        "reconstructionStrategy": "transparent_image",
        "doNotVectorize": True,
        "visualComplexity": 0.9,
        "visionConfidence": 0.95,
    }]})
    element = scene["elements"][0]
    assert element["semanticType"] == "ornament"
    assert element["reconstructionStrategy"] == "transparent_image"
    assert element["doNotVectorize"] is True
    assert element["visualComplexity"] == 0.9


def test_qwen_provider_retries_json_repair(monkeypatch, tmp_path):
    provider = object.__new__(QwenProvider)
    provider.name = "qwen"
    provider.model = "qwen3-vl-flash"
    image_path = tmp_path / "tiny.png"
    Image.new("RGB", (2, 2), "white").save(image_path)
    responses = iter(["not json", '{"elements": [{"ocrId": "ocr_1", "role": "body"}]}'])
    calls = []

    def fake_request(messages, repair_prompt=None):
        calls.append(repair_prompt)
        return next(responses)

    monkeypatch.setattr(provider, "_request", fake_request)
    result = provider.analyze_scene(image_path, {"ocr_elements": []})
    assert result["elements"][0]["role"] == "body"
    assert len(calls) == 2
    assert calls[0] is None
    assert calls[1] is not None
    assert provider.vision_debug()["repairUsed"] is True


def test_scene_payload_normalizes_common_qwen_aliases():
    diagnostics = {}
    scene = validate_json({
        "confidence": 85,
        "layers": [{"name": "background"}, {"name": "text"}],
        "elements": [{
            "id": 17,
            "visualComplexity": "high",
            "visionConfidence": "92%",
            "confidence": "80",
            "fontWeight": "bold",
            "alignment": "centre",
            "doNotVectorize": "yes",
            "reconstructionStrategy": "transparent",
        }],
    }, diagnostics=diagnostics)

    element = scene["elements"][0]
    assert scene["confidence"] == 0.85
    assert scene["layers"] == ["background", "text"]
    assert element["id"] == "17"
    assert element["visualComplexity"] == 0.85
    assert element["visionConfidence"] == 0.92
    assert element["confidence"] == 0.8
    assert element["fontWeight"] == 700
    assert element["alignment"] == "center"
    assert element["doNotVectorize"] is True
    assert element["reconstructionStrategy"] == "transparent_image"
    assert diagnostics["normalizationWarnings"]


def test_scene_payload_keeps_one_malformed_element_and_ten_good_elements():
    elements = [
        {
            "id": "bad-but-retained",
            "visualComplexity": {"unexpected": True},
            "visionConfidence": None,
            "fontWeight": ["bold"],
            "alignment": "未知",
            "doNotVectorize": "off",
            "reconstructionStrategy": "unsupported_strategy",
        }
    ] + [
        {
            "id": f"good-{index}",
            "visionConfidence": 0.9,
            "reconstructionStrategy": "editable_text",
        }
        for index in range(10)
    ]
    diagnostics = {}

    scene = validate_json({
        "elements": elements,
        "groups": None,
        "relations": "invalid",
        "regions": {"region_1": {"type": "content"}},
        "layers": {"background": {"name": "background"}},
    }, diagnostics=diagnostics)

    assert len(scene["elements"]) == 11
    malformed = scene["elements"][0]
    assert malformed["id"] == "bad-but-retained"
    assert malformed["visualComplexity"] == 0.0
    assert malformed["visionConfidence"] == 0.5
    assert malformed["fontWeight"] == 400
    assert malformed["alignment"] == "left"
    assert malformed["doNotVectorize"] is False
    assert malformed["reconstructionStrategy"] is None
    assert scene["groups"] == []
    assert scene["relations"] == []
    assert len(scene["regions"]) == 1
    assert scene["layers"] == ["background"]
    assert diagnostics["droppedElements"] == 0


def test_valid_json_with_bad_field_types_does_not_call_repair(monkeypatch, tmp_path):
    provider = object.__new__(QwenProvider)
    provider.name = "qwen"
    provider.model = "qwen3-vl-flash"
    image_path = tmp_path / "tiny.png"
    Image.new("RGB", (2, 2), "white").save(image_path)
    calls = []
    raw = '{"elements":[{"id":"one","visualComplexity":"high","fontWeight":"bold","alignment":"middle"}]}'

    def fake_request(messages, repair_prompt=None):
        calls.append(repair_prompt)
        return raw

    monkeypatch.setattr(provider, "_request", fake_request)
    result = provider.analyze_scene(image_path, {"ocr_elements": []})

    assert calls == [None]
    assert result["elements"][0]["visualComplexity"] == 0.85
    assert result["elements"][0]["fontWeight"] == 700
    assert result["elements"][0]["alignment"] == "center"
    debug = provider.vision_debug()
    assert debug["rawResponseAvailable"] is True
    assert debug["repairUsed"] is False
    assert debug["normalizationApplied"] is True


def test_normalize_scene_payload_accepts_null_and_object_arrays():
    diagnostics = {}
    normalized = normalize_scene_payload({
        "elements": None,
        "groups": {"card": {"role": "card"}},
        "relations": 123,
        "repeatedComponents": None,
        "layers": [{"name": "icons"}, None],
    }, diagnostics)

    assert normalized["elements"] == []
    assert normalized["groups"] == [{"role": "card", "id": "card"}]
    assert normalized["relations"] == []
    assert normalized["repeatedComponents"] == []
    assert normalized["layers"] == ["icons"]
