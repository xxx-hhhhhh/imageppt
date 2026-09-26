from __future__ import annotations

import base64
import json

import pytest
from PIL import Image

from app.services.vision.qwen_provider import QwenProvider, VisionProviderError, image_to_data_url
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
        calls.append(messages)
        return next(responses)

    monkeypatch.setattr(provider, "_request", fake_request)
    result = provider.analyze_scene(image_path, {"ocr_elements": []})
    assert result["elements"][0]["role"] == "body"
    assert len(calls) == 2
    assert len(calls[1]) == len(calls[0]) + 2
    assert calls[1][-2] == {"role": "assistant", "content": "not json"}
    assert provider.vision_debug()["repairUsed"] is True


def test_qwen_page_plan_survives_semantic_scene_failure(monkeypatch, tmp_path):
    provider = object.__new__(QwenProvider)
    provider.name = "qwen"
    provider.model = "qwen3-vl-flash"
    image_path = tmp_path / "page.png"
    Image.new("RGB", (64, 64), "white").save(image_path)
    requests = []
    plan = {"modules": [{"id": "visual", "role": "chart", "bbox": {"left": 0.1, "top": 0.1, "width": 0.5, "height": 0.5}, "strategy": "whole_image", "confidence": 0.9}]}

    def fake_request(messages, repair_prompt=None):
        requests.append(messages[0]["content"])
        if len(requests) == 1:
            raise RuntimeError("semantic request timed out")
        return json.dumps(plan)

    monkeypatch.setattr(provider, "_request", fake_request)
    result = provider.analyze_scene(image_path, {"candidate_elements": [{"id": "chart", "type": "image", "bbox": {"left": 6, "top": 6, "width": 32, "height": 32}}]})
    assert len(requests) >= 2
    assert "视觉版式分析引擎" in requests[0]
    assert "信息图重建规划器" in requests[1]
    assert result["aiUsed"] is True
    assert result["reconstructionPlan"]["modules"][0]["id"] == "visual"


def test_qwen_requests_dedicated_plan_when_scene_omits_body(monkeypatch, tmp_path):
    provider = object.__new__(QwenProvider)
    provider.name = "qwen"
    provider.model = "qwen3-vl-flash"
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    incomplete = {"elements": [], "reconstructionPlan": {"modules": [{"id": "header", "bbox": {"left": 0, "top": 0, "width": 1, "height": 0.2}, "confidence": 0.9}]}}
    complete = {"modules": [{"id": "body", "bbox": {"left": 0, "top": 0.2, "width": 1, "height": 0.8}, "confidence": 0.9}]}
    responses = iter([json.dumps(incomplete), json.dumps(complete)])
    calls = []

    def fake_request(messages, repair_prompt=None):
        calls.append(messages[0]["content"])
        return next(responses)

    monkeypatch.setattr(provider, "_request", fake_request)
    context = {"width": 100, "height": 100, "candidate_elements": [
        {"id": str(index), "type": "text", "text": "line", "bbox": {"left": 10, "top": 35 + index * 10, "width": 40, "height": 8}}
        for index in range(4)
    ]}
    result = provider.analyze_scene(image_path, context)
    assert len(calls) == 2
    assert result["reconstructionPlan"]["modules"][0]["id"] == "body"


def test_high_quality_rejects_plan_that_still_omits_body(monkeypatch, tmp_path):
    provider = object.__new__(QwenProvider)
    provider.name = "qwen"
    provider.model = "qwen3-vl-flash"
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    incomplete = {"modules": [{"id": "header", "bbox": {"left": 0, "top": 0, "width": 1, "height": 0.2}, "confidence": 0.9}]}
    responses = iter([json.dumps({"elements": [], "reconstructionPlan": incomplete}), json.dumps(incomplete)])
    monkeypatch.setattr(provider, "_request", lambda messages, repair_prompt=None: next(responses))
    context = {"width": 100, "height": 100, "candidate_elements": [
        {"id": str(index), "type": "text", "text": "line", "bbox": {"left": 10, "top": 35 + index * 10, "width": 40, "height": 8}}
        for index in range(4)
    ]}
    with pytest.raises(VisionProviderError, match="does not cover"):
        provider.analyze_scene(image_path, context, mode="high")


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
