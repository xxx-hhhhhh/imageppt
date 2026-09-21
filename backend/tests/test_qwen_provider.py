from __future__ import annotations

import base64

import pytest
from PIL import Image

from app.services.vision.qwen_provider import QwenProvider, image_to_data_url
from app.services.vision.schemas import extract_json, validate_json


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


def test_qwen_provider_retries_json_repair(monkeypatch, tmp_path):
    provider = object.__new__(QwenProvider)
    provider.name = "qwen"
    image_path = tmp_path / "tiny.png"
    Image.new("RGB", (2, 2), "white").save(image_path)
    responses = iter(["not json", '{"elements": [{"ocrId": "ocr_1", "role": "body"}]}'])
    monkeypatch.setattr(provider, "_request", lambda messages, repair_prompt=None: next(responses))
    result = provider.analyze_scene(image_path, {"ocr_elements": []})
    assert result["elements"][0]["role"] == "body"
