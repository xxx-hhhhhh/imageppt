from __future__ import annotations

import io
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from app.services.inpainting import qwen_image_edit


def test_complex_text_repair_is_bounded_to_ocr_box(tmp_path, monkeypatch):
    image = np.full((520, 520, 3), 245, dtype=np.uint8)
    cv2.putText(image, "TEXT", (190, 255), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (10, 10, 10), 3)
    source = tmp_path / "source.png"
    background = tmp_path / "background.png"
    cv2.imwrite(str(source), image)
    cv2.imwrite(str(background), image)
    settings = SimpleNamespace(enabled=True, timeout=60, providers={"qwen": SimpleNamespace(enabled=True, api_key="test-key", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")})
    monkeypatch.setattr(qwen_image_edit, "load_vision_settings", lambda: settings)

    class Response:
        def __init__(self, *, content=None, data=None):
            self.content = content
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    calls = []

    def post(url, *, headers, json, timeout):
        calls.append(json["model"])
        assert headers["Authorization"] == "Bearer test-key"
        return Response(data={"output": {"choices": [{"message": {"content": [{"image": "https://example.test/image.png"}]}}]}})

    def get(url, *, timeout):
        edited = Image.new("RGB", (512, 512), (245, 245, 245))
        stream = io.BytesIO()
        edited.save(stream, format="PNG")
        return Response(content=stream.getvalue())

    monkeypatch.setattr(qwen_image_edit.requests, "post", post)
    monkeypatch.setattr(qwen_image_edit.requests, "get", get)
    strategies = [{"bbox": [180, 215, 310, 275], "cleanBBox": [175, 210, 315, 280], "category": "complex", "willReconstruct": True}]
    assert qwen_image_edit.repair_complex_text(source, background, strategies) == 1
    result = cv2.imread(str(background))
    assert np.array_equal(result[:215], image[:215])
    assert np.array_equal(result[275:], image[275:])
    assert np.array_equal(result[:, :180], image[:, :180])
    assert np.array_equal(result[:, 310:], image[:, 310:])
    assert np.mean(result[215:275, 180:310]) > np.mean(image[215:275, 180:310])
    assert strategies[0]["aiRepair"] == "accepted"
    assert calls == ["qwen-image-edit-plus"]


def test_image_edit_skips_other_region_keys(tmp_path, monkeypatch):
    settings = SimpleNamespace(enabled=True, timeout=60, providers={"qwen": SimpleNamespace(enabled=True, api_key="test-key", base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")})
    monkeypatch.setattr(qwen_image_edit, "load_vision_settings", lambda: settings)
    assert qwen_image_edit.repair_complex_text(tmp_path / "missing.png", tmp_path / "missing-bg.png", []) == 0
