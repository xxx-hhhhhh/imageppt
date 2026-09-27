from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import requests

from app.services.inpainting.local_client import LocalIOPaintClient
from app.services.inpainting.service import InpaintingService
from app.services.ocr.provider import OCRResult


class Response:
    def __init__(self, data=None, content=b"", status=200):
        self.data, self.content, self.status_code = data, content, status

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("mock failure")


def png(image: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def mock_service(monkeypatch, *, fail_post=False):
    calls = []
    def get(url, **kwargs):
        calls.append(("get", url))
        if url.endswith("/openapi.json"):
            return Response({"paths": {"/api/v1/inpaint": {"post": {"requestBody": {"content": {"application/json": {"schema": {"properties": {"image": {}, "mask": {}}}}}}}}, "/api/v1/model": {"get": {}}}})
        return Response({"name": "lama"})
    def post(url, *, json, **kwargs):
        calls.append(("post", url))
        image = Image.open(BytesIO(base64.b64decode(json["image"])))
        mask = np.asarray(Image.open(BytesIO(base64.b64decode(json["mask"]))).convert("L"))
        assert image.size == (64, 48)
        assert mask.shape == (48, 64)
        assert mask.max() == 255
        if fail_post:
            raise requests.Timeout("mock timeout")
        return Response(content=png(np.full((48, 64, 3), 255, np.uint8)))
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    return calls


def test_discovers_running_iopaint_route_and_calls_image_mask(monkeypatch, tmp_path):
    calls = mock_service(monkeypatch)
    client = LocalIOPaintClient("http://127.0.0.1:8080", "lama")
    assert client.probe() == {"connected": True, "route": "/api/v1/inpaint", "model": "lama"}
    source, result = tmp_path / "source.png", tmp_path / "result.png"
    cv2.imwrite(str(source), np.full((48, 64, 3), 100, np.uint8))
    mask = np.zeros((48, 64), np.uint8)
    mask[10:20, 10:25] = 255
    client.inpaint(source, mask, result)
    assert result.is_file()
    assert client.successes == 1
    assert ("post", "http://127.0.0.1:8080/api/v1/inpaint") in calls


def test_main_background_repair_uses_lama_then_opencv_fallback(monkeypatch, tmp_path):
    source, output = tmp_path / "source.png", tmp_path / "background.png"
    image = np.full((48, 64, 3), 240, np.uint8)
    image[15:22, 15:30] = 0
    cv2.imwrite(str(source), image)
    regions = [OCRResult("Test", [15, 15, 30, 22], 0.9, {})]

    mock_service(monkeypatch)
    client = LocalIOPaintClient("http://127.0.0.1:8080")
    assert client.probe()["connected"]
    monkeypatch.setattr("app.services.inpainting.service.create_inpainting_provider", lambda preferred: (client, []))
    service = InpaintingService()
    service.restore_background(source, regions, output)
    assert client.successes == 1
    assert service.ai_repaired_regions == 1
    assert np.all(cv2.imread(str(output))[15:22, 15:30] == 255)

    mock_service(monkeypatch, fail_post=True)
    failed_client = LocalIOPaintClient("http://127.0.0.1:8080")
    assert failed_client.probe()["connected"]
    monkeypatch.setattr("app.services.inpainting.service.create_inpainting_provider", lambda preferred: (failed_client, []))
    service = InpaintingService()
    service.restore_background(source, regions, output)
    assert output.is_file()
    assert failed_client.failures == 1
    assert service.last_strategies[0]["professionalRepair"] == "fallback_opencv"


def test_unavailable_local_service_does_not_interrupt_background_repair(monkeypatch, tmp_path):
    def unreachable(*args, **kwargs):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(requests, "get", unreachable)
    monkeypatch.setattr("app.services.inpainting.provider.LOCAL_INPAINT_ENABLED", True)
    client = LocalIOPaintClient("http://127.0.0.1:8080")
    assert client.probe()["connected"] is False
    source, output = tmp_path / "source.png", tmp_path / "background.png"
    cv2.imwrite(str(source), np.full((48, 64, 3), 220, np.uint8))
    service = InpaintingService("opencv")
    assert service.provider.name == "opencv"
    service.restore_background(source, [OCRResult("Test", [15, 15, 30, 22], 0.9, {})], output)
    assert output.is_file()


def test_local_status_endpoint_reports_live_probe_without_credentials(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main

    mock_service(monkeypatch)
    monkeypatch.setattr(main, "LOCAL_INPAINT_ENABLED", True)
    response = TestClient(main.app).get("/api/inpainting/local/status")
    assert response.status_code == 200
    assert response.json() == {"enabled": True, "connected": True, "model": "lama", "route": "/api/v1/inpaint"}
