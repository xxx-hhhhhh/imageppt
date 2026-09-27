from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from app.services.inpainting.provider import StabilityInpaintingProvider
from app.services.inpainting.service import InpaintingService


def test_stability_provider_sends_explicit_mask_without_network(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    Image.new("RGB", (40, 30), "white").save(source)
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[8:16, 12:24] = 255
    seen = {}

    class Response:
        content = source.read_bytes()

        def raise_for_status(self):
            pass

    def fake_post(url, *, headers, files, data, timeout):
        seen["url"] = url
        seen["mask"] = np.asarray(Image.open(BytesIO(files["mask"][1])))
        seen["prompt"] = data["prompt"]
        assert headers["authorization"] == "Bearer test-only"
        return Response()

    monkeypatch.setattr("requests.post", fake_post)
    StabilityInpaintingProvider(api_key="test-only").inpaint(source, mask, output)
    assert output.is_file()
    assert np.array_equal(seen["mask"], mask)
    assert "add no text" in seen["prompt"]


def test_complex_background_remains_pending_when_professional_provider_unavailable(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    background = tmp_path / "background.png"
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(source), image)
    cv2.imwrite(str(background), image)
    service = InpaintingService("opencv")
    service.prefer_inpaint = True
    monkeypatch.setattr(service, "_professional_provider", lambda: None)

    def fake_restore(*_args, **_kwargs):
        return background, [{"bbox": [10, 10, 50, 30], "cleanBBox": [8, 8, 52, 32], "category": "texture", "willReconstruct": True}]

    monkeypatch.setattr("app.services.inpainting.service.restore_with_strategy", fake_restore)
    service.restore_background(source, [], background)
    assert service.professional_pending == 1
    assert service.last_strategies[0]["professionalRepair"] == "unavailable"
    assert service.professional_attempts == 0
