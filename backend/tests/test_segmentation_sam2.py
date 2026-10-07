import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from app.services.segmentation.segmentation_provider import (
    OptionalSAM2Provider,
    create_segmentation_provider,
)


def test_sam2_adapter_writes_contour_alpha_and_unique_assets(tmp_path, monkeypatch):
    source = np.full((140, 220, 3), 255, np.uint8)
    cv2.circle(source, (65, 65), 35, (20, 80, 190), -1)
    image = tmp_path / "source.png"
    cv2.imwrite(str(image), source)
    mask = np.zeros(source.shape[:2], np.float32)
    cv2.circle(mask, (65, 65), 35, 1, -1)
    data = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: mask[None]))
    model = SimpleNamespace(predict=lambda **kwargs: [SimpleNamespace(masks=SimpleNamespace(data=data))])
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(SAM=lambda _: model))
    provider = OptionalSAM2Provider(tmp_path / "model.pt")
    first = provider.segment(image, tmp_path / "assets", "project")
    second = provider.segment(image, tmp_path / "assets", "project")
    assert first and first[0]["source"] == "ultralytics_sam2"
    assert first[0]["alphaCrop"] != second[0]["alphaCrop"]
    asset = cv2.imread(str(tmp_path / "assets" / Path(first[0]["alphaCrop"]).name), -1)
    assert asset[0, 0, 3] == 0 and asset[:, :, 3].max() == 255
    assert first[0]["contour"]


def test_missing_local_model_reports_opencv_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("SAM_MODEL_PATH", str(tmp_path / "missing.pt"))
    provider, warnings = create_segmentation_provider("sam2")
    assert provider.name == "opencv" and warnings
