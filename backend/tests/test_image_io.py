from pathlib import Path

import cv2
import numpy as np
import pytest
from app.services.reconstruction.white_objectization import objectize_on_white
from app.utils import image_io
from PIL import Image


@pytest.mark.parametrize("channels,dtype", [(3, np.uint8), (4, np.uint8), (1, np.uint8), (1, np.uint16)])
def test_unicode_png_roundtrip_preserves_pixels_and_alpha(tmp_path, monkeypatch, channels, dtype):
    path = tmp_path / "我的云端硬盘" / "图片 PPT" / "真实轮廓🖼️.PNG"
    path.parent.mkdir(parents=True)
    shape = (28, 42) if channels == 1 else (28, 42, channels)
    pixels = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    # Exercise Windows filename API failure on every OS, not just the host.
    monkeypatch.setattr(cv2, "imread", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "imwrite", lambda *_args, **_kwargs: False)
    assert image_io.imwrite(path, pixels)
    assert np.array_equal(image_io.imread(path, cv2.IMREAD_UNCHANGED), pixels)
    with Image.open(path) as actual:
        actual.load()
        assert actual.size == (42, 28)


def test_unicode_jpeg_codec_parameters_and_grayscale(tmp_path):
    path = tmp_path / "中文文件名.jpeg"
    pixels = np.full((20, 30, 3), (30, 80, 170), np.uint8)
    assert image_io.imwrite(path, pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert image_io.imread(path).shape == (20, 30, 3)
    assert image_io.imread(path, cv2.IMREAD_GRAYSCALE).shape == (20, 30)


def test_missing_corrupt_and_unwritable_paths(tmp_path):
    assert image_io.imread(tmp_path / "不存在.png") is None
    empty = tmp_path / "空文件.png"
    empty.touch()
    assert image_io.imread(empty) is None
    invalid = tmp_path / "无效.png"
    invalid.write_bytes(b"not an image")
    assert image_io.imread(invalid) is None
    assert not image_io.imwrite(tmp_path / "missing" / "图片.png", np.zeros((5, 5, 3), np.uint8))


def test_objectization_uses_unicode_io_even_if_native_filename_apis_fail(tmp_path, monkeypatch):
    root = tmp_path / "我的云端硬盘" / "图片PPT"
    root.mkdir(parents=True)
    source = np.full((90, 150, 3), 255, np.uint8)
    cv2.circle(source, (75, 45), 28, (240, 246, 253), -1)
    path = root / "原图.png"
    assert image_io.imwrite(path, source)
    monkeypatch.setattr(cv2, "imread", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "imwrite", lambda *_args, **_kwargs: False)
    layout = {"elements": []}
    objectize_on_white(path, root / "background.png", layout, root / "assets", "project", 1)
    assert layout["metadata"]["ownerGate"]["unownedPixelCount"] == 0
    assert image_io.imread(root / "background.png") is not None
    assert any(item["type"] == "image" for item in layout["elements"])


def test_services_cannot_regress_to_native_filename_io():
    root = Path(__file__).resolve().parents[1] / "app"
    violating = [str(path.relative_to(root)) for path in root.rglob("*.py")
                 if path != root / "utils" / "image_io.py"
                 and any(token in path.read_text(encoding="utf-8") for token in ("cv2.imread(", "cv2.imwrite("))]
    assert not violating, violating
