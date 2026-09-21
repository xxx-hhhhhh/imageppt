import shutil

from PIL import Image

from app.services.vision.test_assets import create_test_image


def test_vision_test_image_can_be_reopened():
    path = create_test_image()
    try:
        data = path.read_bytes()
        assert len(data) > 0
        with Image.open(path) as image:
            assert image.size == (32, 32)
    finally:
        shutil.rmtree(path.parent, ignore_errors=True)


def test_vision_test_image_can_be_reopened_three_times():
    for _ in range(3):
        path = create_test_image()
        try:
            with Image.open(path) as image:
                assert image.size == (32, 32)
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)
