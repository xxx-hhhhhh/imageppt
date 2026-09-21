from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def preprocess_image(image_path: Path, output_path: Path) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.save(output_path, format="PNG")
        return image.width, image.height

