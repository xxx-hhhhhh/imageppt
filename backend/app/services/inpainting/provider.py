from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class InpaintingProvider:
    name = "none"

    def inpaint(self, image_path: Path, mask: np.ndarray, output_path: Path) -> Path:
        raise NotImplementedError


class OpenCVInpaintingProvider(InpaintingProvider):
    name = "opencv"

    def inpaint(self, image_path: Path, mask: np.ndarray, output_path: Path) -> Path:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        output = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output)
        return output_path


class LamaInpaintingProvider(InpaintingProvider):
    name = "lama"

    def __init__(self) -> None:
        from simple_lama_inpainting import SimpleLama
        self.engine = SimpleLama()

    def inpaint(self, image_path: Path, mask: np.ndarray, output_path: Path) -> Path:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        mask_image = Image.fromarray(mask)
        output = self.engine(image, mask_image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(output_path)
        return output_path


def create_inpainting_provider(preferred: str = "opencv") -> tuple[InpaintingProvider, list[str]]:
    warnings: list[str] = []
    if preferred == "lama":
        try:
            return LamaInpaintingProvider(), warnings
        except Exception as exc:
            warnings.append(f"LAMA unavailable, using OpenCV inpaint: {exc}")
    return OpenCVInpaintingProvider(), warnings

