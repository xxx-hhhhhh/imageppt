from __future__ import annotations

from pathlib import Path
from io import BytesIO
import os

import cv2
import numpy as np

from app.config import LOCAL_INPAINT_ENABLED, LOCAL_INPAINT_MODEL, LOCAL_INPAINT_URL
from app.services.inpainting.local_client import LocalIOPaintClient


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


class StabilityInpaintingProvider(InpaintingProvider):
    """Explicit opt-in masked image restoration using Stability AI's inpaint API."""

    name = "stability"
    endpoint = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("STABILITY_API_KEY", "")
        if not self.api_key:
            raise ValueError("STABILITY_API_KEY is not configured")

    def inpaint(self, image_path: Path, mask: np.ndarray, output_path: Path) -> Path:
        import requests
        from PIL import Image

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if mask.shape != (image.height, image.width):
            raise ValueError("Inpainting mask size does not match source image")
        image_bytes, mask_bytes = BytesIO(), BytesIO()
        image.save(image_bytes, format="PNG")
        Image.fromarray(mask).save(mask_bytes, format="PNG")
        response = requests.post(
            self.endpoint,
            headers={"authorization": f"Bearer {self.api_key}", "accept": "image/*"},
            files={"image": ("source.png", image_bytes.getvalue(), "image/png"), "mask": ("mask.png", mask_bytes.getvalue(), "image/png")},
            data={"prompt": "Restore the original background texture behind removed printed text. Preserve all surrounding visual details and add no text.", "output_format": "png", "grow_mask": "0"},
            timeout=120,
        )
        response.raise_for_status()
        with Image.open(BytesIO(response.content)) as edited:
            result = edited.convert("RGB")
            if result.size != image.size:
                result = result.resize(image.size, Image.Resampling.LANCZOS)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path)
        return output_path


def create_inpainting_provider(preferred: str = "opencv") -> tuple[InpaintingProvider, list[str]]:
    warnings: list[str] = []
    if LOCAL_INPAINT_ENABLED:
        local = LocalIOPaintClient(LOCAL_INPAINT_URL, LOCAL_INPAINT_MODEL)
        status = local.probe()
        if status["connected"]:
            return local, warnings
        warnings.append("Local LaMa unavailable; using existing inpainting fallback")
    if preferred == "stability":
        try:
            return StabilityInpaintingProvider(), warnings
        except ValueError as exc:
            warnings.append(f"Professional inpainting unavailable: {exc}")
    if preferred == "lama":
        try:
            return LamaInpaintingProvider(), warnings
        except Exception as exc:
            warnings.append(f"LAMA unavailable, using OpenCV inpaint: {exc}")
    return OpenCVInpaintingProvider(), warnings

