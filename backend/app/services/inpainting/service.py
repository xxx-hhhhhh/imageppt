from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from app.services.inpainting.provider import create_inpainting_provider
from app.services.ocr.provider import OCRResult


class InpaintingService:
    def __init__(self, preferred: str = "opencv") -> None:
        self.provider, self.warnings = create_inpainting_provider(preferred)
        self.last_strategies: list[dict] = []
        self.last_stats = {"ghostingRegionsDetected": 0, "ghostingRegionsRecleaned": 0}

    def restore_owned_text(self, image_path: Path, text_mask: np.ndarray, protected_mask: np.ndarray, ledger, output_path: Path) -> Path:
        """One local patch. Protected/unowned pixels are byte-identical afterwards."""
        from app.services.reconstruction.object_first import read_image, write_image
        image = read_image(image_path)
        authorized = ledger.authorize(text_mask)
        authorized[protected_mask > 0] = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not np.any(authorized):
            write_image(output_path, image)
            return output_path
        try:
            self.provider.inpaint(image_path, authorized, output_path)
            patched = read_image(output_path)
        except Exception:  # noqa: BLE001 -- optional model boundary; retain protected pixels
            self.warnings.append("Local inpaint unavailable; using one OpenCV ink-only patch")
            patched = cv2.inpaint(image, authorized, 3, cv2.INPAINT_TELEA)
        patched[authorized == 0] = image[authorized == 0]
        write_image(output_path, patched)
        self.last_strategies = [{"strategy": self.provider.name, "mask": "tight-text-only", "authorizedPixels": int(np.count_nonzero(authorized)), "protectedPixels": int(np.count_nonzero(protected_mask)), "passes": 1}]
        return output_path

    def create_mask(self, image_path: Path, regions: list[OCRResult]) -> np.ndarray:
        from app.services.reconstruction.object_first import read_image, tight_text_mask
        image = read_image(image_path)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for region in regions:
            x1, y1, x2, y2 = region.bbox
            mask |= tight_text_mask(image, {"x": x1, "y": y1, "width": x2-x1, "height": y2-y1})
        return mask

    def restore_background(self, image_path: Path, regions: list[OCRResult], output_path: Path, preserve_regions: list[list[float]] | None = None) -> Path:
        # Legacy caller supplied no ownership proof. Deny destructive processing.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, output_path)
        self.warnings.append("Background deletion denied: replacement ownership required")
        self.last_strategies = []
        return output_path

    def reclean_background(self, background_path: Path, bboxes: list[list[float]]) -> int:
        self.warnings.append("Repeated bbox inpaint denied; revise owned objects instead")
        return 0
