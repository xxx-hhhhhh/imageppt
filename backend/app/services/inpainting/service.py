from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.inpainting.provider import create_inpainting_provider
from app.services.ocr.provider import OCRResult
from app.services.background.strategy import reclean_background as reclean_with_strategy
from app.services.background.strategy import restore_background as restore_with_strategy
from app.services.inpainting.qwen_image_edit import repair_complex_text


class InpaintingService:
    def __init__(self, preferred: str = "opencv") -> None:
        self.provider, self.warnings = create_inpainting_provider(preferred)
        self.last_strategies: list[dict] = []
        self.last_stats = {"ghostingRegionsDetected": 0, "ghostingRegionsRecleaned": 0}
        self.force_clean = False
        self.prefer_inpaint = False
        self.ai_repaired_regions = 0

    def create_mask(self, image_path: Path, regions: list[OCRResult]) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        mask = np.zeros_like(image, dtype=np.uint8)
        for region in regions:
            x1, y1, x2, y2 = region.bbox
            pad_x = min(12, max(2, int((x2 - x1) * 0.06)))
            pad_y = max(2, int((y2 - y1) * 0.25))
            points = [max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)), min(image.shape[1] - 1, int(x2 + pad_x)), min(image.shape[0] - 1, int(y2 + pad_y))]
            cv2.rectangle(mask, (points[0], points[1]), (points[2], points[3]), 255, -1)
        if regions:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def restore_background(self, image_path: Path, regions: list[OCRResult], output_path: Path, preserve_regions: list[list[float]] | None = None) -> Path:
        restored, self.last_strategies = restore_with_strategy(image_path, regions, output_path, preserve_regions, allow_complex_text_preservation=not self.force_clean, prefer_inpaint=self.prefer_inpaint)
        if self.prefer_inpaint:
            self.ai_repaired_regions = repair_complex_text(image_path, output_path, self.last_strategies)
        self.last_stats = {
            "ghostingRegionsDetected": sum(1 for item in self.last_strategies if item.get("ghostingDetected")),
            "ghostingRegionsRecleaned": sum(1 for item in self.last_strategies if item.get("ghostingRecleaned")),
        }
        return restored

    def reclean_background(self, background_path: Path, bboxes: list[list[float]]) -> int:
        count = reclean_with_strategy(background_path, bboxes)
        self.last_stats["ghostingRegionsDetected"] += count
        self.last_stats["ghostingRegionsRecleaned"] += count
        return count
