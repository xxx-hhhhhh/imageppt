from __future__ import annotations
from app.utils import image_io

from pathlib import Path
import tempfile

import cv2
import numpy as np

from app.services.inpainting.provider import InpaintingProvider, LamaInpaintingProvider, create_inpainting_provider
from app.services.inpainting.local_client import LocalIOPaintClient
from app.services.ocr.provider import OCRResult
from app.services.background.strategy import reclean_background as reclean_with_strategy
from app.services.background.strategy import restore_background as restore_with_strategy


class InpaintingService:
    def __init__(self, preferred: str = "opencv") -> None:
        self.provider, self.warnings = create_inpainting_provider(preferred)
        self.last_strategies: list[dict] = []
        self.last_stats = {"ghostingRegionsDetected": 0, "ghostingRegionsRecleaned": 0}
        self.force_clean = False
        self.prefer_inpaint = False
        self.ai_repaired_regions = 0
        self.professional_attempts = 0
        self.professional_pending = 0
        self.professional_provider_name = "none"

    def create_mask(self, image_path: Path, regions: list[OCRResult]) -> np.ndarray:
        image = image_io.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
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
        if isinstance(self.provider, LocalIOPaintClient):
            self.ai_repaired_regions = self._repair_local_text_regions(image_path, output_path)
        elif self.prefer_inpaint:
            self.ai_repaired_regions = self._repair_complex_regions(image_path, output_path)
        self.last_stats = {
            "ghostingRegionsDetected": sum(1 for item in self.last_strategies if item.get("ghostingDetected")),
            "ghostingRegionsRecleaned": sum(1 for item in self.last_strategies if item.get("ghostingRecleaned")),
        }
        return restored

    def _professional_provider(self) -> InpaintingProvider | None:
        if self.provider.name in {"lama", "stability", "local_lama"}:
            return self.provider
        try:
            return LamaInpaintingProvider()
        except (ImportError, OSError, RuntimeError):
            return None

    def _repair_local_text_regions(self, source_path: Path, background_path: Path) -> int:
        targets = [item for item in self.last_strategies if item.get("willReconstruct") and item.get("cleanBBox")]
        if not targets:
            return 0
        source = image_io.imread(str(source_path), cv2.IMREAD_COLOR)
        background = image_io.imread(str(background_path), cv2.IMREAD_COLOR)
        if source is None or background is None:
            return 0
        mask = np.zeros(source.shape[:2], dtype=np.uint8)
        for item in targets:
            x1, y1, x2, y2 = map(int, item["cleanBBox"])
            mask[max(0, y1):min(mask.shape[0], y2), max(0, x1):min(mask.shape[1], x2)] = 255
        try:
            candidate = self.provider.inpaint_array(source, mask)
            background[mask > 0] = candidate[mask > 0]
            image_io.imwrite(str(background_path), background)
            for item in targets:
                item["professionalRepair"] = "accepted"
                item["reconstructionStrategy"] = "local_lama"
            self.professional_provider_name = "local_lama"
            return len(targets)
        except Exception:
            for item in targets:
                item["professionalRepair"] = "fallback_opencv"
            return 0

    def clean_array(self, image: np.ndarray, mask: np.ndarray, *, owner_mask: np.ndarray | None = None, protected_mask: np.ndarray | None = None) -> np.ndarray:
        # A caller must supply replacement-owner evidence; requests without it
        # preserve the input. LaMa is not a generic module-cleaning operation.
        if owner_mask is None or owner_mask.shape != mask.shape:
            return image.copy()
        mask = np.uint8((mask > 0) & (owner_mask > 0)) * 255
        if protected_mask is not None:
            if protected_mask.shape != mask.shape:
                return image.copy()
            mask[protected_mask > 0] = 0
        if not np.any(mask):
            return image.copy()
        if isinstance(self.provider, LocalIOPaintClient):
            try:
                candidate = self.provider.inpaint_array(image, mask)
                return np.where(mask[:, :, None] > 0, candidate, image)
            except Exception:
                pass
        return cv2.inpaint(image, mask, 4, cv2.INPAINT_TELEA)

    def _repair_complex_regions(self, source_path: Path, background_path: Path) -> int:
        targets = [item for item in self.last_strategies if item.get("willReconstruct") and item.get("category") in {"complex", "texture"}]
        if not targets:
            return 0
        provider = self._professional_provider()
        self.professional_provider_name = provider.name if provider else "unavailable"
        if provider is None:
            for item in targets:
                item["professionalRepair"] = "unavailable"
            self.professional_pending = len(targets)
            return 0
        source = image_io.imread(str(source_path), cv2.IMREAD_COLOR)
        background = image_io.imread(str(background_path), cv2.IMREAD_COLOR)
        if source is None or background is None:
            self.professional_pending = len(targets)
            return 0
        repaired = 0
        with tempfile.TemporaryDirectory(prefix="imageppt-inpaint-") as workspace:
            candidate_path = Path(workspace) / "candidate.png"
            for item in targets:
                box = item.get("cleanBBox") or item.get("bbox")
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    item["professionalRepair"] = "invalid_mask"
                    continue
                x1, y1, x2, y2 = (int(value) for value in box)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(source.shape[1], x2), min(source.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    item["professionalRepair"] = "invalid_mask"
                    continue
                mask = np.zeros(source.shape[:2], dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                self.professional_attempts += 1
                try:
                    provider.inpaint(source_path, mask, candidate_path)
                    candidate = image_io.imread(str(candidate_path), cv2.IMREAD_COLOR)
                    if candidate is None or candidate.shape != source.shape:
                        raise ValueError("Inpainting output dimensions changed")
                    before = cv2.Canny(source[y1:y2, x1:x2], 50, 150)
                    after = cv2.Canny(candidate[y1:y2, x1:x2], 50, 150)
                    if np.count_nonzero(before) > 5 and np.count_nonzero(after) > np.count_nonzero(before) * 0.9:
                        item["professionalRepair"] = "rejected"
                        continue
                    background[y1:y2, x1:x2] = candidate[y1:y2, x1:x2]
                    item["professionalRepair"] = "accepted"
                    item["reconstructionStrategy"] = "professional_inpaint"
                    repaired += 1
                except Exception:
                    item["professionalRepair"] = "failed"
        self.professional_pending = len(targets) - repaired
        if repaired:
            image_io.imwrite(str(background_path), background)
        return repaired

    def reclean_background(self, background_path: Path, bboxes: list[list[float]]) -> int:
        count = 0
        if isinstance(self.provider, LocalIOPaintClient) and bboxes:
            image = image_io.imread(str(background_path), cv2.IMREAD_COLOR)
            if image is not None:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                for box in bboxes:
                    x1, y1, x2, y2 = map(int, box)
                    mask[max(0, y1):min(mask.shape[0], y2), max(0, x1):min(mask.shape[1], x2)] = 255
                try:
                    candidate = self.provider.inpaint_array(image, mask)
                    image[mask > 0] = candidate[mask > 0]
                    image_io.imwrite(str(background_path), image)
                    count = len(bboxes)
                except Exception:
                    pass
        if not count:
            count = reclean_with_strategy(background_path, bboxes)
        self.last_stats["ghostingRegionsDetected"] += count
        self.last_stats["ghostingRegionsRecleaned"] += count
        return count
