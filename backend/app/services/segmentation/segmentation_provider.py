from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class SegmentationProvider(ABC):
    name = "base"

    @abstractmethod
    def segment(self, image_path: Path, asset_dir: Path | None = None, project_id: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError


class OpenCVSegmentationProvider(SegmentationProvider):
    name = "opencv"

    def segment(self, image_path: Path, asset_dir: Path | None = None, project_id: str | None = None) -> list[dict[str, Any]]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foreground = cv2.Canny(gray, 80, 180)
        kernel = np.ones((5, 5), np.uint8)
        foreground = cv2.dilate(foreground, kernel, iterations=1)
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = image.shape[:2]
        objects: list[dict[str, Any]] = []
        for index, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:30]):
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < width * height * 0.01 or w < 16 or h < 16 or w > width * 0.92 or h > height * 0.92:
                continue
            mask = np.zeros((h, w), np.uint8)
            shifted = contour - np.array([[x, y]])
            cv2.drawContours(mask, [shifted], -1, 255, -1)
            item: dict[str, Any] = {"id": f"segment_{index + 1:03d}", "bbox": {"left": float(x), "top": float(y), "width": float(w), "height": float(h)}, "confidence": 0.4, "mask": mask.tolist()}
            if asset_dir and project_id:
                asset_dir.mkdir(parents=True, exist_ok=True)
                rgba = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = mask
                path = asset_dir / f"segment_{index + 1:03d}.png"
                cv2.imwrite(str(path), rgba)
                item["alphaCrop"] = f"/media/assets/{project_id}/{path.name}"
            objects.append(item)
        return objects


class OptionalSAM2Provider(OpenCVSegmentationProvider):
    name = "sam2-fallback"


def create_segmentation_provider(preferred: str = "auto") -> tuple[SegmentationProvider, list[str]]:
    warnings: list[str] = []
    if preferred in {"sam", "sam2"}:
        warnings.append("SAM/SAM2 is optional; using OpenCV segmentation fallback")
    return OpenCVSegmentationProvider(), warnings
