from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class LayoutProvider(ABC):
    name = "base"

    @abstractmethod
    def analyze(self, image_path: Path) -> list[dict[str, Any]]:
        raise NotImplementedError


class OpenCVLayoutProvider(LayoutProvider):
    name = "opencv"

    def analyze(self, image_path: Path) -> list[dict[str, Any]]:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return []
        height, width = image.shape[:2]
        edges = cv2.Canny(image, 60, 160)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: list[dict[str, Any]] = []
        for index, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:40]):
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < width * height * 0.015 or w < 24 or h < 18:
                continue
            ratio = w / max(1, h)
            role = "wide_region" if ratio > 2.8 else "tall_region" if ratio < 0.35 else "content_region"
            regions.append({"id": f"region_{index + 1:03d}", "role": role, "bbox": {"left": float(x), "top": float(y), "width": float(w), "height": float(h)}, "confidence": 0.45})
        return regions


class PPStructureV3Provider(LayoutProvider):
    name = "pp-structure-v3"

    def __init__(self) -> None:
        module = __import__("paddleocr")
        structure = getattr(module, "PPStructureV3", None)
        if structure is None:
            raise ImportError("PPStructureV3 is not available in installed paddleocr")
        self.engine = structure()

    def analyze(self, image_path: Path) -> list[dict[str, Any]]:
        raw = self.engine.predict(str(image_path))
        return [{"id": f"provider_region_{index + 1:03d}", "role": str(item.get("type", "content_region")), "bbox": item.get("bbox", {}), "confidence": float(item.get("score", 0.7))} for index, item in enumerate(raw or []) if isinstance(item, dict)]


def create_layout_provider(preferred: str = "auto") -> tuple[LayoutProvider, list[str]]:
    warnings: list[str] = []
    if preferred in {"auto", "pp-structure-v3"}:
        try:
            return PPStructureV3Provider(), warnings
        except Exception as exc:
            warnings.append(f"PP-StructureV3 unavailable, using OpenCV layout fallback: {exc}")
    return OpenCVLayoutProvider(), warnings
