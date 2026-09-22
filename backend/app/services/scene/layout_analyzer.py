from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.paddle_runtime import pp_structure_v3_available, predict_structure, set_active_layout_provider


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
            regions.append({"id": f"layout_{index + 1:03d}", "type": role, "role": role, "bbox": {"left": float(x), "top": float(y), "width": float(w), "height": float(h)}, "confidence": 0.45, "source": "opencv", "readingOrder": None})
        return regions


class PPStructureV3Provider(LayoutProvider):
    name = "pp-structure-v3"

    def __init__(self) -> None:
        if not pp_structure_v3_available():
            raise ImportError("PPStructureV3 is not available in installed paddleocr")

    def analyze(self, image_path: Path) -> list[dict[str, Any]]:
        raw = predict_structure(str(image_path))
        return _normalize_pp_structure(raw)


class AutoLayoutProvider(LayoutProvider):
    def __init__(self) -> None:
        self.primary = PPStructureV3Provider()
        self.fallback = OpenCVLayoutProvider()
        self.active: LayoutProvider = self.primary
        self.warnings: list[str] = []

    @property
    def name(self) -> str:
        return self.active.name

    def analyze(self, image_path: Path) -> list[dict[str, Any]]:
        try:
            regions = self.primary.analyze(image_path)
            self.active = self.primary
            return regions
        except Exception as exc:
            self.active = self.fallback
            set_active_layout_provider("opencv")
            warning = f"PP-StructureV3 unavailable, using OpenCV layout fallback: {exc}"
            if warning not in self.warnings:
                self.warnings.append(warning)
            return self.fallback.analyze(image_path)


def create_layout_provider(preferred: str = "auto") -> tuple[LayoutProvider, list[str]]:
    warnings: list[str] = []
    if preferred in {"auto", "pp-structure-v3"}:
        try:
            return AutoLayoutProvider(), warnings
        except Exception as exc:
            warnings.append(f"PP-StructureV3 unavailable, using OpenCV layout fallback: {exc}")
    return OpenCVLayoutProvider(), warnings


def _normalize_pp_structure(raw: list[Any]) -> list[dict[str, Any]]:
    detected: list[dict[str, Any]] = []
    parsing: list[dict[str, Any]] = []
    for item in raw or []:
        payload = item if isinstance(item, dict) else getattr(item, "json", None)
        if callable(payload):
            payload = payload()
        if not isinstance(payload, dict):
            continue
        result = payload.get("res") if isinstance(payload.get("res"), dict) else payload
        layout = result.get("layout_det_res") or {}
        detected.extend(box for box in (layout.get("boxes") or []) if isinstance(box, dict))
        parsing.extend(block for block in (result.get("parsing_res_list") or []) if isinstance(block, dict))

    regions: list[dict[str, Any]] = []
    source_items = detected or parsing
    for index, item in enumerate(source_items, start=1):
        raw_label = str(item.get("label") or item.get("block_label") or "content_region")
        raw_bbox = item.get("coordinate") or item.get("block_bbox") or item.get("bbox")
        bbox = _normalize_bbox(raw_bbox)
        if bbox is None:
            continue
        reading_order = item.get("block_order")
        if reading_order is None and parsing:
            reading_order = _matching_reading_order(raw_label, raw_bbox, parsing)
        region_type = _normalize_region_type(raw_label)
        regions.append({
            "id": f"layout_{index:03d}",
            "type": region_type,
            "role": region_type,
            "bbox": bbox,
            "confidence": float(item.get("score", 0.7)),
            "source": "pp-structure-v3",
            "readingOrder": reading_order,
        })
    return regions


def _normalize_bbox(value: Any) -> dict[str, float] | None:
    if isinstance(value, dict):
        left = float(value.get("left", value.get("x", 0)))
        top = float(value.get("top", value.get("y", 0)))
        width = float(value.get("width", 0))
        height = float(value.get("height", 0))
        return {"left": left, "top": top, "width": width, "height": height}
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        x1, y1, x2, y2 = map(float, value[:4])
        return {"left": x1, "top": y1, "width": max(0.0, x2 - x1), "height": max(0.0, y2 - y1)}
    return None


def _normalize_region_type(label: str) -> str:
    aliases = {
        "paragraph_title": "title",
        "doc_title": "title",
        "figure": "figure",
        "chart": "figure",
        "image": "image",
        "table": "table",
        "header": "header",
        "footer": "footer",
        "text": "text",
    }
    return aliases.get(label.lower(), "content_region")


def _matching_reading_order(label: str, bbox: Any, parsing: list[dict[str, Any]]) -> Any:
    normalized = _normalize_bbox(bbox)
    if normalized is None:
        return None
    center = (normalized["left"] + normalized["width"] / 2, normalized["top"] + normalized["height"] / 2)
    candidates = [item for item in parsing if str(item.get("block_label")) == label]
    if not candidates:
        return None
    match = min(candidates, key=lambda item: _center_distance(center, _normalize_bbox(item.get("block_bbox"))))
    return match.get("block_order")


def _center_distance(center: tuple[float, float], bbox: dict[str, float] | None) -> float:
    if bbox is None:
        return float("inf")
    other = (bbox["left"] + bbox["width"] / 2, bbox["top"] + bbox["height"] / 2)
    return (center[0] - other[0]) ** 2 + (center[1] - other[1]) ** 2
