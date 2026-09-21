from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.typography.analyzer import analyze_text_style


@dataclass
class OCRResult:
    text: str
    bbox: list[float]
    confidence: float
    style: dict[str, Any]


class OCRProvider:
    name = "none"

    def recognize(self, image_path: Path) -> list[OCRResult]:
        raise NotImplementedError


def _style_from_region(image_path: Path, bbox: list[float], text: str, confidence: float) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        return {"fontFamily": "Microsoft YaHei", "fontClass": "unknown", "fontSize": 24, "fontWeight": 400, "color": "#111827", "align": "left", "verticalAlign": "top", "role": "body_text"}
    x1, y1, x2, y2 = [max(0, int(value)) for value in bbox]
    crop = image[y1:min(image.shape[0], max(y1 + 1, y2)), x1:min(image.shape[1], max(x1 + 1, x2))]
    if crop.size == 0:
        return {"fontFamily": "Microsoft YaHei", "fontClass": "unknown", "fontSize": 24, "fontWeight": 400, "color": "#111827", "align": "left", "verticalAlign": "top", "role": "body_text"}
    return analyze_text_style(text, bbox, image.shape[1], image.shape[0], crop=crop)


class PaddleOCRProvider(OCRProvider):
    name = "paddleocr"

    def __init__(self) -> None:
        module = importlib.import_module("paddleocr")
        PaddleOCR = getattr(module, "PaddleOCR")
        try:
            self.engine = PaddleOCR(lang="ch", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
        except TypeError:
            self.engine = PaddleOCR(lang="ch", use_angle_cls=True, show_log=False)

    def recognize(self, image_path: Path) -> list[OCRResult]:
        raw = self.engine.predict(str(image_path)) if hasattr(self.engine, "predict") else self.engine.ocr(str(image_path), cls=True)
        results: list[OCRResult] = []
        for item in raw or []:
            if isinstance(item, dict):
                boxes = item.get("rec_boxes") or item.get("dt_polys") or item.get("boxes")
                texts = item.get("rec_texts") or item.get("texts") or []
                scores = item.get("rec_scores") or item.get("scores") or []
                for index, polygon in enumerate(boxes or []):
                    points = np.asarray(polygon).reshape(-1, 2)
                    x1, y1 = points.min(axis=0)
                    x2, y2 = points.max(axis=0)
                    text = str(texts[index]) if index < len(texts) else ""
                    score = float(scores[index]) if index < len(scores) else 0.8
                    if text.strip():
                        bbox = [float(x1), float(y1), float(x2), float(y2)]
                        results.append(OCRResult(text.strip(), bbox, score, _style_from_region(image_path, bbox, text, score)))
            elif isinstance(item, list):
                for row in item:
                    if len(row) >= 2:
                        points = np.asarray(row[0]).reshape(-1, 2)
                        payload = row[1]
                        text = str(payload[0]) if isinstance(payload, (list, tuple)) else str(payload)
                        score = float(payload[1]) if isinstance(payload, (list, tuple)) and len(payload) > 1 else 0.8
                        x1, y1 = points.min(axis=0)
                        x2, y2 = points.max(axis=0)
                        bbox = [float(x1), float(y1), float(x2), float(y2)]
                        if text.strip():
                            results.append(OCRResult(text.strip(), bbox, score, _style_from_region(image_path, bbox, text, score)))
        return results


class RapidOCRProvider(OCRProvider):
    name = "rapidocr"

    def __init__(self) -> None:
        module = importlib.import_module("rapidocr_onnxruntime")
        RapidOCR = getattr(module, "RapidOCR")
        self.engine = RapidOCR()

    def recognize(self, image_path: Path) -> list[OCRResult]:
        raw = self.engine(str(image_path))
        rows = raw[0] if isinstance(raw, tuple) else raw
        results: list[OCRResult] = []
        for row in rows or []:
            if isinstance(row, dict):
                polygon = row.get("box") or row.get("boxes")
                text = str(row.get("text", ""))
                score = float(row.get("score", 0.8))
            else:
                polygon = row[0]
                text = str(row[1]) if len(row) > 1 else ""
                score = float(row[2]) if len(row) > 2 else 0.8
            if not text.strip() or polygon is None:
                continue
            points = np.asarray(polygon).reshape(-1, 2)
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            results.append(OCRResult(text.strip(), bbox, score, _style_from_region(image_path, bbox, text, score)))
        return results


def create_ocr_provider(preferred: str = "auto") -> tuple[OCRProvider, list[str]]:
    warnings: list[str] = []
    candidates = [PaddleOCRProvider, RapidOCRProvider] if preferred in ("auto", "paddleocr") else [RapidOCRProvider]
    for provider_class in candidates:
        try:
            return provider_class(), warnings
        except Exception as exc:  # optional providers must never block the pipeline
            warnings.append(f"{provider_class.name if hasattr(provider_class, 'name') else provider_class.__name__} unavailable: {exc}")
    return OCRProvider(), warnings
