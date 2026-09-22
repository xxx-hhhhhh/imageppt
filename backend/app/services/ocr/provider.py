from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.typography.analyzer import analyze_text_style
from app.services.paddle_runtime import predict_ocr


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
        if importlib.util.find_spec("paddleocr") is None:
            raise ImportError("paddleocr is not installed")

    def recognize(self, image_path: Path) -> list[OCRResult]:
        raw = predict_ocr(str(image_path))
        results: list[OCRResult] = []
        for item in raw or []:
            payload = _paddle_payload(item)
            if payload:
                boxes = payload.get("rec_polys") or payload.get("rec_boxes") or payload.get("dt_polys") or payload.get("boxes")
                texts = payload.get("rec_texts") or payload.get("texts") or []
                scores = payload.get("rec_scores") or payload.get("scores") or []
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


def _paddle_payload(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        payload = item
    else:
        payload = getattr(item, "json", None)
        if callable(payload):
            payload = payload()
    if not isinstance(payload, dict):
        return None
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


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
