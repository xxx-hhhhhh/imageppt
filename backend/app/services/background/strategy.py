from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.background.analyzer import analyze_background
from app.services.background.mask_builder import build_stroke_mask
from app.services.ocr.provider import OCRResult


def restore_background(image_path: Path, regions: list[OCRResult], output_path: Path) -> tuple[Path, list[dict]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    result = image.copy()
    complex_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    strategies = []
    for region in regions:
        profile = analyze_background(image, region.bbox)
        category = profile["category"]
        x1, y1, x2, y2 = [int(value) for value in profile["bbox"]]
        if category == "solid":
            result[y1:y2, x1:x2] = np.asarray(profile["medianBgr"], dtype=np.uint8)
        elif category == "gradient":
            _paint_gradient(result, profile)
        else:
            complex_mask = cv2.bitwise_or(complex_mask, build_stroke_mask(image, region.bbox, profile))
        strategies.append({"text": region.text, "bbox": region.bbox, "category": category, "reconstructionStrategy": "native_fill" if category == "solid" else "gradient_fill" if category == "gradient" else "local_inpaint"})
    if np.any(complex_mask):
        result = cv2.inpaint(result, complex_mask, 2.0, cv2.INPAINT_TELEA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    return output_path, strategies


def _paint_gradient(image: np.ndarray, profile: dict) -> None:
    x1, y1, x2, y2 = profile["bbox"]
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    left = np.asarray(profile["leftColor"], dtype=np.float32)
    right = np.asarray(profile["rightColor"], dtype=np.float32)
    top = np.asarray(profile["topColor"], dtype=np.float32)
    bottom = np.asarray(profile["bottomColor"], dtype=np.float32)
    horizontal = np.linalg.norm(left - right) >= np.linalg.norm(top - bottom)
    for offset in range(width if horizontal else height):
        ratio = offset / max(1, (width if horizontal else height) - 1)
        color = left * (1 - ratio) + right * ratio if horizontal else top * (1 - ratio) + bottom * ratio
        if horizontal:
            image[y1:y2, x1 + offset] = np.asarray(color, dtype=np.uint8)
        else:
            image[y1 + offset, x1:x2] = np.asarray(color, dtype=np.uint8)
