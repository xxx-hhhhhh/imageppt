from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from app.services.background.analyzer import analyze_background
from app.services.ocr.provider import OCRResult


def restore_background(
    image_path: Path,
    regions: list[OCRResult],
    output_path: Path,
    preserve_regions: list[list[float]] | None = None,
    allow_complex_text_preservation: bool = True,
    prefer_inpaint: bool = False,
) -> tuple[Path, list[dict]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    result = image.copy()
    strategies: list[dict] = []
    preserved = preserve_regions or []
    active = [region for region in regions if not _inside_preserved(region.bbox, preserved)]

    for region in regions:
        if region not in active:
            strategies.append({
                "text": region.text,
                "bbox": region.bbox,
                "category": "preserved_asset",
                "reconstructionStrategy": "preserve_image",
                "willReconstruct": False,
                "sourceContentPreserved": True,
                "ghostingDetected": False,
                "ghostingRecleaned": False,
            })

    for region in active:
        clean_bbox = _expanded_bbox(region.bbox, [item.bbox for item in active], image.shape, 1.0)
        clean_bbox = _clip_against_preserved(clean_bbox, region.bbox, preserved)
        profile = analyze_background(image, region.bbox)
        if allow_complex_text_preservation and _prefer_source_preservation(profile):
            strategies.append({
                "text": region.text,
                "bbox": region.bbox,
                "cleanBBox": clean_bbox,
                "category": profile["category"],
                "reconstructionStrategy": "preserve_complex_text",
                "willReconstruct": False,
                "sourceContentPreserved": True,
                "ghostingDetected": False,
                "ghostingRecleaned": False,
                "cleanPasses": 0,
            })
            continue
        _clean_region(result, clean_bbox, profile, force_inpaint=prefer_inpaint and profile["category"] not in {"solid", "gradient"})
        strategy = {
            "text": region.text,
            "bbox": region.bbox,
            "cleanBBox": clean_bbox,
            "category": profile["category"],
            "reconstructionStrategy": "native_fill" if profile["category"] == "solid" else "gradient_fill" if profile["category"] == "gradient" else "full_bbox_inpaint",
            "willReconstruct": True,
            "sourceContentPreserved": False,
            "ghostingDetected": False,
            "ghostingRecleaned": False,
            "cleanPasses": 1,
        }
        if _has_text_residual(result, region.bbox, strategy["cleanBBox"]):
            strategy["ghostingDetected"] = True
            larger = _expanded_bbox(region.bbox, [item.bbox for item in active], image.shape, 1.45)
            larger = _clip_against_preserved(larger, region.bbox, preserved)
            second_profile = analyze_background(result, larger)
            _clean_region(result, larger, second_profile, force_inpaint=second_profile["category"] not in {"solid", "gradient"})
            strategy["cleanBBox"] = larger
            strategy["cleanPasses"] = 2
            strategy["ghostingRecleaned"] = True
        strategies.append(strategy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    return output_path, strategies


def reclean_background(background_path: Path, bboxes: Iterable[list[float]]) -> int:
    image = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if image is None:
        return 0
    boxes = [list(map(float, bbox)) for bbox in bboxes]
    for bbox in boxes:
        expanded = _expanded_bbox(bbox, boxes, image.shape, 1.45)
        profile = analyze_background(image, expanded)
        _clean_region(image, expanded, profile, force_inpaint=profile["category"] not in {"solid", "gradient"})
    if boxes:
        cv2.imwrite(str(background_path), image)
    return len(boxes)


def _clean_region(image: np.ndarray, bbox: list[int], profile: dict, *, force_inpaint: bool = False) -> None:
    x1, y1, x2, y2 = bbox
    category = profile["category"]
    if category == "solid" and not force_inpaint:
        image[y1:y2, x1:x2] = np.asarray(profile["medianBgr"], dtype=np.uint8)
        return
    if category == "gradient" and not force_inpaint:
        gradient_profile = {**profile, "bbox": bbox}
        _paint_gradient(image, gradient_profile)
        return
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), 255, -1)
    cleaned = cv2.inpaint(image, mask, 4.0, cv2.INPAINT_TELEA)
    image[y1:y2, x1:x2] = cleaned[y1:y2, x1:x2]


def _expanded_bbox(bbox: list[float], neighbors: list[list[float]], shape: tuple[int, ...], scale: float) -> list[int]:
    height, width = shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    box_width, box_height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad_x = min(12.0 * scale, max(2.0, box_width * 0.05 * scale))
    pad_y = max(2.0, box_height * 0.25 * scale)
    left, top, right, bottom = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y
    for other in neighbors:
        ox1, oy1, ox2, oy2 = map(float, other)
        if other is bbox or (ox1 == x1 and oy1 == y1 and ox2 == x2 and oy2 == y2):
            continue
        vertical_overlap = min(y2, oy2) - max(y1, oy1)
        if vertical_overlap > min(box_height, oy2 - oy1) * 0.25:
            if ox2 <= x1 and left < (ox2 + x1) / 2:
                left = (ox2 + x1) / 2
            elif ox1 >= x2 and right > (x2 + ox1) / 2:
                right = (x2 + ox1) / 2
        horizontal_overlap = min(x2, ox2) - max(x1, ox1)
        if horizontal_overlap > min(box_width, ox2 - ox1) * 0.25:
            if oy2 <= y1 and top < (oy2 + y1) / 2:
                top = (oy2 + y1) / 2
            elif oy1 >= y2 and bottom > (y2 + oy1) / 2:
                bottom = (y2 + oy1) / 2
    return [max(0, int(left)), max(0, int(top)), min(width, int(np.ceil(right))), min(height, int(np.ceil(bottom)))]


def _has_text_residual(image: np.ndarray, bbox: list[float], clean_bbox: list[int]) -> bool:
    x1, y1, x2, y2 = [int(max(0, value)) for value in bbox]
    crop = image[y1:y2, x1:x2]
    if crop.size < 27:
        return False
    cx1, cy1, cx2, cy2 = clean_bbox
    ring_mask = np.ones((max(1, cy2 - cy1), max(1, cx2 - cx1)), dtype=np.uint8)
    ix1, iy1, ix2, iy2 = max(0, x1 - cx1), max(0, y1 - cy1), min(cx2 - cx1, x2 - cx1), min(cy2 - cy1, y2 - cy1)
    ring_mask[iy1:iy2, ix1:ix2] = 0
    clean_crop = image[cy1:cy2, cx1:cx2]
    ring_pixels = clean_crop[ring_mask > 0] if clean_crop.size else np.empty((0, 3), dtype=np.uint8)
    background = np.median(ring_pixels, axis=0) if len(ring_pixels) else np.median(crop.reshape(-1, 3), axis=0)
    contrast = np.linalg.norm(crop.astype(np.float32) - background.astype(np.float32), axis=2)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edge_ratio = float((cv2.Canny(gray, 40, 120) > 0).mean())
    contrast_ratio = float((contrast > 28).mean())
    return edge_ratio > 0.075 and contrast_ratio > 0.08


def _inside_preserved(bbox: list[float], preserved: list[list[float]]) -> bool:
    x1, y1, x2, y2 = map(float, bbox)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return any(float(px1) <= cx <= float(px2) and float(py1) <= cy <= float(py2) for px1, py1, px2, py2 in preserved)


def _prefer_source_preservation(profile: dict) -> bool:
    bgr = np.asarray(profile.get("medianBgr", [255, 255, 255]), dtype=np.uint8).reshape(1, 1, 3)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    saturation, value = int(hsv[1]), int(hsv[2])
    if saturation >= 65 or value <= 170:
        return True
    return float(profile.get("texture", 0.0)) >= 2400 and saturation >= 20 and value < 250


def _clip_against_preserved(clean_bbox: list[int], source_bbox: list[float], preserved: list[list[float]]) -> list[int]:
    left, top, right, bottom = clean_bbox
    sx1, sy1, sx2, sy2 = map(float, source_bbox)
    for px1, py1, px2, py2 in preserved:
        horizontal_overlap = min(right, float(px2)) - max(left, float(px1))
        vertical_overlap = min(bottom, float(py2)) - max(top, float(py1))
        if horizontal_overlap <= 0 or vertical_overlap <= 0:
            continue
        if float(py2) <= sy1 + 1:
            top = max(top, int(np.ceil(float(py2))))
        elif float(py1) >= sy2 - 1:
            bottom = min(bottom, int(float(py1)))
        elif float(px2) <= sx1 + 1:
            left = max(left, int(np.ceil(float(px2))))
        elif float(px1) >= sx2 - 1:
            right = min(right, int(float(px1)))
    return [left, top, max(left + 1, right), max(top + 1, bottom)]


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


__all__ = ["restore_background", "reclean_background"]
