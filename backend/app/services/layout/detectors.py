from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.ocr.provider import OCRResult


def _hex(rgb: np.ndarray) -> str:
    values = [int(max(0, min(255, value))) for value in rgb.tolist()]
    return "#%02X%02X%02X" % tuple(values)


def detect_text_elements(regions: list[OCRResult]) -> list[dict[str, Any]]:
    elements = []
    for index, region in enumerate(regions):
        x1, y1, x2, y2 = region.bbox
        role = (region.style or {}).get("role", "body_text")
        component_type = "titleText" if role in {"main_title", "subtitle", "card_title", "label_text"} else "bodyText"
        elements.append({
            "id": f"text_{index + 1:03d}",
            "type": "text",
            "x": x1,
            "y": y1,
            "width": max(4, x2 - x1),
            "height": max(4, y2 - y1),
            "rotation": 0,
            "zIndex": 20,
            "text": region.text,
            "confidence": region.confidence,
            "style": region.style,
            "role": role,
            "componentType": component_type,
            "lines": [{"text": region.text, "bbox": list(region.bbox), "baseline": float(y2), "height": float(y2 - y1)}],
            "metadata": {"role": role, "componentType": component_type},
        })
    return elements


def _color_at(image: np.ndarray, x: int, y: int) -> str:
    crop = image[max(0, y - 2):min(image.shape[0], y + 3), max(0, x - 2):min(image.shape[1], x + 3)]
    if crop.size == 0:
        return "#DCE6F1"
    return _hex(crop[:, :, ::-1].reshape(-1, 3).mean(axis=0))


def detect_simple_shapes(image_path: Path, text_regions: list[OCRResult]) -> list[dict[str, Any]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return []
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elements: list[dict[str, Any]] = []
    next_id = 1
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if w < 30 or h < 20 or area < width * height * 0.005 or area > width * height * 0.75:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True) if perimeter else contour
        contour_area = max(1.0, cv2.contourArea(contour))
        circularity = float(4 * np.pi * contour_area / max(1.0, perimeter * perimeter))
        fill_ratio = float(contour_area / max(1, w * h))
        fill = _sample_shape_fill(image, x, y, w, h, text_regions)
        if len(approx) >= 8 and 0.72 < w / max(1, h) < 1.38 and circularity >= 0.58 and fill_ratio >= 0.45:
            shape_type = "ellipse"
        elif len(approx) == 4:
            radius_hint = min(w, h) * 0.12
            shape_type = "roundedRectangle" if radius_hint > 5 else "rectangle"
        else:
            continue
        elements.append({
            "id": f"shape_{next_id:03d}",
            "type": shape_type,
            "x": float(x), "y": float(y), "width": float(w), "height": float(h),
            "rotation": 0, "zIndex": 10,
            "style": {"fill": fill, "stroke": fill, "strokeWidth": 1, "opacity": 1},
            "confidence": round(max(0.42, min(0.9, 0.45 + circularity * 0.35 + fill_ratio * 0.2)), 3),
        })
        next_id += 1
    # Closed badge outlines are sometimes interrupted by a white icon and do
    # not survive contour approximation. Hough candidates recover those full
    # circular components, then strict edge support and deduplication keep the
    # detector generic.
    for x, y, w, h, support in _hough_badges(image, gray, edges):
        if any(_box_iou((x, y, w, h), (item["x"], item["y"], item["width"], item["height"])) > 0.65 for item in elements):
            continue
        fill = _sample_shape_fill(image, x, y, w, h, text_regions)
        elements.append({
            "id": f"shape_{next_id:03d}",
            "type": "ellipse",
            "x": float(x), "y": float(y), "width": float(w), "height": float(h),
            "rotation": 0, "zIndex": 10,
            "style": {"fill": fill, "stroke": fill, "strokeWidth": 1, "opacity": 1},
            "confidence": round(min(0.88, 0.62 + support * 0.25), 3),
        })
        next_id += 1
    return elements


def _hough_badges(image: np.ndarray, gray: np.ndarray, edges: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    height, width = gray.shape[:2]
    minimum = max(24, int(min(width, height) * 0.04))
    maximum = max(minimum + 2, int(min(width, height) * 0.09))
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.8)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.35,
        minDist=max(36, minimum * 2),
        param1=100,
        param2=42,
        minRadius=minimum,
        maxRadius=maximum,
    )
    if circles is None:
        return []
    result: list[tuple[int, int, int, int, float]] = []
    for cx, cy, radius in np.round(circles[0]).astype(int):
        samples = 96
        hits = 0
        valid = 0
        for angle in np.linspace(0, 2 * np.pi, samples, endpoint=False):
            px, py = int(round(cx + radius * np.cos(angle))), int(round(cy + radius * np.sin(angle)))
            if 2 <= px < width - 2 and 2 <= py < height - 2:
                valid += 1
                if np.any(edges[py - 2:py + 3, px - 2:px + 3] > 0):
                    hits += 1
        support = hits / max(1, valid)
        if support < 0.28:
            continue
        x, y = max(0, cx - radius), max(0, cy - radius)
        x2, y2 = min(width, cx + radius), min(height, cy + radius)
        crop = image[y:y2, x:x2]
        if crop.size == 0:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        yy, xx = np.ogrid[:crop.shape[0], :crop.shape[1]]
        local_cx, local_cy = cx - x, cy - y
        disk = (xx - local_cx) ** 2 + (yy - local_cy) ** 2 <= (radius * 0.88) ** 2
        saturation = hsv[:, :, 1][disk]
        if len(saturation) == 0 or float((saturation > 70).mean()) < 0.42 or float(np.mean(saturation)) < 75:
            continue
        result.append((x, y, x2 - x, y2 - y, support))
    return result


def _box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    overlap = max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    union = lw * lh + rw * rh - overlap
    return overlap / max(1.0, union)


def _sample_shape_fill(image: np.ndarray, x: int, y: int, width: int, height: int, text_regions: list[OCRResult]) -> str:
    margin_x, margin_y = max(2, int(width * 0.16)), max(2, int(height * 0.16))
    crop = image[y + margin_y:min(image.shape[0], y + height - margin_y), x + margin_x:min(image.shape[1], x + width - margin_x)]
    if crop.size == 0:
        return _color_at(image, x + width // 2, y + height // 2)
    mask = np.ones(crop.shape[:2], dtype=np.uint8)
    for region in text_regions:
        rx1, ry1, rx2, ry2 = [int(value) for value in region.bbox]
        ix1, iy1 = max(0, rx1 - x - margin_x), max(0, ry1 - y - margin_y)
        ix2, iy2 = min(crop.shape[1], rx2 - x - margin_x), min(crop.shape[0], ry2 - y - margin_y)
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = 0
    pixels = crop[mask > 0].reshape(-1, 3)
    if len(pixels) == 0:
        pixels = crop.reshape(-1, 3)
    bgr = np.median(pixels, axis=0).astype(int)
    return _hex(bgr[::-1])


def detect_image_regions(image_path: Path, text_regions: list[OCRResult], shapes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return []
    height, width = image.shape[:2]
    # Heuristic regions: large, textured rectangular components. Text and simple
    # shapes are excluded so photos/logos stay independent assets.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture = cv2.Laplacian(gray, cv2.CV_64F).var()
    if texture < 80:
        return []
    edges = cv2.Canny(gray, 80, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[dict[str, Any]] = []
    excluded = [(r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3]) for r in text_regions]
    excluded += [(s["x"], s["y"], s["x"] + s["width"], s["y"] + s["height"]) for s in shapes]
    for index, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:20]):
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.12 or h < height * 0.12 or w * h < width * height * 0.04:
            continue
        if any(x < ex2 and x + w > ex1 and y < ey2 and y + h > ey1 for ex1, ey1, ex2, ey2 in excluded):
            continue
        boxes.append({
            "id": f"image_{index + 1:03d}", "type": "image",
            "x": float(x), "y": float(y), "width": float(w), "height": float(h),
            "rotation": 0, "zIndex": 8, "confidence": 0.48,
        })
    return boxes[:6]
