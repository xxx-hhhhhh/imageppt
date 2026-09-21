from __future__ import annotations

import cv2
import numpy as np


def build_stroke_mask(image: np.ndarray, bbox: list[float], profile: dict) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = [int(value) for value in profile.get("bbox", bbox)]
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return mask
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ring = np.asarray(profile.get("medianBgr", [255, 255, 255]), dtype=np.float32)
    background_gray = float(cv2.cvtColor(ring.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY)[0, 0])
    delta = np.abs(gray.astype(np.float32) - background_gray)
    threshold = max(12.0, float(np.percentile(delta, 72)) * 0.45)
    ink = (delta >= threshold).astype(np.uint8) * 255
    edges = cv2.Canny(gray, 30, 100)
    local = cv2.bitwise_or(ink, edges)
    local = cv2.morphologyEx(local, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask[y1:y2, x1:x2] = local
    return mask
