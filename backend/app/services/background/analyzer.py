from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def analyze_background(image: np.ndarray, bbox: list[float]) -> dict[str, Any]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(max(0, value)) for value in bbox]
    x1, x2 = min(x1, width - 1), min(max(x2, x1 + 1), width)
    y1, y2 = min(y1, height - 1), min(max(y2, y1 + 1), height)
    pad = max(3, int(min(x2 - x1, y2 - y1) * 0.6))
    samples = []
    for a, b in ((max(0, y1 - pad), y1), (y2, min(height, y2 + pad))):
        if b > a:
            samples.append(image[a:b, x1:x2])
    for a, b in ((max(0, x1 - pad), x1), (x2, min(width, x2 + pad))):
        if b > a:
            samples.append(image[y1:y2, a:b])
    ring = np.concatenate([item.reshape(-1, 3) for item in samples if item.size], axis=0) if samples else image[y1:y2, x1:x2].reshape(-1, 3)
    median = np.median(ring, axis=0)
    variance = float(np.mean(np.var(ring.astype(np.float32), axis=0)))
    distances = np.linalg.norm(ring.astype(np.float32) - median.astype(np.float32), axis=1)
    dominant = ring[distances <= np.percentile(distances, 68)]
    if len(dominant) < 12:
        dominant = ring
    robust_variance = float(np.mean(np.var(dominant.astype(np.float32), axis=0)))
    left = np.median(image[y1:y2, max(0, x1 - pad):x1].reshape(-1, 3), axis=0) if x1 > 0 else median
    right = np.median(image[y1:y2, x2:min(width, x2 + pad)].reshape(-1, 3), axis=0) if x2 < width else median
    top = np.median(image[max(0, y1 - pad):y1, x1:x2].reshape(-1, 3), axis=0) if y1 > 0 else median
    bottom = np.median(image[y2:min(height, y2 + pad), x1:x2].reshape(-1, 3), axis=0) if y2 < height else median
    horizontal_gradient = float(np.linalg.norm(left - right))
    vertical_gradient = float(np.linalg.norm(top - bottom))
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size > 9 else 0.0
    if robust_variance < 180:
        category = "solid"
    elif max(horizontal_gradient, vertical_gradient) >= 12 and robust_variance < 900:
        category = "gradient"
    elif robust_variance < 1400:
        category = "texture"
    else:
        category = "complex"
    return {"category": category, "medianBgr": median.tolist(), "leftColor": left.tolist(), "rightColor": right.tolist(), "topColor": top.tolist(), "bottomColor": bottom.tolist(), "variance": variance, "robustVariance": robust_variance, "texture": texture, "bbox": [x1, y1, x2, y2]}
