from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _hex(color: np.ndarray) -> str:
    b, g, r = [int(max(0, min(255, value))) for value in color.tolist()]
    return f"#{r:02X}{g:02X}{b:02X}"


def analyze_colors(image_path: Path, max_samples: int = 12000) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {"primary": "#FFFFFF", "secondary": "#D9E2EC", "accent": "#2563A6", "textPrimary": "#111827", "textSecondary": "#64748B", "background": "#FFFFFF"}
    pixels = image.reshape(-1, 3)
    step = max(1, len(pixels) // max_samples)
    samples = np.float32(pixels[::step])
    clusters = min(6, max(2, len(samples) // 1000))
    _, labels, centers = cv2.kmeans(samples, clusters, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2), 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=clusters)
    ordered = [centers[index] for index in np.argsort(counts)[::-1]]
    luminance = [float(cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_BGR2GRAY)[0, 0]) for center in ordered]
    background = ordered[0]
    accent = max(ordered, key=lambda color: float(np.ptp(color)))
    dark = min(ordered, key=lambda color: float(np.mean(color)))
    return {"primary": _hex(ordered[0]), "secondary": _hex(ordered[1] if len(ordered) > 1 else ordered[0]), "accent": _hex(accent), "textPrimary": _hex(dark), "textSecondary": _hex(ordered[-1]), "background": _hex(background), "clusters": [_hex(color) for color in ordered], "luminance": luminance}
