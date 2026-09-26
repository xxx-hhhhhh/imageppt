from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .font_classifier import classify_font
from .font_estimator import estimate_style


def analyze_text_style(text: str, bbox: list[float], image_width: int, image_height: int, crop: np.ndarray | None = None, region_hint: str | None = None) -> dict[str, Any]:
    color = foreground_color(crop) if crop is not None and crop.size else "#111827"
    style = estimate_style(text, bbox, image_width, image_height, color, region_hint=region_hint, crop=crop)
    style["fontClass"] = classify_font(text, style.get("textRole", style.get("role", "unknown")), bbox, crop)
    return style


def foreground_color(crop: np.ndarray) -> str:
    pixels = crop.reshape(-1, 3).astype(np.float32)
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]], axis=0).astype(np.float32)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(pixels - background, axis=1)
    # Antialiased glyph edges are far more numerous than the opaque strokes.
    # Use the strongest foreground pixels so editable text does not inherit a
    # washed-out gray from dark source lettering.
    selected = pixels[distance >= max(12.0, float(np.percentile(distance, 90)))]
    if len(selected) < 3:
        selected = pixels
    bgr = np.median(selected, axis=0).astype(int).tolist()
    r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
    return f"#{r:02X}{g:02X}{b:02X}"
