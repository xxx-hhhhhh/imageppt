"""Backward-compatible typography imports."""

from app.services.typography.font_estimator import estimate_style


def estimate_text_style(text: str, bbox: list[float], image_width: int, image_height: int, color: str):
    return estimate_style(text, bbox, image_width, image_height, color)
