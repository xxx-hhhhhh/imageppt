from __future__ import annotations

from typing import Any

from app.services.typography.font_estimator import estimate_style
from app.services.typography.paragraph_analyzer import group_lines_into_paragraphs
from app.services.typography.text_style_analyzer import enrich_text_style


def analyze_text_regions(regions: list[Any], image_width: int, image_height: int) -> dict[str, list[dict[str, Any]]]:
    lines: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        style = region.style or estimate_style(region.text, region.bbox, image_width, image_height, "#111827")
        lines.append({"id": f"line_{index + 1:04d}", "text": region.text, "bbox": list(region.bbox), "baseline": region.bbox[3], "height": region.bbox[3] - region.bbox[1], "confidence": region.confidence, "style": style})
    return {"lines": lines, "paragraphs": group_lines_into_paragraphs(lines)}
