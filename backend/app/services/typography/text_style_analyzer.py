from __future__ import annotations

from typing import Any


def estimate_alignment(bbox: list[float], container: list[float] | None = None) -> str:
    if not container:
        return "left"
    left, _, right, _ = bbox
    c_left, _, c_right, _ = container
    center_delta = abs((left + right) / 2 - (c_left + c_right) / 2)
    if center_delta <= max(2.0, (c_right - c_left) * 0.04):
        return "center"
    return "right" if abs(c_right - right) < abs(left - c_left) else "left"


def enrich_text_style(style: dict[str, Any], bbox: list[float], container: list[float] | None = None) -> dict[str, Any]:
    return {**style, "align": style.get("align") or estimate_alignment(bbox, container), "bbox": bbox, "verticalAlign": style.get("verticalAlign", "middle")}
