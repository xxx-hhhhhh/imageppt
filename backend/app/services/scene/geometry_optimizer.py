from __future__ import annotations

from typing import Any

from app.services.layout.position_refiner import refine_layout


def optimize_layout(elements: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    for item in elements:
        item.setdefault("metadata", {})["rawBBox"] = {key: item.get(key, 0) for key in ("x", "y", "width", "height")}
    refined = refine_layout(elements, width, height)
    for item in refined:
        item.setdefault("metadata", {})["refinedBBox"] = {key: item.get(key, 0) for key in ("x", "y", "width", "height")}
    return refined
