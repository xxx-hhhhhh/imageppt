"""Separate page environment from bounded visual surfaces inside modules."""

from __future__ import annotations

from typing import Any


def is_page_environment(module: dict[str, Any], box: tuple[int, int, int, int], width: int, height: int) -> bool:
    """A local colored panel remains an object even if a planner calls it background.

    This decision uses page-relative geometry and semantic role, never a
    particular color or test-image coordinate.
    """
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1) / max(1, width * height)
    span_x = (x2 - x1) / max(1, width)
    span_y = (y2 - y1) / max(1, height)
    role = " ".join(str(module.get(key) or "") for key in ("role", "semanticType", "name", "componentType")).lower()
    local_roles = ("card", "panel", "module", "title", "header", "label", "badge", "icon", "frame", "border", "bar", "strip", "container", "chart", "photo", "illustration")
    if any(word in role for word in local_roles) and area < 0.85:
        return False
    edge_count = sum((x1 <= width * 0.02, y1 <= height * 0.02, x2 >= width * 0.98, y2 >= height * 0.98))
    return area >= 0.75 or (span_x >= 0.88 and span_y >= 0.70) or (edge_count >= 3 and area >= 0.45)
