from __future__ import annotations

import math
from typing import Any


def _finite(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _center(item: dict[str, Any]) -> tuple[float, float]:
    return (item["x"] + item["width"] / 2, item["y"] + item["height"] / 2)


def refine_layout(elements: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    """Keep all layout coordinates in source-image pixels and apply gentle calibration.

    The detector contract is always left/top/width/height.  This function deliberately
    avoids a second normalization pass: it only removes invalid values and aligns
    children when a detector has already declared a semantic group.
    """
    by_id = {item.get("id"): item for item in elements}
    for item in elements:
        item["x"] = _finite(item.get("x"))
        item["y"] = _finite(item.get("y"))
        item["width"] = max(1.0, _finite(item.get("width"), 1.0))
        item["height"] = max(1.0, _finite(item.get("height"), 1.0))
        item["rotation"] = _finite(item.get("rotation"))
        item["zIndex"] = int(_finite(item.get("zIndex")))
        # Do not allow a bad detector bbox to shift the entire slide.
        if item.get("type") != "background":
            item["x"] = min(max(-2.0, item["x"]), max(0.0, width - item["width"] + 2.0))
            item["y"] = min(max(-2.0, item["y"]), max(0.0, height - item["height"] + 2.0))

    for item in elements:
        metadata = item.setdefault("metadata", {})
        group_id = metadata.get("groupId")
        if not group_id or item.get("type") != "text":
            continue
        group_members = [candidate for candidate in elements if candidate.get("metadata", {}).get("groupId") == group_id and candidate.get("id") != item.get("id")]
        circle = next((candidate for candidate in group_members if candidate.get("metadata", {}).get("componentType") == "iconCircle"), None)
        if circle:
            # Card titles sit under their icon; use the icon center only when the OCR
            # box is already within the group so unrelated text is never reflowed.
            cx, _ = _center(circle)
            if abs((item["x"] + item["width"] / 2) - cx) < max(circle["width"] * 1.7, 120):
                item["x"] = max(0.0, cx - item["width"] / 2)
                metadata["anchor"] = "center"
        if metadata.get("componentType") == "titleText":
            underline = next((candidate for candidate in group_members if candidate.get("metadata", {}).get("componentType") == "underline"), None)
            if underline:
                underline["x"] = item["x"] + item["width"] * 0.12
                underline["width"] = max(20.0, item["width"] * 0.28)
                underline["y"] = item["y"] + item["height"] + max(8.0, item["height"] * 0.18)

    return elements
