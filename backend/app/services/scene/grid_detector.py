from __future__ import annotations

from typing import Any


def detect_grid(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible = [item for item in elements if item.get("type") not in {"background", "group"}]
    if len(visible) < 2:
        return []
    return [{"type": "grid_candidate", "members": [item["id"] for item in visible], "confidence": 0.35}]
