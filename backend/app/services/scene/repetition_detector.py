from __future__ import annotations

from collections import defaultdict
from typing import Any


def detect_repetition(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for element in elements:
        if element.get("type") in {"background", "text", "group"}:
            continue
        box = element.get("bbox") or {}
        width = float(box.get("width", element.get("width", 0)))
        height = float(box.get("height", element.get("height", 0)))
        aspect = int(round((width / max(1.0, height)) * 10))
        buckets[(element.get("type", "unknown"), int(round(width / 8)), aspect)].append(element["id"])
    return [{"type": "repeated_component", "members": ids, "confidence": 0.72} for ids in buckets.values() if len(ids) >= 2]
