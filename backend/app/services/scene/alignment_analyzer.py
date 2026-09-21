from __future__ import annotations

from collections import defaultdict
from typing import Any


def analyze_alignment(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[int, list[str]] = defaultdict(list)
    cols: dict[int, list[str]] = defaultdict(list)
    for element in elements:
        box = element.get("bbox") or {}
        left = float(box.get("left", element.get("x", 0)))
        top = float(box.get("top", element.get("y", 0)))
        rows[int(round(top / 12))].append(element["id"])
        cols[int(round(left / 12))].append(element["id"])
    relations: list[dict[str, Any]] = []
    relations.extend({"type": "row_alignment", "members": ids, "confidence": 0.5} for ids in rows.values() if len(ids) >= 2)
    relations.extend({"type": "column_alignment", "members": ids, "confidence": 0.5} for ids in cols.values() if len(ids) >= 2)
    return relations
