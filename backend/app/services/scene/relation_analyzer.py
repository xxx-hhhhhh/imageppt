from __future__ import annotations

from typing import Any


def _bbox(item: dict[str, Any]) -> tuple[float, float, float, float]:
    box = item.get("bbox") or {}
    if box:
        return float(box["left"]), float(box["top"]), float(box["left"] + box["width"]), float(box["top"] + box["height"])
    return float(item.get("x", 0)), float(item.get("y", 0)), float(item.get("x", 0) + item.get("width", 0)), float(item.get("y", 0) + item.get("height", 0))


def analyze_relations(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for index, left in enumerate(elements):
        lx1, ly1, lx2, ly2 = _bbox(left)
        lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        for right in elements[index + 1:]:
            rx1, ry1, rx2, ry2 = _bbox(right)
            rcx, rcy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
            if lx1 <= rx1 and ly1 <= ry1 and lx2 >= rx2 and ly2 >= ry2:
                relations.append({"type": "contains", "source": left["id"], "target": right["id"], "confidence": 0.78})
            elif rx1 <= lx1 and ry1 <= ly1 and rx2 >= lx2 and ry2 >= ly2:
                relations.append({"type": "contains", "source": right["id"], "target": left["id"], "confidence": 0.78})
            else:
                gap_x = max(0.0, max(lx1, rx1) - min(lx2, rx2))
                gap_y = max(0.0, max(ly1, ry1) - min(ly2, ry2))
                scale = max(lx2 - lx1, ly2 - ly1, rx2 - rx1, ry2 - ry1, 1.0)
                if max(gap_x, gap_y) <= scale * 1.4:
                    relations.append({"type": "near", "source": left["id"], "target": right["id"], "distance": round(max(gap_x, gap_y), 2), "confidence": 0.55})
            if abs(lcy - rcy) <= max(2.0, min(ly2 - ly1, ry2 - ry1) * 0.18):
                relations.append({"type": "center_y_aligned", "members": [left["id"], right["id"]], "confidence": 0.62})
            if abs(lcx - rcx) <= max(2.0, min(lx2 - lx1, rx2 - rx1) * 0.18):
                relations.append({"type": "center_x_aligned", "members": [left["id"], right["id"]], "confidence": 0.62})
    return relations
