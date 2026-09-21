from __future__ import annotations

from typing import Any


def infer_layers(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {"background": 0, "texture": 2, "container": 8, "decoration": 12, "ellipse": 14, "image": 16, "photo": 16, "text": 20}
    for element in elements:
        role = element.get("role") or ""
        kind = element.get("type", "decoration")
        if kind == "group":
            element["zIndex"] = min(element.get("zIndex", 5), 5)
        else:
            element["zIndex"] = max(int(element.get("zIndex", 0)), priority.get(kind, 10))
            if "title" in role or role in {"main_title", "body", "label"}:
                element["zIndex"] = max(element["zIndex"], 20)
    return elements
