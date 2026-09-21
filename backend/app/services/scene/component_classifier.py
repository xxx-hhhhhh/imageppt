from __future__ import annotations

from typing import Any


def classify_element(element: dict[str, Any]) -> tuple[str, str]:
    kind = element.get("type", "decoration")
    metadata = element.get("metadata") or {}
    role = element.get("role") or metadata.get("role")
    if kind == "text":
        return "text", role or (element.get("style") or {}).get("textRole") or (element.get("style") or {}).get("role", "body")
    if kind == "image":
        return ("photo" if role in {"photo", "image_region"} else "image"), role or "visual_asset"
    if kind == "background":
        return "background", "background"
    if kind == "group":
        return "group", role or "component"
    if kind in {"ellipse", "circle"}:
        return "ellipse", role or "shape"
    if kind in {"rectangle", "roundedRectangle"}:
        return kind, role or "container"
    if kind in {"line", "arrow", "connector"}:
        return kind, role or "connector"
    return kind, role or "decoration"
