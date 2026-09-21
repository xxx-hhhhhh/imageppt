from __future__ import annotations

from typing import Any


def apply_safe_adjustments(scene: dict[str, Any], critic: dict[str, Any]) -> dict[str, Any]:
    width = float(scene.get("canvas", {}).get("width", 1))
    height = float(scene.get("canvas", {}).get("height", 1))
    by_id = {item["id"]: item for item in scene.get("elements", [])}
    applied, rejected = [], []
    for issue in critic.get("issues", []) if isinstance(critic, dict) else []:
        item = by_id.get(issue.get("elementId"))
        adjustment = issue.get("adjustment") or {}
        if not item or not isinstance(adjustment, dict):
            rejected.append(issue)
            continue
        box = item.setdefault("bbox", {})
        if "moveX" in adjustment:
            move = _bounded(adjustment["moveX"], width * 0.1)
            box["left"] = float(box.get("left", 0)) + move
        if "moveY" in adjustment:
            move = _bounded(adjustment["moveY"], height * 0.1)
            box["top"] = float(box.get("top", 0)) + move
        for key, lower, upper in (("widthScale", 0.8, 1.2), ("heightScale", 0.8, 1.2)):
            if key in adjustment:
                factor = _clamp(adjustment[key], lower, upper)
                dimension = "width" if key == "widthScale" else "height"
                box[dimension] = float(box.get(dimension, 1)) * factor
        if "fontSizeScale" in adjustment:
            factor = _clamp(adjustment["fontSizeScale"], 0.7, 1.3)
            item.setdefault("style", {})["fontSize"] = float(item.get("style", {}).get("fontSize", 24)) * factor
        applied.append(issue.get("elementId"))
    scene["criticAdjustments"] = {"applied": applied, "rejected": rejected}
    return scene


def _bounded(value: Any, limit: float) -> float:
    try:
        return max(-limit, min(limit, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: Any, lower: float, upper: float) -> float:
    try:
        return max(lower, min(upper, float(value)))
    except (TypeError, ValueError):
        return 1.0
