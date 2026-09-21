"""Generic relationship-based component grouping.

The function name is kept for compatibility with the first layout service, but
the implementation no longer assumes a theme, color, page ratio, or component
count. It groups visual objects from relative geometry and containment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _bbox(item: dict[str, Any]) -> tuple[float, float, float, float]:
    return float(item["x"]), float(item["y"]), float(item["x"] + item["width"]), float(item["y"] + item["height"])


def _mark(item: dict[str, Any], group_id: str, role: str, component_type: str) -> None:
    item["groupId"] = group_id
    item["role"] = role
    item["componentType"] = component_type
    item.setdefault("metadata", {}).update({"groupId": group_id, "role": role, "componentType": component_type})


def _center(item: dict[str, Any]) -> tuple[float, float]:
    return item["x"] + item["width"] / 2, item["y"] + item["height"] / 2


def _near(a: dict[str, Any], b: dict[str, Any], factor: float = 1.6) -> bool:
    ax, ay = _center(a)
    bx, by = _center(b)
    scale = max(8.0, min(a["width"], a["height"], b["width"], b["height"]))
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= scale * factor + max(a["width"], b["width"]) * 0.75


def _best_text(shape: dict[str, Any], texts: list[dict[str, Any]]) -> dict[str, Any] | None:
    horizontal = [text for text in texts if text["x"] >= shape["x"] + shape["width"] * 0.55 and abs(_center(text)[1] - _center(shape)[1]) <= max(shape["height"], text["height"]) * 1.1]
    if horizontal:
        return min(horizontal, key=lambda item: abs(_center(item)[1] - _center(shape)[1]) + max(0.0, item["x"] - shape["x"]) * 0.02)
    vertical = [text for text in texts if text["y"] >= shape["y"] + shape["height"] * 0.55 and abs(_center(text)[0] - _center(shape)[0]) <= max(shape["width"], text["width"]) * 0.9]
    if vertical:
        return min(vertical, key=lambda item: abs(item["y"] - (shape["y"] + shape["height"])))
    return None


def _sample_fill(image: np.ndarray, bounds: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = [int(max(0, value)) for value in bounds]
    crop = image[min(y1, image.shape[0] - 1):max(y1 + 1, min(y2, image.shape[0])), min(x1, image.shape[1] - 1):max(x1 + 1, min(x2, image.shape[1]))]
    if crop.size == 0:
        return "#DCE6F1"
    b, g, r = crop.reshape(-1, 3).mean(axis=0).astype(int).tolist()
    return f"#{r:02X}{g:02X}{b:02X}"


def _extract_alpha_crop(image: np.ndarray, item: dict[str, Any], destination: Path) -> bool:
    x, y, w, h = [int(max(0, item[key])) for key in ("x", "y", "width", "height")]
    crop = image[y:min(image.shape[0], y + h), x:min(image.shape[1], x + w)]
    if crop.size == 0:
        return False
    background = np.median(np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]], axis=0), axis=0)
    distance = np.linalg.norm(crop.astype(np.float32) - background.astype(np.float32), axis=2)
    alpha = np.uint8(np.clip((distance - 8) * 12, 0, 255))
    if int((alpha > 32).sum()) < max(20, crop.shape[0] * crop.shape[1] // 50):
        return False
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    destination.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(destination), rgba))


def detect_label_groups(image_path: Path, image_width: int, image_height: int, ocr_results: list[Any], elements: list[dict[str, Any]], asset_dir: Path | None, project_id: str | None) -> list[dict[str, Any]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return elements
    detected_shapes = []
    if not any(item.get("type") == "ellipse" for item in elements):
        from app.services.layout.detectors import detect_simple_shapes
        detected_shapes = detect_simple_shapes(image_path, [])
    base_elements = [*elements, *detected_shapes]
    shapes = [item for item in base_elements if item.get("type") in {"ellipse", "rectangle", "roundedRectangle", "image"}]
    texts = [item for item in base_elements if item.get("type") == "text"]
    additions: list[dict[str, Any]] = []
    next_group = 1

    # First, use explicit containment: a container and its text/icon children
    # are a component even when the page has no theme-specific colors.
    for container in [item for item in shapes if item.get("type") in {"rectangle", "roundedRectangle"}]:
        x1, y1, x2, y2 = _bbox(container)
        children = [item for item in elements if item is not container and x1 <= item["x"] and y1 <= item["y"] and item["x"] + item["width"] <= x2 and item["y"] + item["height"] <= y2 and item.get("type") in {"text", "ellipse", "image"}]
        if children:
            group_id = f"component_{next_group:04d}"
            next_group += 1
            _mark(container, group_id, "component", "container")
            for child in children:
                _mark(child, group_id, "component", "titleText" if child.get("type") == "text" else "icon" if child.get("type") in {"ellipse", "image"} else "child")
            _mark(container, group_id, "component", "container")

    for shape in [item for item in shapes if item.get("type") == "ellipse" and not item.get("groupId")]:
        candidates = [text for text in texts if not text.get("groupId") and _near(shape, text)]
        text = _best_text(shape, candidates)
        if text is None:
            continue
        group_id = f"component_{next_group:04d}"
        next_group += 1
        horizontal = text["x"] > shape["x"] + shape["width"] * 0.65
        role = "iconWithText" if horizontal else "badge"
        _mark(shape, group_id, role, "iconCircle")
        _mark(text, group_id, role, "titleText")
        members = [shape, text]
        if horizontal:
            left = min(shape["x"], text["x"]) - min(shape["width"], shape["height"]) * 0.35
            top = min(shape["y"], text["y"]) - min(shape["width"], shape["height"]) * 0.28
            right = max(shape["x"] + shape["width"], text["x"] + text["width"]) + min(shape["width"], shape["height"]) * 0.35
            bottom = max(shape["y"] + shape["height"], text["y"] + text["height"]) + min(shape["width"], shape["height"]) * 0.28
            container = {"id": f"component_container_{next_group:04d}", "type": "roundedRectangle", "x": float(max(0, left)), "y": float(max(0, top)), "width": float(min(image_width, right) - max(0, left)), "height": float(min(image_height, bottom) - max(0, top)), "rotation": 0, "zIndex": 6, "style": {"fill": _sample_fill(image, (left, top, right, bottom)), "stroke": _sample_fill(image, (left, top, right, bottom)), "strokeWidth": 1, "opacity": 0.2}, "confidence": 0.52}
            _mark(container, group_id, role, "container")
            additions.append(container)
            members.append(container)
        if asset_dir and project_id:
            icon_path = asset_dir / f"component_{group_id}.png"
            if _extract_alpha_crop(image, shape, icon_path):
                icon = {"id": f"component_asset_{next_group:04d}", "type": "image", "x": shape["x"], "y": shape["y"], "width": shape["width"], "height": shape["height"], "rotation": 0, "zIndex": 16, "src": f"/media/assets/{project_id}/{icon_path.name}", "style": {"opacity": 1}, "confidence": 0.5}
                _mark(icon, group_id, role, "optionalIconImage")
                additions.append(icon)
        group_box = (min(item["x"] for item in members), min(item["y"] for item in members), max(item["x"] + item["width"] for item in members), max(item["y"] + item["height"] for item in members))
        group = {"id": f"group_{group_id}", "type": "group", "x": group_box[0], "y": group_box[1], "width": group_box[2] - group_box[0], "height": group_box[3] - group_box[1], "rotation": 0, "zIndex": 5, "style": {"opacity": 1}, "confidence": 0.55}
        _mark(group, group_id, role, "group")
        additions.append(group)
    return base_elements + additions
