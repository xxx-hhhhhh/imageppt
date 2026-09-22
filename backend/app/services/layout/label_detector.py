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


def _extract_alpha_crop(image: np.ndarray, item: dict[str, Any], destination: Path) -> tuple[int, int, int, int] | None:
    x, y, w, h = [int(max(0, item[key])) for key in ("x", "y", "width", "height")]
    padding = max(2, int(round(min(w, h) * 0.04)))
    x, y = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(image.shape[1], x + w + padding * 2), min(image.shape[0], y + h + padding * 2)
    w, h = x2 - x, y2 - y
    crop = image[y:min(image.shape[0], y + h), x:min(image.shape[1], x + w)]
    if crop.size == 0:
        return None
    background = np.median(np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]], axis=0), axis=0)
    distance = np.linalg.norm(crop.astype(np.float32) - background.astype(np.float32), axis=2)
    alpha = np.uint8(np.clip((distance - 8) * 12, 0, 255))
    if int((alpha > 32).sum()) < max(20, crop.shape[0] * crop.shape[1] // 50):
        return None
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    destination.parent.mkdir(parents=True, exist_ok=True)
    return (x, y, w, h) if cv2.imwrite(str(destination), rgba) else None


def _complex_badge(image: np.ndarray, item: dict[str, Any]) -> bool:
    x, y, w, h = [int(max(0, item[key])) for key in ("x", "y", "width", "height")]
    crop = image[y:min(image.shape[0], y + h), x:min(image.shape[1], x + w)]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.ellipse(mask, (gray.shape[1] // 2, gray.shape[0] // 2), (max(1, gray.shape[1] // 2 - 2), max(1, gray.shape[0] // 2 - 2)), 0, 0, 360, 255, -1)
    edges = cv2.Canny(gray, 45, 135)
    edge_ratio = float((edges[mask > 0] > 0).mean()) if np.any(mask) else 0.0
    color_variance = float(np.mean(np.var(crop[mask > 0].astype(np.float32), axis=0))) if np.any(mask) else 0.0
    return edge_ratio >= 0.018 and color_variance >= 100


def _inside_shape(text: dict[str, Any], shape: dict[str, Any]) -> bool:
    tx, ty = _center(text)
    sx, sy = _center(shape)
    rx, ry = max(1.0, float(shape["width"]) / 2), max(1.0, float(shape["height"]) / 2)
    return ((tx - sx) / rx) ** 2 + ((ty - sy) / ry) ** 2 <= 0.82


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
        raw_text_box = (text.get("metadata") or {}).get("rawOCRBBox")
        if isinstance(raw_text_box, list) and len(raw_text_box) == 4:
            right_edge = text["x"] + text["width"]
            bottom_edge = text["y"] + text["height"]
            if horizontal:
                text["x"] = max(float(text["x"]), float(raw_text_box[0]) - 2.0, float(shape["x"] + shape["width"]) + 4.0)
                text["width"] = max(4.0, right_edge - text["x"])
            else:
                text["y"] = max(float(text["y"]), float(raw_text_box[1]) - 2.0, float(shape["y"] + shape["height"]) + 3.0)
                text["height"] = max(4.0, bottom_edge - text["y"])
        members = [shape, text]
        if asset_dir and project_id and _complex_badge(image, shape):
            icon_path = asset_dir / f"component_{group_id}.png"
            crop_bounds = _extract_alpha_crop(image, shape, icon_path)
            if crop_bounds:
                crop_x, crop_y, crop_w, crop_h = crop_bounds
                icon = {"id": f"component_asset_{next_group:04d}", "type": "image", "x": float(crop_x), "y": float(crop_y), "width": float(crop_w), "height": float(crop_h), "rotation": 0, "zIndex": 16, "src": f"/media/assets/{project_id}/{icon_path.name}", "style": {"opacity": 1}, "confidence": 0.82}
                _mark(icon, group_id, role, "wholeBadgeImage")
                icon.setdefault("metadata", {}).update({"wholeBadgeAsset": True, "preserveAsImage": True, "doNotVectorize": True, "reconstructionStrategy": "transparent_image", "sourceContentPreserved": True, "owns": [shape["id"]]})
                shape.setdefault("metadata", {}).update({"ownedBy": icon["id"], "suppressRender": True, "duplicateSuppressed": True, "reconstructionStrategy": "group", "sourceContentPreserved": False})
                for contained in texts:
                    if contained is not text and _inside_shape(contained, shape):
                        contained.setdefault("metadata", {}).update({"ownedBy": icon["id"], "suppressRender": True, "duplicateSuppressed": True, "reconstructionStrategy": "group", "sourceContentPreserved": True})
                additions.append(icon)
        group_box = (min(item["x"] for item in members), min(item["y"] for item in members), max(item["x"] + item["width"] for item in members), max(item["y"] + item["height"] for item in members))
        group = {"id": f"group_{group_id}", "type": "group", "x": group_box[0], "y": group_box[1], "width": group_box[2] - group_box[0], "height": group_box[3] - group_box[1], "rotation": 0, "zIndex": 5, "style": {"opacity": 1}, "confidence": 0.55}
        _mark(group, group_id, role, "group")
        additions.append(group)
    return base_elements + additions
