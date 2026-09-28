"""Build an editable white slide from source pixels and existing scene owners."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def objectize_on_white(source_path: Path, background_path: Path, layout: dict, asset_dir: Path, project_id: str, page_index: int) -> dict[str, int]:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(source_path)
    height, width = source.shape[:2]
    asset_dir.mkdir(parents=True, exist_ok=True)
    occupied = np.zeros((height, width), np.uint8)
    elements = layout.setdefault("elements", [])
    active = [item for item in elements if item.get("type") != "background" and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    for item in active:
        box = _box(item, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        # Text is owned by its textbox. Its source glyphs must never enter a cutout.
        if item.get("type") == "text":
            raw = (item.get("metadata") or {}).get("rawOCRBBox")
            if isinstance(raw, list) and len(raw) == 4:
                x1, y1, x2, y2 = _clip(raw, width, height)
            pad = max(3, round((y2 - y1) * 0.15))
            x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)
        occupied[y1:y2, x1:x2] = 255

    # Inspect only pixels without a current owner. The full-page source is never
    # used as an output layer; bounded residual components become movable assets.
    distance = np.max(255 - source.astype(np.int16), axis=2)
    foreground = np.uint8((distance > 12) & (occupied == 0)) * 255
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < max(80, width * height * 0.00008):
            continue
        if w < 5 or h < 5:
            continue
        candidates.append((x, y, w, h, contour))
    candidates.sort(key=lambda entry: entry[2] * entry[3], reverse=True)
    added = 0
    native_shapes = 0
    for x, y, w, h, contour in candidates[:100]:
        pad = 2
        x1, y1, x2, y2 = max(0, x - pad), max(0, y - pad), min(width, x + w + pad), min(height, y + h + pad)
        region = foreground[y1:y2, x1:x2]
        if np.count_nonzero(region) < 30:
            continue
        crop = source[y1:y2, x1:x2]
        if w * h > width * height * 0.65:
            # A large uniform panel is still an object, represented natively.
            visible = crop[region > 0]
            if len(visible) and float(np.mean(np.std(visible.astype(np.float32), axis=0))) < 9:
                color = np.median(visible, axis=0).astype(np.uint8)
                native_shapes += 1
                elements.append({"id": f"white_panel_page_{page_index}_{native_shapes:03d}", "type": "rectangle", "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "rotation": 0, "zIndex": 1, "style": {"fill": f"#{color[2]:02X}{color[1]:02X}{color[0]:02X}", "strokeWidth": 0}, "metadata": {"reconstructionStrategy": "native_shape", "reconstructionStrategySource": "white_objectization"}})
            continue
        alpha = cv2.dilate(region, np.ones((3, 3), np.uint8), iterations=1)
        alpha[occupied[y1:y2, x1:x2] != 0] = 0
        rgba = np.dstack((crop, alpha))
        added += 1
        asset_id = f"white_object_page_{page_index}_{added:03d}"
        path = asset_dir / f"{asset_id}.png"
        cv2.imwrite(str(path), rgba)
        elements.append({"id": asset_id, "type": "image", "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "rotation": 0, "zIndex": 1, "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1}, "metadata": {"reconstructionStrategy": "cutout_image", "reconstructionStrategySource": "white_objectization", "preserveWholeAsset": True, "doNotVectorize": True}})
    background_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(background_path), np.full_like(source, 255))
    layout["backgroundUrl"] = f"/media/backgrounds/{project_id}/{background_path.name}"
    layout.setdefault("metadata", {})["reconstructionSurfaceMode"] = "white_objectized"
    return {"whiteObjectAssets": added, "whiteObjectShapes": native_shapes, "whiteBackgroundPixels": width * height}


def _clip(values: list, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [round(float(value)) for value in values]
    return max(0, min(width, x1)), max(0, min(height, y1)), max(0, min(width, x2)), max(0, min(height, y2))


def _box(item: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    x = float(item.get("x") or 0)
    y = float(item.get("y") or 0)
    box = _clip([x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)], width, height)
    return box if box[2] > box[0] and box[3] > box[1] else None
