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
        metadata = item.setdefault("metadata", {})
        if item.get("type") in {"rectangle", "roundedRectangle", "ellipse", "line", "arrow"} and not item.get("src") and (item.get("style") or {}).get("fill") and metadata.get("reconstructionStrategy") in {None, "local_image", "background_image"}:
            metadata.update({"reconstructionStrategy": "native_shape", "reconstructionStrategySource": "white_objectization"})
    bordered_shapes, bordered_assets = _extract_bordered_containers(source, active, elements, occupied, asset_dir, project_id, page_index)
    container_count = _extract_flat_containers(source, active, elements, occupied, page_index)
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
    return {"whiteObjectAssets": added + bordered_assets, "whiteObjectShapes": native_shapes + container_count + bordered_shapes, "whiteContainerShapes": container_count + bordered_shapes + bordered_assets, "whiteBackgroundPixels": width * height}


def _extract_bordered_containers(source: np.ndarray, active: list[dict], elements: list[dict], occupied: np.ndarray, asset_dir: Path, project_id: str, page_index: int) -> tuple[int, int]:
    height, width = source.shape[:2]
    edges = cv2.Canny(source, 45, 130)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 10 or w * h < max(90, width * height * 0.0001) or w * h > width * height * 0.80:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if not 4 <= len(polygon) <= 12 or cv2.contourArea(contour) / max(1, w * h) < 0.68:
            continue
        if np.mean(occupied[y:y + h, x:x + w] > 0) > 0.35:
            continue
        border = edges[y:y + h, x:x + w]
        band = max(2, min(w, h) // 18)
        if min(np.mean(border[:band] > 0), np.mean(border[-band:] > 0), np.mean(border[:, :band] > 0), np.mean(border[:, -band:] > 0)) < 0.045:
            continue
        candidates.append((x, y, w, h, contour, len(polygon) > 4))
    candidates.sort(key=lambda value: value[2] * value[3], reverse=True)
    shapes = assets = 0
    selected: list[tuple[int, int, int, int]] = []
    for x, y, w, h, contour, rounded in candidates[:80]:
        box = (x, y, x + w, y + h)
        if any((w * h) <= 1.6 * float(item.get("width") or 0) * float(item.get("height") or 0) and _overlap_min(box, _box(item, width, height)) > 0.65 for item in active if item.get("type") == "text" and _box(item, width, height)):
            continue
        if any(_overlap_min(box, prior) > 0.75 for prior in selected):
            continue
        if any(_overlap_min(box, _box(item, width, height)) > 0.80 for item in active if item.get("type") != "text" and _box(item, width, height)):
            continue
        texts = [item for item in active if item.get("type") == "text" and _text_inside(item, box)]
        crop = source[y:y + h, x:x + w]
        inset = max(2, min(w, h) // 10)
        inner = crop[inset:h - inset, inset:w - inset]
        if inner.size == 0:
            continue
        # Border-only cards are native when their interior is flat. Text and
        # textured interiors use a text-cleaned movable crop instead.
        text_mask = np.zeros((h, w), np.uint8)
        for item in texts:
            text_box = _box(item, width, height)
            if text_box:
                tx1, ty1, tx2, ty2 = text_box
                cv2.rectangle(text_mask, (max(0, tx1 - x - 2), max(0, ty1 - y - 2)), (min(w - 1, tx2 - x + 2), min(h - 1, ty2 - y + 2)), 255, -1)
        clean_pixels = inner[text_mask[inset:h - inset, inset:w - inset] == 0]
        if len(clean_pixels) < 12:
            continue
        flat = float(np.max(np.std(clean_pixels.astype(np.float32), axis=0))) < 12
        z_index = min([int(item.get("zIndex") or 20) for item in texts], default=10) - 1
        if flat:
            fill_color = np.median(clean_pixels, axis=0).astype(np.uint8)
            border_color = np.median(crop[edges[y:y + h, x:x + w] > 0], axis=0).astype(np.uint8)
            shapes += 1
            elements.append({"id": f"white_border_page_{page_index}_{shapes:03d}", "type": "roundedRectangle" if rounded else "rectangle", "x": x, "y": y, "width": w, "height": h, "rotation": 0, "zIndex": z_index, "style": {"fill": _hex_bgr(fill_color), "stroke": _hex_bgr(border_color), "strokeWidth": 1, "opacity": 1}, "metadata": {"reconstructionStrategy": "native_shape", "reconstructionStrategySource": "white_objectization", "layerRole": "container"}})
        else:
            assets += 1
            clean = cv2.inpaint(crop, text_mask, 3, cv2.INPAINT_TELEA) if np.any(text_mask) else crop
            alpha = np.zeros((h, w), np.uint8)
            local_contour = contour - np.array([[[x, y]]])
            cv2.drawContours(alpha, [local_contour], -1, 255, -1)
            path = asset_dir / f"white_border_page_{page_index}_{assets:03d}.png"
            cv2.imwrite(str(path), np.dstack((clean, alpha)))
            elements.append({"id": path.stem, "type": "image", "x": x, "y": y, "width": w, "height": h, "rotation": 0, "zIndex": z_index, "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1}, "metadata": {"reconstructionStrategy": "cutout_image", "reconstructionStrategySource": "white_objectization", "layerRole": "container", "textCleaned": bool(texts)}})
        occupied[y:y + h, x:x + w] = 255
        selected.append(box)
    return shapes, assets


def _hex_bgr(color: np.ndarray) -> str:
    return f"#{color[2]:02X}{color[1]:02X}{color[0]:02X}"


def _extract_flat_containers(source: np.ndarray, active: list[dict], elements: list[dict], occupied: np.ndarray, page_index: int) -> int:
    """Recover bounded flat panels before text masks split their source pixels."""
    height, width = source.shape[:2]
    quantized = ((source.astype(np.uint16) + 8) // 16).clip(0, 15).astype(np.uint8)
    packed = (quantized[:, :, 0].astype(np.uint16) << 8) | (quantized[:, :, 1].astype(np.uint16) << 4) | quantized[:, :, 2].astype(np.uint16)
    counts = np.bincount(packed.ravel(), minlength=4096)
    order = np.argsort(counts)[::-1]
    source_int = source.astype(np.int16)
    candidates: list[tuple[int, int, int, int, str, str]] = []
    minimum_area = max(70, round(width * height * 0.00008))
    for index in order[:40]:
        color = np.array([(index >> 8) & 15, (index >> 4) & 15, index & 15], dtype=np.int16) * 16
        if np.max(255 - color) < 9 or counts[index] < minimum_area:
            continue
        delta = np.max(np.abs(source_int - color), axis=2)
        mask = np.uint8(delta <= 13) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if w < 10 or h < 8 or area < minimum_area or area > width * height * 0.85:
                continue
            fill_ratio = cv2.contourArea(contour) / area
            if fill_ratio < 0.70:
                continue
            region = source[y:y + h, x:x + w]
            matching = delta[y:y + h, x:x + w] <= 13
            if np.mean(matching) < 0.65:
                continue
            median = np.median(region[matching], axis=0).astype(np.uint8)
            fill = f"#{median[2]:02X}{median[1]:02X}{median[0]:02X}"
            radius = _rounded_corner_hint(mask[y:y + h, x:x + w])
            candidates.append((x, y, w, h, fill, "roundedRectangle" if radius else "rectangle"))
    # Prefer larger panels; nested differently colored blocks remain separate.
    candidates.sort(key=lambda item: item[2] * item[3], reverse=True)
    created = 0
    created_boxes: list[tuple[int, int, int, int]] = []
    for x, y, w, h, fill, kind in candidates:
        box = (x, y, x + w, y + h)
        if np.mean(occupied[y:y + h, x:x + w] > 0) > 0.35:
            continue
        if any((w * h) <= 1.6 * float(item.get("width") or 0) * float(item.get("height") or 0) and _overlap_min(box, _box(item, width, height)) > 0.65 for item in active if item.get("type") == "text" and _box(item, width, height)):
            continue
        if any(_overlap_min(box, previous) > 0.78 for previous in created_boxes):
            continue
        if any(_overlap_min(box, _box(item, width, height)) > 0.80 for item in active if item.get("type") not in {"text", "background"} and _box(item, width, height)):
            continue
        if any(_overlap_min(box, _box(item, width, height)) > 0.80 and item.get("style", {}).get("fill") == fill for item in elements if item.get("metadata", {}).get("reconstructionStrategySource") == "white_objectization" and _box(item, width, height)):
            continue
        created += 1
        identifier = f"white_container_page_{page_index}_{created:03d}"
        contained_text = [item for item in active if item.get("type") == "text" and _text_inside(item, box)]
        elements.append({"id": identifier, "type": kind, "x": x, "y": y, "width": w, "height": h, "rotation": 0, "zIndex": min([int(item.get("zIndex") or 20) for item in contained_text], default=10) - 1, "style": {"fill": fill, "stroke": fill, "strokeWidth": 0, "opacity": 1}, "metadata": {"reconstructionStrategy": "native_shape", "reconstructionStrategySource": "white_objectization", "layerRole": "container"}})
        occupied[y:y + h, x:x + w] = 255
        created_boxes.append(box)
    return created


def _overlap_min(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    overlap = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(0, min(left[3], right[3]) - max(left[1], right[1]))
    return overlap / max(1, min((left[2] - left[0]) * (left[3] - left[1]), (right[2] - right[0]) * (right[3] - right[1])))


def _text_inside(item: dict, box: tuple[int, int, int, int]) -> bool:
    x = float(item.get("x") or 0) + float(item.get("width") or 0) / 2
    y = float(item.get("y") or 0) + float(item.get("height") or 0) / 2
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _rounded_corner_hint(mask: np.ndarray) -> bool:
    h, w = mask.shape
    size = max(2, min(w, h) // 8)
    return h >= 20 and w >= 20 and np.mean(mask[:size, :size]) < 180 and np.mean(mask[size:2 * size, size:2 * size]) > 180


def _clip(values: list, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [round(float(value)) for value in values]
    return max(0, min(width, x1)), max(0, min(height, y1)), max(0, min(width, x2)), max(0, min(height, y2))


def _box(item: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    x = float(item.get("x") or 0)
    y = float(item.get("y") or 0)
    box = _clip([x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)], width, height)
    return box if box[2] > box[0] and box[3] > box[1] else None
