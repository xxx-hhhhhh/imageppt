"""Conservative post-objectization checks for complete movable modules."""

from __future__ import annotations
from app.utils import image_io

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.pptx.renderer import _path_from_src


def repair_objectized_modules(source_path: Path, layout: dict[str, Any], asset_dir: Path, project_id: str, target_ids: set[str] | None = None) -> dict[str, Any]:
    source = image_io.imread(str(source_path), cv2.IMREAD_COLOR)
    report: dict[str, Any] = {"checkedModules": 0, "missingBackplates": 0, "sharedBackplateCandidates": 0, "recoveredBackplates": 0, "reboundBackplates": 0, "moduleImageFallbacks": 0, "squareCutouts": 0, "repairedCutouts": 0, "roundCutoutsChecked": 0, "roundCutoutIssues": 0, "issues": []}
    if source is None:
        report["issues"].append({"problem": "sourceUnavailable"})
        return report
    elements = layout.get("elements") or []
    active = [item for item in elements if item.get("type") not in {"background", "group"} and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    groups: dict[str, list[dict]] = {}
    for item in active:
        group_id = item.get("groupId") or (item.get("metadata") or {}).get("groupId")
        if group_id:
            groups.setdefault(str(group_id), []).append(item)
    for group_id, members in groups.items():
        if target_ids is not None and not any(str(item.get("id")) in target_ids for item in members):
            continue
        foreground = [item for item in members if item.get("type") in {"text", "image", "ellipse", "line", "arrow"} and (item.get("metadata") or {}).get("layerRole") != "container"]
        if not foreground:
            continue
        report["checkedModules"] += 1
        if any((item.get("metadata") or {}).get("layerRole") == "container" or item.get("type") in {"rectangle", "roundedRectangle"} for item in members):
            continue
        candidate = _local_plate(source, foreground)
        if candidate is None:
            continue
        x, y, w, h, fill, flat, contour = candidate
        box = (x, y, x + w, y + h)
        if w * h > source.shape[0] * source.shape[1] * 0.10:
            foreign_groups = {
                str(item.get("groupId") or (item.get("metadata") or {}).get("groupId"))
                for item in active if item.get("type") in {"text", "image"}
                and (item.get("groupId") or (item.get("metadata") or {}).get("groupId"))
                and str(item.get("groupId") or (item.get("metadata") or {}).get("groupId")) != group_id
                and x <= float(item.get("x") or 0) + float(item.get("width") or 0) / 2 <= x + w
                and y <= float(item.get("y") or 0) + float(item.get("height") or 0) / 2 <= y + h
            }
            if len(foreign_groups) >= 2:
                # This is a parent column/card shared by several groups, not a
                # missing plate belonging to each child. Pixel QA still detects
                # the parent surface if it is actually absent from the preview.
                report["sharedBackplateCandidates"] += 1
                continue
        report["missingBackplates"] += 1
        plate = next((item for item in active if (item.get("metadata") or {}).get("layerRole") == "residual" and _covered_fraction(item, box) >= 0.8 and _box_area(item) >= w * h * 0.5), None)
        if plate is not None:
            plate["groupId"] = group_id
            plate["zIndex"] = min(int(item.get("zIndex") or 20) for item in foreground) - 1
            plate.setdefault("metadata", {}).update({"groupId": group_id, "layerRole": "container", "moduleMemberIds": [item["id"] for item in foreground if item.get("id")], "objectizationQARecovered": True})
            report["reboundBackplates"] += 1
            continue
        identifier = f"qa_backplate_{len(elements) + 1:04d}"
        z_index = min(int(item.get("zIndex") or 20) for item in foreground) - 1
        metadata = {"reconstructionStrategySource": "objectization_qa", "layerRole": "container", "groupId": group_id, "objectizationQARecovered": True}
        if flat:
            metadata.update({"reconstructionStrategy": "native_shape", "moduleMemberIds": [item["id"] for item in foreground if item.get("id")]})
            elements.append({"id": identifier, "type": "rectangle", "x": x, "y": y, "width": w, "height": h, "rotation": 0, "zIndex": z_index, "groupId": group_id, "style": {"fill": fill, "stroke": fill, "strokeWidth": 0, "opacity": 1}, "metadata": metadata})
            report["recoveredBackplates"] += 1
        else:
            crop = source[y:y + h, x:x + w]
            text_mask = np.zeros((h, w), np.uint8)
            for item in foreground:
                if item.get("type") == "text":
                    raw = (item.get("metadata") or {}).get("rawOCRBBox")
                    tx1, ty1, tx2, ty2 = raw if isinstance(raw, list) and len(raw) == 4 else _box(item)
                    cv2.rectangle(text_mask, (max(0, round(tx1 - x - 2)), max(0, round(ty1 - y - 2))), (min(w - 1, round(tx2 - x + 2)), min(h - 1, round(ty2 - y + 2))), 255, -1)
            if np.count_nonzero(text_mask) / max(1, text_mask.size) > 0.20:
                report["issues"].append({"problem": "moduleTextTooLargeForCleanFallback", "groupId": group_id})
                continue
            visual_mask = np.zeros((h, w), np.uint8)
            for item in foreground:
                if item.get("type") == "text":
                    continue
                ix1, iy1, ix2, iy2 = _box(item)
                left, top = max(0, round(ix1 - x)), max(0, round(iy1 - y))
                right, bottom = min(w, round(ix2 - x)), min(h, round(iy2 - y))
                if right <= left or bottom <= top:
                    continue
                if item.get("type") == "ellipse":
                    cv2.ellipse(visual_mask, ((left + right) // 2, (top + bottom) // 2),
                                (max(1, (right - left) // 2), max(1, (bottom - top) // 2)),
                                0, 0, 360, 255, -1)
                elif item.get("type") == "image" and _mark_image_alpha(visual_mask, item, asset_dir, (x, y, w, h)):
                    pass
                else:
                    visual_mask[top:bottom, left:right] = 255
            if np.any(visual_mask):
                visual_mask = cv2.dilate(visual_mask, np.ones((3, 3), np.uint8))
            repair_mask = cv2.max(text_mask, visual_mask)
            clean = cv2.inpaint(crop, repair_mask, 3, cv2.INPAINT_TELEA) if np.any(repair_mask) else crop
            alpha = np.zeros((h, w), np.uint8)
            cv2.drawContours(alpha, [contour - np.array([[[x, y]]])], -1, 255, -1, cv2.LINE_AA)
            path = asset_dir / f"{identifier}.png"
            asset_dir.mkdir(parents=True, exist_ok=True)
            if not image_io.imwrite(str(path), np.dstack((clean, alpha))):
                report["issues"].append({"problem": "moduleFallbackWriteFailed", "groupId": group_id})
                continue
            metadata.update({"reconstructionStrategy": "cutout_image", "moduleMemberIds": [item["id"] for item in foreground if item.get("id")],
                             "textCleaned": bool(np.any(text_mask)), "foregroundCleaned": bool(np.any(visual_mask)),
                             "fallbackReason": "texturedModuleBackplate"})
            elements.append({"id": identifier, "type": "image", "x": x, "y": y, "width": w, "height": h, "rotation": 0, "zIndex": z_index, "groupId": group_id, "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1}, "metadata": metadata})
            report["moduleImageFallbacks"] += 1
    for item in active:
        if target_ids is not None and str(item.get("id")) not in target_ids:
            continue
        if item.get("type") != "image" or not item.get("src") or (item.get("metadata") or {}).get("suppressed"):
            continue
        path = _asset_path(item, asset_dir)
        if not path.exists():
            continue
        image = image_io.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3:
            continue
        is_round = (item.get("metadata") or {}).get("reconstructionStrategySource") == "round_contour"
        if is_round:
            report["roundCutoutsChecked"] += 1
            if image.shape[2] == 4 and _round_cutout_quality(image[:, :, 3]):
                continue
        if image.shape[2] == 4 and _transparent_corners(image[:, :, 3]) and not is_round:
            continue
        shape = _matching_ellipse(item, elements)
        alpha = _ellipse_mask(item, shape or item, image.shape[:2]) if shape is not None or is_round else _icon_contour_mask(item, image)
        if alpha is None:
            if is_round:
                report["roundCutoutIssues"] += 1
            if shape is not None or _looks_like_opaque_icon(item, image):
                report["squareCutouts"] += 1
                report["issues"].append({"problem": "squareCutoutUnresolved", "elementId": item.get("id")})
            continue
        report["squareCutouts"] += 1
        repaired_path = asset_dir / f"qa_contour_{path.stem}.png"
        rgba = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha if image.shape[2] == 3 or is_round else cv2.min(image[:, :, 3], alpha)
        if not image_io.imwrite(str(repaired_path), rgba):
            report["issues"].append({"problem": "cutoutRepairFailed", "elementId": item.get("id")})
            continue
        item["src"] = f"/media/assets/{project_id}/{repaired_path.name}"
        item.setdefault("metadata", {}).update({"transparent": True, "reconstructionStrategy": "transparent_image", "objectizationQAContourRepaired": True})
        report["repairedCutouts"] += 1
        if is_round and not _round_cutout_quality(alpha):
            report["roundCutoutIssues"] += 1
            report["issues"].append({"problem": "roundCutoutQualityLow", "elementId": item.get("id")})
    report["unresolvedBackplates"] = report["missingBackplates"] - report["recoveredBackplates"] - report["reboundBackplates"] - report["moduleImageFallbacks"]
    return report


def _mark_image_alpha(mask: np.ndarray, item: dict, asset_dir: Path,
                      plate: tuple[int, int, int, int]) -> bool:
    """Mask only visible pixels of a transparent foreground image."""
    path = _asset_path(item, asset_dir)
    image = image_io.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        return False
    px, py, width, height = plate
    ix1, iy1, ix2, iy2 = _box(item)
    left, top = max(0, round(ix1 - px)), max(0, round(iy1 - py))
    right, bottom = min(width, round(ix2 - px)), min(height, round(iy2 - py))
    if right <= left or bottom <= top:
        return False
    scaled = cv2.resize(image[:, :, 3], (max(1, round(ix2 - ix1)), max(1, round(iy2 - iy1))),
                        interpolation=cv2.INTER_LINEAR)
    offset_x, offset_y = left - round(ix1 - px), top - round(iy1 - py)
    visible = scaled[offset_y:offset_y + bottom - top, offset_x:offset_x + right - left]
    if visible.shape != (bottom - top, right - left):
        return False
    region = mask[top:bottom, left:right]
    region[visible > 32] = 255
    return True


def _asset_path(item: dict, asset_dir: Path) -> Path:
    local = asset_dir / Path(str(item.get("src") or "")).name
    if local.is_file():
        return local
    return _path_from_src(item.get("src")) or local


def _local_plate(source: np.ndarray, foreground: list[dict]) -> tuple[int, int, int, int, str, bool, np.ndarray] | None:
    height, width = source.shape[:2]
    border = np.concatenate((source[0], source[-1], source[:, 0], source[:, -1]))
    page_color = np.median(border, axis=0).astype(np.int16)
    difference = np.max(np.abs(source.astype(np.int16) - page_color), axis=2)
    mask = np.uint8(difference >= 5) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = [(float(item.get("x") or 0) + float(item.get("width") or 0) / 2, float(item.get("y") or 0) + float(item.get("height") or 0) / 2) for item in foreground]
    candidate = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if w < 12 or h < 8 or area > width * height * 0.6 or cv2.contourArea(contour) / max(1, area) < 0.82:
            continue
        if sum(x <= cx <= x + w and y <= cy <= y + h for cx, cy in centers) < max(1, len(centers) - 1):
            continue
        region = source[y:y + h, x:x + w]
        median = np.median(region.reshape(-1, 3), axis=0).astype(np.int16)
        flat_fraction = float(np.mean(np.max(np.abs(region.astype(np.int16) - median), axis=2) <= 6))
        if np.max(np.abs(median - page_color)) < 5 or flat_fraction < 0.35:
            continue
        if candidate is None or area < candidate[0]:
            candidate = (area, x, y, w, h, f"#{median[2]:02X}{median[1]:02X}{median[0]:02X}", flat_fraction >= 0.65, contour)
    return candidate[1:] if candidate else None


def _matching_ellipse(image_item: dict, elements: list[dict]) -> dict | None:
    group = image_item.get("groupId") or (image_item.get("metadata") or {}).get("groupId")
    for item in elements:
        if item.get("type") != "ellipse" or (item.get("groupId") or (item.get("metadata") or {}).get("groupId")) != group:
            continue
        if group and _covered_fraction(item, _box(image_item)) >= 0.85:
            return item
    return None


def _ellipse_mask(image_item: dict, shape: dict, size: tuple[int, int]) -> np.ndarray | None:
    height, width = size
    x_scale = width / max(1.0, float(image_item.get("width") or 0))
    y_scale = height / max(1.0, float(image_item.get("height") or 0))
    cx = (float(shape.get("x") or 0) + float(shape.get("width") or 0) / 2 - float(image_item.get("x") or 0)) * x_scale
    cy = (float(shape.get("y") or 0) + float(shape.get("height") or 0) / 2 - float(image_item.get("y") or 0)) * y_scale
    rx, ry = float(shape.get("width") or 0) * x_scale / 2, float(shape.get("height") or 0) * y_scale / 2
    if min(rx, ry) < 3 or rx > width * 0.6 or ry > height * 0.6:
        return None
    scale = 4
    alpha = np.zeros((height * scale, width * scale), np.uint8)
    cv2.ellipse(alpha, (round(cx * scale), round(cy * scale)), (round(rx * scale), round(ry * scale)), float(shape.get("rotation") or 0), 0, 360, 255, -1, cv2.LINE_AA)
    return cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)


def _icon_contour_mask(item: dict, image: np.ndarray) -> np.ndarray | None:
    if not _looks_like_opaque_icon(item, image):
        return None
    height, width = image.shape[:2]
    if min(height, width) < 12:
        return None
    pixels = image[:, :, :3]
    corners = np.array([pixels[0, 0], pixels[0, -1], pixels[-1, 0], pixels[-1, -1]], dtype=np.int16)
    if np.max(np.ptp(corners, axis=0)) > 12:
        return None
    exterior = np.median(corners, axis=0)
    foreground = np.uint8(np.max(np.abs(pixels.astype(np.int16) - exterior), axis=2) > 12) * 255
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    area_fraction = cv2.contourArea(contour) / max(1, width * height)
    if area_fraction < 0.25 or area_fraction > 0.88 or w < width * 0.72 or h < height * 0.72:
        return None
    alpha = np.zeros((height, width), np.uint8)
    cv2.drawContours(alpha, [contour], -1, 255, -1, cv2.LINE_AA)
    return alpha


def _looks_like_opaque_icon(item: dict, image: np.ndarray) -> bool:
    metadata = item.get("metadata") or {}
    semantic = str(item.get("role") or item.get("componentType") or metadata.get("componentType") or "").lower()
    if not any(word in semantic for word in ("icon", "badge", "emblem", "decorative")):
        return False
    if image.shape[0] < 12 or image.shape[1] < 12:
        return False
    pixels = image[:, :, :3]
    corners = np.array([pixels[0, 0], pixels[0, -1], pixels[-1, 0], pixels[-1, -1]], dtype=np.int16)
    exterior = np.median(corners, axis=0)
    return np.max(np.ptp(corners, axis=0)) <= 12 and np.mean(np.max(np.abs(pixels.astype(np.int16) - exterior), axis=2) > 12) > 0.15


def _transparent_corners(alpha: np.ndarray) -> bool:
    return max(int(alpha[0, 0]), int(alpha[0, -1]), int(alpha[-1, 0]), int(alpha[-1, -1])) < 32


def _round_cutout_quality(alpha: np.ndarray) -> bool:
    height, width = alpha.shape
    if min(height, width) < 12 or not _transparent_corners(alpha):
        return False
    coverage = float(np.mean(alpha > 128))
    return 0.42 <= coverage <= 0.86 and int(alpha[height // 2, width // 2]) > 220


def _box(item: dict) -> tuple[float, float, float, float]:
    x, y = float(item.get("x") or 0), float(item.get("y") or 0)
    return x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)


def _box_area(item: dict) -> float:
    return max(0.0, float(item.get("width") or 0) * float(item.get("height") or 0))


def _covered_fraction(item: dict, box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = _box(item)
    overlap = max(0.0, min(x2, box[2]) - max(x1, box[0])) * max(0.0, min(y2, box[3]) - max(y1, box[1]))
    return overlap / max(1.0, _box_area(item))
