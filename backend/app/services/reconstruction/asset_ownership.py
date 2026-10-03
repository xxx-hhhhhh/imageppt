"""Keep visual assets complete after final text ownership has been resolved."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.services.pptx.renderer import _path_from_src


def mark_image_dominant_ocr(source_path: Path, layout: dict) -> list[str]:
    """Identify uncertain OCR boxes dominated by a single visual mass.

    Text such as chart numbers has separate strokes/glyphs. A solar panel,
    photograph or icon can instead fill almost the whole OCR box with one
    connected region. Require both source evidence and a larger movable owner;
    recognition confidence alone never changes ownership in this path.
    """
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        return []
    marked = []
    for item in layout.get("elements", []):
        confidence = item.get("confidence", item.get("finalConfidence"))
        if item.get("type") != "text" or _hidden(item) or confidence is None or not 0 <= float(confidence) < .75:
            continue
        x1, y1, x2, y2 = [round(v) for v in _text_box(item)]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(source.shape[1], x2), min(source.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        owners = [asset for asset in layout.get("elements", []) if asset.get("type") == "image"
                  and not _hidden(asset) and (asset.get("metadata") or {}).get("preserveWholeAsset")
                  and _overlap_fraction((x1, y1, x2, y2), _image_box(asset)) >= .75
                  and float(asset.get("width") or 0) * float(asset.get("height") or 0) >= area * 3]
        if not owners:
            continue
        region = source[y1:y2, x1:x2]
        surface = np.median(np.concatenate((region[0], region[-1], region[:, 0], region[:, -1])), axis=0)
        ink = np.uint8(np.max(np.abs(region.astype(np.float32) - surface), axis=2) > 35)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        dominant = max(stats[1:, cv2.CC_STAT_AREA], default=0)
        if count <= 1 or dominant < area * .35:
            continue
        dominant_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        component = stats[dominant_label]
        component_area = int(component[cv2.CC_STAT_WIDTH]) * int(component[cv2.CC_STAT_HEIGHT])
        if component_area < area * .70 or dominant / component_area < .50:
            continue
        # Genuine single digits/letters may be one connected component too.
        # Their enclosed counters distinguish them from a filled visual mass.
        contours, hierarchy = cv2.findContours(np.uint8(labels == dominant_label), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hole_area = sum(cv2.contourArea(contour) for index, contour in enumerate(contours)
                        if hierarchy is not None and hierarchy[0, index, 3] >= 0)
        if hole_area > max(4, component_area * .035):
            continue
        metadata = item.setdefault("metadata", {})
        metadata["imageDominantOCREvidence"] = {"connectedPixelFraction": round(float(dominant / area), 4)}
        marked.append(str(item["id"]))
    return marked


def _overlap_fraction(box: tuple, owner: tuple) -> float:
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    overlap = max(0, min(box[2], owner[2]) - max(box[0], owner[0])) * max(0, min(box[3], owner[3]) - max(box[1], owner[1]))
    return overlap / area if area else 0


def _uncertain_visual_candidate(item: dict) -> bool:
    confidence = item.get("confidence", item.get("finalConfidence"))
    if confidence is None or not 0 <= float(confidence) < .75:
        return False
    return float(confidence) < .5 or bool((item.get("metadata") or {}).get("imageDominantOCREvidence"))


def is_badge_owned_text(item: dict, layout: dict) -> bool:
    """An OCR fragment inside a complete badge is visual content, not missing text."""
    if item.get("type") != "text" or not _hidden(item):
        return False
    item_id = str(item.get("id"))
    owner_id = str((item.get("metadata") or {}).get("ownedBy") or "")
    for asset in layout.get("elements", []):
        if asset.get("type") != "image":
            continue
        metadata = asset.get("metadata") or {}
        whole_badge = metadata.get("wholeBadgeAsset") or metadata.get("componentType") == "wholeBadgeImage"
        if str(asset.get("id")) == owner_id and whole_badge:
            return True
        image_text_ids = [*metadata.get("editableTextIds", []), *metadata.get("restoredImageOwnedTextIds", [])]
        if (whole_badge and item_id in image_text_ids
                and _overlap(_text_box(item), _image_box(asset))):
            return True
    return False


def is_uncertain_image_owned_text(item: dict, layout: dict) -> bool:
    """Honor the planner's image fallback for OCR below its text confidence gate.

    This never claims unowned/missing text or text already erased from an asset.
    Reliable ordinary labels continue to require an editable owner.
    """
    if item.get("type") != "text" or not _hidden(item):
        return False
    if not _uncertain_visual_candidate(item):
        return False
    owner_id = str((item.get("metadata") or {}).get("ownedBy") or "")
    owner = next((asset for asset in layout.get("elements", [])
                  if str(asset.get("id")) == owner_id and asset.get("type") == "image" and not _hidden(asset)), None)
    if owner is None:
        return False
    metadata = owner.get("metadata") or {}
    if not metadata.get("preserveWholeAsset") or str(item.get("id")) in {str(value) for value in metadata.get("editableTextIds") or []}:
        return False
    x1, y1, x2, y2 = _text_box(item)
    left, top, right, bottom = _image_box(owner)
    area = max(0, x2 - x1) * max(0, y2 - y1)
    intersection = max(0, min(x2, right) - max(x1, left)) * max(0, min(y2, bottom) - max(y1, top))
    return area > 0 and intersection / area >= 0.75


def preserve_uncertain_text_as_visual(source_path: Path, layout: dict, item: dict,
                                      asset_dir: Path, project_id: str, *, prefix: str) -> list[str] | None:
    """Keep a dubious OCR fragment in a movable visual instead of erasing it."""
    if item.get("type") != "text" or not _uncertain_visual_candidate(item):
        return None
    box = _text_box(item)
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    if area <= 0:
        return None
    owners = []
    for asset in layout.get("elements", []):
        if asset.get("type") != "image" or _hidden(asset) or not (asset.get("metadata") or {}).get("preserveWholeAsset"):
            continue
        x1, y1, x2, y2 = _image_box(asset)
        overlap = max(0, min(box[2], x2) - max(box[0], x1)) * max(0, min(box[3], y2) - max(box[1], y1))
        path = _path_from_src(asset.get("src"))
        if overlap / area >= .75 and path is not None and path.is_file():
            owners.append(asset)
    if not owners:
        return None
    owner = min(owners, key=lambda asset: float(asset.get("width") or 0) * float(asset.get("height") or 0))
    metadata = item.setdefault("metadata", {})
    metadata.update({"suppressed": True, "ownedBy": owner["id"], "reconstructionStrategy": "group",
                     "sourceContentPreserved": True, "fallbackReason": "uncertain_ocr_preserved_in_visual"})
    metadata.pop("textOwner", None)
    metadata.pop("ghostingDetected", None)
    return restore_image_owned_text(source_path, layout, asset_dir, project_id,
                                   prefix=prefix, target_boxes=[list(box)])


def restore_image_owned_text(source_path: Path, layout: dict, asset_dir: Path,
                             project_id: str, *, prefix: str = "owned_text",
                             target_boxes: list[list[float]] | None = None) -> list[str]:
    """Restore source pixels of hidden text without duplicating editable text.

    The image is staged under a new name before its scene reference changes.
    Its original alpha is retained so a circular badge stays circular.
    """
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        return []
    height, width = source.shape[:2]
    by_id = {str(item.get("id")): item for item in layout.get("elements", [])}
    restored: list[str] = []
    for item in layout.get("elements", []):
        metadata = item.get("metadata") or {}
        if item.get("type") != "image" or not metadata.get("textCleaned") or any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        text_ids = [str(value) for value in metadata.get("editableTextIds") or []]
        hidden_ids = [text_id for text_id in text_ids if text_id in by_id and _hidden(by_id[text_id])]
        if not hidden_ids:
            continue
        x, y = round(float(item.get("x") or 0)), round(float(item.get("y") or 0))
        w, h = round(float(item.get("width") or 0)), round(float(item.get("height") or 0))
        box = (x, y, x + w, y + h)
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            continue
        if target_boxes is not None and not any(_overlap(box, candidate) for candidate in target_boxes):
            continue
        active_text = [other for other in layout.get("elements", []) if other.get("type") == "text"
                       and not _hidden(other) and _overlap(box, _text_box(other))]
        prior_path = _path_from_src(item.get("src"))
        if prior_path is None or not prior_path.is_file():
            continue
        prior = cv2.imread(str(prior_path), cv2.IMREAD_UNCHANGED)
        if prior is None or prior.ndim != 3 or prior.shape[2] not in (3, 4):
            continue
        rgb = cv2.resize(source[y:y + h, x:x + w], (prior.shape[1], prior.shape[0]), interpolation=cv2.INTER_LINEAR)
        candidate = prior.copy()
        restored_ids = hidden_ids
        if len(hidden_ids) == len(text_ids) and not active_text:
            candidate[:, :, :3] = rgb
        else:
            # A mixed module can contain both image-owned and editable labels.
            # Restore only the former; retain the cleaned surface beneath the latter.
            protection = np.zeros(prior.shape[:2], np.uint8)
            for text in active_text:
                _mark_local_text(protection, _text_box(text), box, padding=2)
            restored_ids = []
            for text_id in hidden_ids:
                region = np.zeros(prior.shape[:2], np.uint8)
                _mark_local_text(region, _text_box(by_id[text_id]), box, padding=2)
                selected = (region > 0) & (protection == 0)
                if not np.any(selected):
                    continue
                candidate[selected, :3] = rgb[selected]
                restored_ids.append(text_id)
            if not restored_ids:
                continue
        asset_dir.mkdir(parents=True, exist_ok=True)
        path = asset_dir / f"{prefix}_{item['id']}_{uuid4().hex[:8]}.png"
        if not cv2.imwrite(str(path), candidate):
            continue
        item["src"] = f"/media/assets/{project_id}/{path.name}"
        remaining_ids = [text_id for text_id in text_ids if text_id not in restored_ids]
        metadata.update({"textCleaned": bool(remaining_ids), "editableTextIds": remaining_ids, "sourceContentPreserved": True,
                         "restoredImageOwnedTextIds": sorted(set(metadata.get("restoredImageOwnedTextIds") or []) | set(restored_ids))})
        restored.append(str(item["id"]))
    return restored


def _mark_local_text(mask: np.ndarray, text_box: tuple[float, float, float, float],
                     image_box: tuple[int, int, int, int], *, padding: int) -> None:
    x, y, right, bottom = image_box
    sx, sy = mask.shape[1] / (right - x), mask.shape[0] / (bottom - y)
    tx1, ty1, tx2, ty2 = text_box
    left = max(0, min(mask.shape[1], round((tx1 - x - padding) * sx)))
    top = max(0, min(mask.shape[0], round((ty1 - y - padding) * sy)))
    end_x = max(0, min(mask.shape[1], round((tx2 - x + padding) * sx)))
    end_y = max(0, min(mask.shape[0], round((ty2 - y + padding) * sy)))
    if end_x > left and end_y > top:
        mask[top:end_y, left:end_x] = 255


def _hidden(item: dict) -> bool:
    metadata = item.get("metadata") or {}
    return any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy"))


def _text_box(item: dict) -> tuple[float, float, float, float]:
    raw = (item.get("metadata") or {}).get("rawOCRBBox")
    if isinstance(raw, list) and len(raw) == 4:
        return tuple(float(value) for value in raw)
    x, y = float(item.get("x") or 0), float(item.get("y") or 0)
    return x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)


def _image_box(item: dict) -> tuple[float, float, float, float]:
    x, y = float(item.get("x") or 0), float(item.get("y") or 0)
    return x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)


def _overlap(first: tuple[float, float, float, float], second: tuple[float, float, float, float] | list[float]) -> bool:
    return min(first[2], second[2]) > max(first[0], second[0]) and min(first[3], second[3]) > max(first[1], second[1])
