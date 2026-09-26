from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


class AIReconstructionPlanner:
    """Turn Qwen's page-level plan into bounded, owned source-image regions.

    The model chooses which regions stay whole. Source pixels and existing
    OCR/CV boxes remain the geometry of the actual slide elements.
    """

    def apply(
        self,
        scene: dict[str, Any],
        source_path: Path,
        asset_dir: Path,
        project_id: str,
        page_index: int,
    ) -> dict[str, int]:
        stats = {"plannedModules": 0, "wholeImageRegions": 0, "plannerSuppressedElements": 0, "plannerDuplicateTexts": 0, "plannerSnappedRegions": 0, "plannerCvVisualRegions": 0}
        vision = scene.get("vision") or {}
        original_plan = vision.get("reconstructionPlan") or {}
        modules = (original_plan.get("modules") or []) if vision.get("aiUsed") else []
        if not isinstance(modules, list) or not modules:
            return stats

        canvas = scene.get("canvas") or {}
        width, height = int(canvas.get("width") or 0), int(canvas.get("height") or 0)
        if width < 32 or height < 32:
            return stats
        elements = scene.get("elements") or []
        by_id = {str(item.get("id")): item for item in elements}
        planned_assets: list[dict[str, Any]] = []
        used_boxes: list[tuple[int, int, int, int]] = []
        normalized_modules: list[dict[str, Any]] = []

        with Image.open(source_path) as source:
            source.load()
            for module in modules[:40]:
                if not isinstance(module, dict) or float(module.get("confidence") or 0) < 0.65:
                    continue
                module_box = _pixel_box(module.get("bbox"), width, height)
                if module_box is None:
                    continue
                module_id = str(module.get("id") or f"module_{stats['plannedModules'] + 1}")[:80]
                member_ids = {str(value) for value in module.get("memberIds", []) if str(value) in by_id and _coverage(by_id[str(value)], module_box) >= 0.35}
                editable_ids = {str(value) for value in module.get("editableIds", []) if str(value) in by_id and _coverage(by_id[str(value)], module_box) >= 0.35}
                ignore_ids = {str(value) for value in module.get("ignoreIds", []) if str(value) in by_id and _coverage(by_id[str(value)], module_box) >= 0.35}
                strategy = module.get("reconstructionStrategy") or module.get("strategy")
                strategy = {"mixed_component": "hybrid", "editable_text": "editable", "native_shape": "editable"}.get(strategy, strategy)
                if strategy not in {"editable", "whole_image", "hybrid", "ignore"}:
                    continue
                stats["plannedModules"] += 1
                normalized_modules.append({"moduleId": module_id, "bbox": module.get("bbox"), "role": module.get("role"), "reconstructionStrategy": module.get("reconstructionStrategy") or {"editable": "editable_text", "hybrid": "mixed_component"}.get(strategy, strategy), "resolvedStrategy": strategy, "visualComplexity": module.get("visualComplexity"), "editablePriority": module.get("editablePriority"), "confidence": module.get("confidence"), "bboxPixels": list(module_box), "children": module.get("children", []), "ownership": module.get("ownership", {}), "preserveWhole": bool(module.get("preserveWhole", strategy == "whole_image"))})

                for item_id in member_ids | editable_ids:
                    item = by_id[item_id]
                    if item.get("type") != "background":
                        item["groupId"] = module_id
                        item.setdefault("metadata", {})["plannerModuleId"] = module_id

                boxes = [module_box] if strategy == "whole_image" else [
                    box for raw in module.get("preserveRegions", [])
                    if (box := _pixel_box(raw, width, height)) is not None and _overlap_min(box, module_box) >= 0.25
                ] if strategy == "hybrid" else []
                cv_boxes = _cv_visual_boxes(scene.get("regions") or [], module_box, width, height) if strategy == "hybrid" else []
                for box in boxes + cv_boxes:
                    from_cv = box in cv_boxes
                    snapped = _snap_to_layout_region(box, scene.get("regions") or [], width, height)
                    was_snapped = snapped != box
                    box = snapped
                    # Source pixels can have only one image owner. Even a narrow
                    # overlap may erase a line in one asset while the other asset
                    # still owns its editable textbox.
                    if len(planned_assets) >= 24 or any(_overlap_min(box, prior) >= 0.05 for prior in used_boxes) or _cuts_through_text(box, elements):
                        continue
                    if any(
                        item.get("type") == "text" and item.get("role") == "main_title"
                        and float((item.get("bbox") or {}).get("width") or 0) >= width * 0.25
                        and _coverage(item, box) >= 0.55
                        for item in elements
                    ):
                        continue
                    x1, y1, x2, y2 = box
                    asset_dir.mkdir(parents=True, exist_ok=True)
                    module_dir = asset_dir.parent / "module_assets" / f"page_{page_index}"
                    clean_dir = asset_dir.parent / "text_clean_assets" / f"page_{page_index}"
                    module_dir.mkdir(parents=True, exist_ok=True)
                    clean_dir.mkdir(parents=True, exist_ok=True)
                    asset_id = f"planner_page_{page_index}_region_{len(planned_assets) + 1:03d}"
                    asset_path = asset_dir / f"{asset_id}.png"
                    covered = [
                        item for item in elements
                        if item.get("type") not in {"background", "group"} and _covered_by_asset(item, box)
                    ]
                    editable_text = [
                        item for item in covered
                        if item.get("type") == "text" and item.get("id") not in ignore_ids
                        and item.get("role") not in {"logo", "decorative_text"}
                        and str(item.get("text") or "").strip()
                    ]
                    source.crop(box).convert("RGB").save(module_dir / f"{asset_id}.png", format="PNG")
                    crop = np.asarray(source.crop(box).convert("RGB"))[:, :, ::-1].copy()
                    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                    for text_item in editable_text:
                        bounds = text_item.get("bbox") or {}
                        raw = (text_item.get("metadata") or {}).get("rawOCRBBox")
                        if isinstance(raw, list) and len(raw) == 4:
                            left, top, right, bottom = map(float, raw)
                        else:
                            left, top = float(bounds.get("left", 0)), float(bounds.get("top", 0))
                            right, bottom = left + float(bounds.get("width", 0)), top + float(bounds.get("height", 0))
                        tx1 = max(0, round(left - x1))
                        ty1 = max(0, round(top - y1))
                        tx2 = min(x2 - x1, round(right - x1))
                        ty2 = min(y2 - y1, round(bottom - y1))
                        if tx2 > tx1 and ty2 > ty1:
                            pad_x = min(12, max(2, round((tx2 - tx1) * 0.05)))
                            pad_y = max(2, round((ty2 - ty1) * 0.20))
                            cv2.rectangle(mask, (max(0, tx1 - pad_x), max(0, ty1 - pad_y)), (min(mask.shape[1] - 1, tx2 + pad_x), min(mask.shape[0] - 1, ty2 + pad_y)), 255, -1)
                    text_area_ratio = float(np.count_nonzero(mask)) / max(1, mask.size)
                    safe_to_clean = text_area_ratio <= 0.20
                    if np.any(mask) and safe_to_clean:
                        crop = _clean_text_from_asset(crop, mask)
                    cv2.imwrite(str(asset_path), crop)
                    cv2.imwrite(str(clean_dir / f"{asset_id}.png"), crop)
                    z_index = max((int(item.get("zIndex") or 0) for item in covered), default=1) + 1
                    planned_assets.append({
                        "id": asset_id,
                        "type": "image",
                        "role": module.get("role") or "complex_visual",
                        "componentType": "whole_image_region",
                        "groupId": module_id,
                        "bbox": {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1},
                        "zIndex": z_index,
                        "src": f"/media/assets/{project_id}/{asset_path.name}",
                        "style": {"opacity": 1},
                        "metadata": {"reconstructionStrategy": "local_image", "reconstructionStrategySource": "planner", "preserveWholeAsset": True, "doNotVectorize": True, "plannerModuleId": module_id, "textCleaned": bool(np.any(mask)) and safe_to_clean, "editableTextIds": [item["id"] for item in editable_text] if safe_to_clean else [], "fallbackReason": "text_area_too_large" if not safe_to_clean else None},
                    })
                    used_boxes.append(box)
                    stats["wholeImageRegions"] += 1
                    stats["plannerSnappedRegions"] += int(was_snapped)
                    stats["plannerCvVisualRegions"] += int(from_cv)
                    for item in covered:
                        metadata = item.setdefault("metadata", {})
                        if item in editable_text and np.any(mask) and safe_to_clean:
                            metadata.update({"plannerModuleId": module_id, "reconstructionStrategy": "editable_text", "reconstructionStrategySource": "planner", "textCleanedFromAsset": asset_id})
                            item["zIndex"] = z_index + 1
                            continue
                        if not metadata.get("suppressed"):
                            stats["plannerSuppressedElements"] += 1
                        metadata.update({"suppressed": True, "ownedBy": asset_id, "reconstructionStrategy": "group", "reconstructionStrategySource": "planner"})

                for item_id in ignore_ids:
                    item = by_id[item_id]
                    if item.get("type") == "background":
                        continue
                    metadata = item.setdefault("metadata", {})
                    if not metadata.get("suppressed"):
                        stats["plannerSuppressedElements"] += 1
                    metadata.update({"suppressed": True, "suppressRender": True, "reconstructionStrategy": "group", "reconstructionStrategySource": "planner"})

        if stats["plannedModules"]:
            stats["plannerDuplicateTexts"] = _suppress_duplicate_text(elements)
            stats["plannerSuppressedElements"] += stats["plannerDuplicateTexts"]
        elements.extend(planned_assets)
        scene["reconstructionPlan"] = {**original_plan, "modules": normalized_modules, "assets": [item["id"] for item in planned_assets]}
        return stats


def _clean_text_from_asset(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill text on locally flat colors; inpaint only textured patches."""
    result = crop.copy()
    textured = mask.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    for component in range(1, count):
        x, y, width, height, _ = (int(value) for value in stats[component])
        pad = max(4, min(12, height // 3))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(crop.shape[1], x + width + pad), min(crop.shape[0], y + height + pad)
        ring = crop[y1:y2, x1:x2][mask[y1:y2, x1:x2] == 0]
        if len(ring) < 16:
            continue
        median = np.median(ring, axis=0)
        near_flat = float((np.linalg.norm(ring.astype(np.float32) - median.astype(np.float32), axis=1) < 22).mean())
        if near_flat < 0.78:
            continue
        region = labels[y:y + height, x:x + width] == component
        result[y:y + height, x:x + width][region] = np.asarray(median, dtype=np.uint8)
        textured[y:y + height, x:x + width][region] = 0
    if np.any(textured):
        result = cv2.inpaint(result, textured, 4, cv2.INPAINT_TELEA)
    return result


def _pixel_box(raw: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(raw, dict):
        return None
    try:
        left, top, box_width, box_height = (float(raw[key]) for key in ("left", "top", "width", "height"))
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= left < 1 and 0 <= top < 1 and 0 < box_width <= 1 and 0 < box_height <= 1):
        return None
    if left + box_width > 1.01 or top + box_height > 1.01 or box_width * box_height > 0.80:
        return None
    x1, y1 = max(0, round(left * width)), max(0, round(top * height))
    x2, y2 = min(width, round((left + box_width) * width)), min(height, round((top + box_height) * height))
    return (x1, y1, x2, y2) if x2 - x1 >= 24 and y2 - y1 >= 24 else None


def _overlap_min(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    area = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(0, min(left[3], right[3]) - max(left[1], right[1]))
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return area / max(1, min(left_area, right_area))


def _snap_to_layout_region(
    box: tuple[int, int, int, int], regions: list[dict[str, Any]], width: int, height: int
) -> tuple[int, int, int, int]:
    best, best_score = box, 0.0
    for region in regions:
        if region.get("type") not in {"figure", "image", "chart", "table"} or float(region.get("confidence") or 0) < 0.55:
            continue
        raw = region.get("bbox") or {}
        try:
            x1, y1 = round(float(raw["left"])), round(float(raw["top"]))
            x2, y2 = round(x1 + float(raw["width"])), round(y1 + float(raw["height"]))
        except (KeyError, TypeError, ValueError):
            continue
        candidate = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        if candidate[2] <= candidate[0] or candidate[3] <= candidate[1]:
            continue
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        candidate_area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
        ratio = candidate_area / max(1, box_area)
        if not 0.2 <= ratio <= 5:
            continue
        overlap = _overlap_min(box, candidate)
        score = overlap * float(region["confidence"])
        if overlap >= 0.30 and score > best_score:
            best, best_score = candidate, score
    return best


def _cv_visual_boxes(regions: list[dict[str, Any]], module_box: tuple[int, int, int, int], width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for region in regions:
        if region.get("type") not in {"image", "figure", "chart", "table"} or float(region.get("confidence") or 0) < 0.65:
            continue
        raw = region.get("bbox") or {}
        try:
            x1, y1 = round(float(raw["left"])), round(float(raw["top"]))
            x2, y2 = round(x1 + float(raw["width"])), round(y1 + float(raw["height"]))
        except (KeyError, TypeError, ValueError):
            continue
        box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        center_x = (box[0] + box[2]) / 2
        area = (box[2] - box[0]) * (box[3] - box[1])
        if box[2] - box[0] < 50 or box[3] - box[1] < 50 or area < width * height * 0.008:
            continue
        if not module_box[0] <= center_x <= module_box[2] or box[1] < module_box[1] or box[3] > height * 0.92:
            continue
        boxes.append(box)
    return boxes[:12]


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    area = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(0, min(left[3], right[3]) - max(left[1], right[1]))
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return area / max(1, left_area + right_area - area)


def _coverage(item: dict[str, Any], box: tuple[int, int, int, int]) -> float:
    bounds = item.get("bbox") or {}
    x, y = float(bounds.get("left") or 0), float(bounds.get("top") or 0)
    width, height = float(bounds.get("width") or 0), float(bounds.get("height") or 0)
    if width <= 0 or height <= 0:
        return 0.0
    overlap = max(0, min(x + width, box[2]) - max(x, box[0])) * max(0, min(y + height, box[3]) - max(y, box[1]))
    return overlap / (width * height)


def _covered_by_asset(item: dict[str, Any], box: tuple[int, int, int, int]) -> bool:
    if _coverage(item, box) >= 0.60:
        return True
    bounds = item.get("bbox") or {}
    try:
        item_box = (round(float(bounds["left"])), round(float(bounds["top"])), round(float(bounds["left"] + bounds["width"])), round(float(bounds["top"] + bounds["height"])))
    except (KeyError, TypeError, ValueError):
        return False
    if item.get("type") == "image":
        return _overlap_min(item_box, box) >= 0.45
    if item.get("type") in {"rectangle", "roundedRectangle", "ellipse"}:
        return _iou(item_box, box) >= 0.45
    return False


def _cuts_through_text(box: tuple[int, int, int, int], elements: list[dict[str, Any]]) -> bool:
    for item in elements:
        if item.get("type") != "text" or not str(item.get("text") or "").strip():
            continue
        raw = (item.get("metadata") or {}).get("rawOCRBBox")
        bounds = item.get("bbox") or {}
        if isinstance(raw, list) and len(raw) == 4:
            left, top, right, bottom = (float(value) for value in raw)
        else:
            try:
                left, top = float(bounds["left"]), float(bounds["top"])
                right, bottom = left + float(bounds["width"]), top + float(bounds["height"])
            except (KeyError, TypeError, ValueError):
                continue
        area = max(1.0, (right - left) * (bottom - top))
        overlap = max(0.0, min(right, box[2]) - max(left, box[0])) * max(0.0, min(bottom, box[3]) - max(top, box[1]))
        if 0.03 < overlap / area < 0.95:
            return True
    return False


def _suppress_duplicate_text(elements: list[dict[str, Any]]) -> int:
    seen: dict[str, list[dict[str, Any]]] = {}
    suppressed = 0
    for item in elements:
        if item.get("type") != "text" or (item.get("metadata") or {}).get("suppressed"):
            continue
        normalized = "".join(str(item.get("text") or "").split()).casefold()
        if len(normalized) < 2:
            continue
        box = item.get("bbox") or {}
        try:
            bounds = (int(box["left"]), int(box["top"]), int(box["left"] + box["width"]), int(box["top"] + box["height"]))
        except (KeyError, TypeError, ValueError):
            continue
        duplicate = next((prior for prior in seen.get(normalized, []) if _iou(bounds, prior["bounds"]) >= 0.70), None)
        if duplicate is None:
            seen.setdefault(normalized, []).append({"item": item, "bounds": bounds})
            continue
        previous = duplicate["item"]
        loser, winner = (item, previous) if float(item.get("confidence") or 0) <= float(previous.get("confidence") or 0) else (previous, item)
        loser.setdefault("metadata", {}).update({"suppressed": True, "ownedBy": winner["id"], "reconstructionStrategy": "group", "reconstructionStrategySource": "planner"})
        duplicate["item"] = winner
        duplicate["bounds"] = bounds if winner is item else duplicate["bounds"]
        suppressed += 1
    return suppressed


__all__ = ["AIReconstructionPlanner"]
