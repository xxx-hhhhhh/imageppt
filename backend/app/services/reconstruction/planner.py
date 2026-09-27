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
        *,
        include_detected_visuals: bool = True,
        asset_prefix: str = "planner",
    ) -> dict[str, int]:
        stats = {"plannedModules": 0, "wholeImageRegions": 0, "cutoutImages": 0, "nativeShapesPlanned": 0, "plannerSuppressedElements": 0, "plannerDuplicateTexts": 0, "plannerSnappedRegions": 0, "plannerCvVisualRegions": 0}
        vision = scene.get("vision") or {}
        original_plan = vision.get("reconstructionPlan") or {}
        modules = (original_plan.get("modules") or []) if vision.get("aiUsed") else []
        if not isinstance(modules, list):
            modules = []

        canvas = scene.get("canvas") or {}
        width, height = int(canvas.get("width") or 0), int(canvas.get("height") or 0)
        if width < 32 or height < 32:
            return stats
        if include_detected_visuals:
            modules = list(modules) + _unplanned_visual_modules(scene.get("regions") or [], scene.get("elements") or [], width, height)
            modules += _segmented_visual_modules(scene.get("segmentation") or [], scene.get("elements") or [], width, height)
            modules += _contour_visual_modules(source_path, scene.get("elements") or [], width, height)
        elements = scene.setdefault("elements", [])
        by_id = {str(item.get("id")): item for item in elements}
        planned_assets: list[dict[str, Any]] = []
        used_boxes: list[tuple[int, int, int, int]] = []
        normalized_modules: list[dict[str, Any]] = []

        with Image.open(source_path) as source:
            source.load()
            for module in modules[:100]:
                if not isinstance(module, dict) or float(module.get("confidence") or 0) < 0.65:
                    continue
                requested = module.get("reconstructionStrategy") or module.get("strategy")
                strategy = {"editable": "editable_text", "whole_image": "cutout_image", "hybrid": "mixed_component"}.get(requested, requested)
                if strategy not in {"editable_text", "native_shape", "cutout_image", "mixed_component", "background", "ignore"}:
                    continue
                module_box = _pixel_box(module.get("bbox"), width, height, max_area=1.0 if strategy == "background" else 0.80)
                if module_box is None:
                    continue
                module_id = str(module.get("id") or f"module_{stats['plannedModules'] + 1}")[:80]
                member_ids = {str(value) for value in module.get("memberIds", []) if str(value) in by_id and _coverage(by_id[str(value)], module_box) >= 0.35}
                editable_ids = {str(value) for value in module.get("editableIds", []) if str(value) in by_id and _coverage(by_id[str(value)], module_box) >= 0.35}
                ignore_ids = {str(value) for value in module.get("ignoreIds", []) if str(value) in by_id and _coverage(by_id[str(value)], module_box) >= 0.35}
                if strategy == "ignore":
                    ignore_ids.update(member_ids)
                stats["plannedModules"] += 1
                normalized_modules.append({"moduleId": module_id, "bbox": module.get("bbox"), "role": module.get("role"), "reconstructionStrategy": strategy, "resolvedStrategy": strategy, "visualComplexity": module.get("visualComplexity"), "editablePriority": module.get("editablePriority"), "confidence": module.get("confidence"), "bboxPixels": list(module_box), "children": module.get("children", []), "ownership": module.get("ownership", {}), "preserveWhole": bool(module.get("preserveWhole", requested == "whole_image"))})

                for item_id in member_ids | editable_ids:
                    item = by_id[item_id]
                    if item.get("type") != "background":
                        item["groupId"] = module_id
                        item.setdefault("metadata", {})["plannerModuleId"] = module_id

                if strategy == "native_shape":
                    for item in elements:
                        if item.get("type") not in {"rectangle", "roundedRectangle", "ellipse", "line", "arrow"} or _coverage(item, module_box) < 0.75:
                            continue
                        if float(item.get("confidence") or item.get("finalConfidence") or 0) < 0.85 or float((item.get("metadata") or {}).get("visualComplexity") or 0) >= 0.35:
                            continue
                        item.setdefault("metadata", {}).update({"reconstructionStrategy": "native_shape", "reconstructionStrategySource": "planner", "layerRole": "native_shape"})
                        stats["nativeShapesPlanned"] += 1

                preserve_boxes = [
                    box for raw in module.get("preserveRegions", [])
                    if (box := _pixel_box(raw, width, height)) is not None and _overlap_min(box, module_box) >= 0.25
                ] if strategy in {"cutout_image", "mixed_component"} else []
                cv_boxes = _cv_visual_boxes(scene.get("regions") or [], module_box, width, height) if strategy in {"cutout_image", "mixed_component"} else []
                if strategy == "cutout_image":
                    boxes = [module_box] if requested == "whole_image" or not (preserve_boxes or cv_boxes) else preserve_boxes
                elif strategy == "mixed_component":
                    boxes = preserve_boxes if preserve_boxes or cv_boxes else [module_box]
                else:
                    boxes = []
                for box in boxes + cv_boxes:
                    from_cv = box in cv_boxes
                    snapped = _snap_to_layout_region(box, scene.get("regions") or [], width, height)
                    was_snapped = snapped != box
                    box = _extend_to_text_edges(snapped, elements, width, height)
                    if box is None:
                        continue
                    # Source pixels can have only one image owner. Even a narrow
                    # overlap may erase a line in one asset while the other asset
                    # still owns its editable textbox.
                    if len(planned_assets) >= 40 or any(_overlap_min(box, prior) >= 0.05 for prior in used_boxes):
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
                    asset_id = f"{asset_prefix}_page_{page_index}_region_{len(planned_assets) + 1:03d}"
                    asset_path = asset_dir / f"{asset_id}.png"
                    covered = [
                        item for item in elements
                        if item.get("type") not in {"background", "group"} and _covered_by_asset(item, box)
                    ]
                    editable_text = [
                        item for item in covered
                        if item.get("type") == "text" and item.get("id") not in ignore_ids
                        and item.get("role") not in {"logo", "decorative_text"}
                        and float(item.get("confidence") or 0) >= 0.5
                        and str(item.get("text") or "").strip()
                    ]
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
                    if text_area_ratio > 0.20 and (module_id.startswith(("detected_visual_", "segmented_visual_", "contour_visual_")) or strategy == "mixed_component"):
                        continue
                    safe_to_clean = text_area_ratio <= 0.20
                    source.crop(box).convert("RGB").save(module_dir / f"{asset_id}.png", format="PNG")
                    if np.any(mask) and safe_to_clean:
                        crop = _clean_text_from_asset(crop, mask)
                    alpha = _module_alpha(module, box)
                    if alpha is not None:
                        crop = np.dstack((crop, alpha))
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
                        "metadata": {"reconstructionStrategy": "cutout_image", "reconstructionStrategySource": "planner", "layerRole": "cutout_image", "preserveWholeAsset": True, "doNotVectorize": True, "plannerModuleId": module_id, "textCleaned": bool(np.any(mask)) and safe_to_clean, "editableTextIds": [item["id"] for item in editable_text] if safe_to_clean else [], "fallbackReason": "text_area_too_large" if not safe_to_clean else None},
                    })
                    used_boxes.append(box)
                    stats["wholeImageRegions"] += 1
                    stats["cutoutImages"] += 1
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


def _pixel_box(raw: Any, width: int, height: int, *, max_area: float = 0.80) -> tuple[int, int, int, int] | None:
    if not isinstance(raw, dict):
        return None
    try:
        left, top, box_width, box_height = (float(raw[key]) for key in ("left", "top", "width", "height"))
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= left < 1 and 0 <= top < 1 and 0 < box_width <= 1 and 0 < box_height <= 1):
        return None
    if left + box_width > 1.01 or top + box_height > 1.01 or box_width * box_height > max_area:
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
        kind = region.get("type")
        if kind not in {"image", "figure", "chart", "table"} or float(region.get("confidence") or 0) < (0.55 if kind == "image" else 0.60 if kind in {"figure", "chart"} else 0.65):
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
        min_side = 24 if kind == "image" else 50
        min_area = max(400, width * height * 0.00025) if kind == "image" else width * height * 0.008
        if box[2] - box[0] < min_side or box[3] - box[1] < min_side or area < min_area:
            continue
        if not module_box[0] <= center_x <= module_box[2] or box[1] < module_box[1] or box[3] > height * 0.99:
            continue
        boxes.append(box)
    return boxes[:12]


def _unplanned_visual_modules(regions: list[dict[str, Any]], elements: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    """Promote detected photos, charts and figures outside AI module bounds."""
    modules = []
    for index, region in enumerate(regions):
        kind = region.get("type")
        confidence = float(region.get("confidence") or 0)
        if kind not in {"image", "figure", "chart", "table"} or confidence < (0.55 if kind == "image" else 0.60):
            continue
        raw = region.get("bbox") or {}
        try:
            x, y = float(raw["left"]), float(raw["top"])
            w, h = float(raw["width"]), float(raw["height"])
        except (KeyError, TypeError, ValueError):
            continue
        for item in elements:
            if item.get("type") != "text" or float(item.get("confidence") or 0) < 0.5:
                continue
            text_box = item.get("bbox") or {}
            tx, ty = float(text_box.get("left") or 0), float(text_box.get("top") or 0)
            tw = float(text_box.get("width") or 0)
            if ty > y + h * 0.55 and ty < y + h and max(0, min(x + w, tx + tw) - max(x, tx)) > w * 0.65:
                h = min(h, ty - y - 2)
        if w < 24 or h < 24 or w * h < width * height * 0.00025 or w * h > width * height * 0.35:
            continue
        bbox = {"left": max(0, x / width), "top": max(0, y / height), "width": min(w, width - x) / width, "height": min(h, height - y) / height}
        if bbox["width"] <= 0 or bbox["height"] <= 0:
            continue
        modules.append({
            "id": f"detected_visual_{index + 1}",
            "role": kind,
            "bbox": bbox,
            "confidence": confidence,
            "reconstructionStrategy": "cutout_image",
            "preserveRegions": [bbox],
            "editablePriority": "normal",
        })
    return modules[:32]


def _segmented_visual_modules(segments: list[dict[str, Any]], elements: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    modules = []
    for index, segment in enumerate(segments):
        raw = segment.get("bbox") or {}
        try:
            x, y, w, h = (float(raw[key]) for key in ("left", "top", "width", "height"))
        except (KeyError, TypeError, ValueError):
            continue
        area_ratio = w * h / max(1, width * height)
        if not 0.0003 <= area_ratio <= 0.30 or w < 20 or h < 20 or _text_occupancy((x, y, x + w, y + h), elements) > 0.12:
            continue
        mask = np.asarray(segment.get("mask") or [], dtype=np.uint8)
        if mask.ndim != 2 or not mask.size:
            continue
        fill = float(np.count_nonzero(mask)) / mask.size
        if not 0.15 <= fill <= 0.98:
            continue
        bbox = {"left": x / width, "top": y / height, "width": w / width, "height": h / height}
        modules.append({"id": f"segmented_visual_{index + 1}", "role": "complex_visual", "reconstructionStrategy": "cutout_image", "bbox": bbox, "preserveRegions": [bbox], "confidence": max(0.66, float(segment.get("confidence") or 0)), "maskBBox": [round(x), round(y), round(x + w), round(y + h)], "mask": mask})
    return modules[:24]


def _contour_visual_modules(source_path: Path, elements: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        return []
    edges = cv2.Canny(image, 70, 160)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    proposals = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        ratio = w * h / max(1, width * height)
        if w < 20 or h < 20 or not 0.0002 <= ratio <= 0.12 or max(w / h, h / w) > 6:
            continue
        if (x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2) and ratio > 0.02:
            continue
        fill = cv2.contourArea(contour) / max(1, w * h)
        if fill < 0.40 or _text_occupancy((x, y, x + w, y + h), elements) > 0.08:
            continue
        patch = image[y:y + h, x:x + w]
        interior = np.median(patch.reshape(-1, 3), axis=0)
        pad = 3
        surround = image[max(0, y - pad):min(height, y + h + pad), max(0, x - pad):min(width, x + w + pad)]
        border = np.concatenate((surround[0], surround[-1], surround[:, 0], surround[:, -1]))
        contrast = float(np.linalg.norm(interior - np.median(border, axis=0)))
        if contrast < 18:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour - np.array([[[x, y]]])], -1, 255, -1)
        proposals.append((y, x, w, h, mask))
    modules = []
    for index, (y, x, w, h, mask) in enumerate(sorted(proposals)[:32]):
        bbox = {"left": x / width, "top": y / height, "width": w / width, "height": h / height}
        modules.append({"id": f"contour_visual_{index + 1}", "role": "decorative_visual", "reconstructionStrategy": "cutout_image", "bbox": bbox, "preserveRegions": [bbox], "confidence": 0.7, "maskBBox": [x, y, x + w, y + h], "mask": mask})
    return modules


def _text_occupancy(box: tuple[float, float, float, float], elements: list[dict[str, Any]]) -> float:
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    covered = 0.0
    for item in elements:
        if item.get("type") != "text" or float(item.get("confidence") or 0) < 0.5:
            continue
        raw = item.get("bbox") or {}
        x, y, w, h = (float(raw.get(key) or 0) for key in ("left", "top", "width", "height"))
        covered += max(0, min(box[2], x + w) - max(box[0], x)) * max(0, min(box[3], y + h) - max(box[1], y))
    return covered / area


def _module_alpha(module: dict[str, Any], crop_box: tuple[int, int, int, int]) -> np.ndarray | None:
    raw_mask = module.get("mask")
    mask_box = module.get("maskBBox")
    if raw_mask is None or not isinstance(mask_box, list) or len(mask_box) != 4:
        return None
    mask = np.asarray(raw_mask, dtype=np.uint8)
    if mask.ndim != 2:
        return None
    mx1, my1, mx2, my2 = map(int, mask_box)
    expected = (max(1, my2 - my1), max(1, mx2 - mx1))
    if mask.shape != expected:
        mask = cv2.resize(mask, (expected[1], expected[0]), interpolation=cv2.INTER_NEAREST)
    x1, y1, x2, y2 = crop_box
    alpha = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    left, top = max(x1, mx1), max(y1, my1)
    right, bottom = min(x2, mx2), min(y2, my2)
    if right <= left or bottom <= top:
        return None
    alpha[top - y1:bottom - y1, left - x1:right - x1] = mask[top - my1:bottom - my1, left - mx1:right - mx1]
    return alpha


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


def _extend_to_text_edges(box: tuple[int, int, int, int], elements: list[dict[str, Any]], width: int, height: int) -> tuple[int, int, int, int] | None:
    """Include nearby labels instead of slicing their glyphs at a crop edge."""
    current = box
    max_extension = max(12, round(min(box[2] - box[0], box[3] - box[1]) * 0.25))
    for _ in range(3):
        if not _cuts_through_text(current, elements):
            return current
        left, top, right, bottom = current
        changed = False
        for item in elements:
            if item.get("type") != "text" or not str(item.get("text") or "").strip():
                continue
            raw = (item.get("metadata") or {}).get("rawOCRBBox")
            bounds = item.get("bbox") or {}
            if isinstance(raw, list) and len(raw) == 4:
                tx1, ty1, tx2, ty2 = map(float, raw)
            else:
                try:
                    tx1, ty1 = float(bounds["left"]), float(bounds["top"])
                    tx2, ty2 = tx1 + float(bounds["width"]), ty1 + float(bounds["height"])
                except (KeyError, TypeError, ValueError):
                    continue
            area = max(1.0, (tx2 - tx1) * (ty2 - ty1))
            overlap = max(0.0, min(right, tx2) - max(left, tx1)) * max(0.0, min(bottom, ty2) - max(top, ty1))
            if not 0.03 < overlap / area < 0.95:
                continue
            candidate = (max(0, min(left, int(tx1) - 2)), max(0, min(top, int(ty1) - 2)), min(width, max(right, int(np.ceil(tx2)) + 2)), min(height, max(bottom, int(np.ceil(ty2)) + 2)))
            if any(abs(candidate[index] - box[index]) > max_extension for index in range(4)):
                return None
            left, top, right, bottom = candidate
            changed = True
        current = (left, top, right, bottom)
        if not changed:
            return None
    return current if not _cuts_through_text(current, elements) else None


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
