from __future__ import annotations

from pathlib import Path
from typing import Any

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
        modules = ((vision.get("reconstructionPlan") or {}).get("modules") or []) if vision.get("aiUsed") else []
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
                strategy = module.get("strategy")
                if strategy not in {"editable", "whole_image", "hybrid"}:
                    continue
                stats["plannedModules"] += 1
                normalized_modules.append({"id": module_id, "role": module.get("role"), "strategy": strategy, "bboxPixels": list(module_box)})

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
                    if len(planned_assets) >= 24 or any(_overlap_min(box, prior) >= 0.65 for prior in used_boxes):
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
                    asset_id = f"planner_page_{page_index}_region_{len(planned_assets) + 1:03d}"
                    asset_path = asset_dir / f"{asset_id}.png"
                    source.crop(box).convert("RGB").save(asset_path, format="PNG")
                    covered = [
                        item for item in elements
                        if item.get("type") not in {"background", "group"} and _covered_by_asset(item, box)
                    ]
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
                        "metadata": {"reconstructionStrategy": "local_image", "reconstructionStrategySource": "planner", "preserveWholeAsset": True, "doNotVectorize": True, "plannerModuleId": module_id},
                    })
                    used_boxes.append(box)
                    stats["wholeImageRegions"] += 1
                    stats["plannerSnappedRegions"] += int(was_snapped)
                    stats["plannerCvVisualRegions"] += int(from_cv)
                    for item in covered:
                        metadata = item.setdefault("metadata", {})
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
        scene["reconstructionPlan"] = {"modules": normalized_modules, "assets": [item["id"] for item in planned_assets]}
        return stats


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
