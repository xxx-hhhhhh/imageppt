"""Pixel and ownership audit for white, objectized slide reconstructions."""

from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np

from app.services.reconstruction.asset_ownership import is_badge_owned_text
from app.services.reconstruction.residual_objects import is_meaningful_stroke, visual_candidate_mask


VISUAL_TYPES = {"image", "rectangle", "roundedRectangle", "ellipse", "line", "arrow"}


def audit_objectization(source_path: Path, background_path: Path, preview_path: Path, layout: dict,
                        debug_path: Path | None = None) -> dict:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    preview = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
    report = {"whiteBackground": False, "missingBackplates": 0, "missingVisualObjects": 0, "blankVisualOwners": 0,
              "visualMismatchRegions": 0, "visualMismatchPixels": 0, "salientVisualPixels": 0,
              "missingVisualPixels": 0, "largestMissingVisualRegion": 0, "retainedVisualCoverage": 1.0,
              "backgroundResidualRegions": 0, "ownerRegions": [], "issues": []}
    if source is None or background is None or preview is None or source.shape != background.shape or source.shape != preview.shape:
        report["issues"].append({"problem": "objectizationAuditUnavailable"})
        return report
    height, width = source.shape[:2]
    report["whiteBackground"] = bool(np.mean(np.all(background >= 250, axis=2)) >= 0.995)
    active = [item for item in layout.get("elements", []) if item.get("type") != "background" and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    owner_mask = np.zeros((height, width), np.uint8)
    for item in active:
        box = _box(item, width, height)
        if box is None:
            continue
        role = "editable_text" if item.get("type") == "text" else "movable_image" if item.get("type") == "image" else "native_shape"
        report["ownerRegions"].append({"elementId": item.get("id"), "owner": role, "bbox": list(box)})
        if item.get("type") in VISUAL_TYPES:
            x1, y1, x2, y2 = box
            owner_mask[y1:y2, x1:x2] = cv2.max(owner_mask[y1:y2, x1:x2], _visual_mask(item, box, source_path))
    page_color = np.median(np.concatenate((source[0], source[-1], source[:, 0], source[:, -1])), axis=0).astype(np.int16)
    salient = visual_candidate_mask(source)
    contrast = np.max(np.abs(source.astype(np.int16) - page_color), axis=2)
    # White details inside a badge or card can be intentional. Only count pixels
    # that actually changed to white, rather than every white source pixel.
    erased = np.all(preview >= 253, axis=2) & (np.max(np.abs(source.astype(np.int16) - preview.astype(np.int16)), axis=2) >= 2)
    lost = salient & erased & (owner_mask == 0)
    # Editable glyphs need not land on precisely the same raster pixels.
    for item in layout.get("elements", []):
        if item.get("type") == "text":
            if item not in active and is_badge_owned_text(item, layout):
                continue
            box = _box(item, width, height)
            if box:
                x1, y1, x2, y2 = box
                lost[y1:y2, x1:x2] = False
    for item in layout.get("elements", []):
        if item.get("type") == "text":
            if item not in active and is_badge_owned_text(item, layout):
                continue
            box = _box(item, width, height)
            if box:
                x1, y1, x2, y2 = box
                salient[y1:y2, x1:x2] = False
    missing_mask = np.uint8(salient & erased)
    missing_pixels = int(np.count_nonzero(missing_mask))
    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(missing_mask, 8)
    report["largestMissingVisualRegion"] = int(np.max(component_stats[1:, cv2.CC_STAT_AREA])) if component_count > 1 else 0
    report["salientVisualPixels"] = int(np.count_nonzero(salient))
    report["missingVisualPixels"] = missing_pixels
    report["retainedVisualCoverage"] = round(1 - missing_pixels / max(1, int(np.count_nonzero(salient))), 4)
    minimum = max(24, round(width * height * 0.0001))
    color_error = np.max(np.abs(source.astype(np.int16) - preview.astype(np.int16)), axis=2)
    mismatched = salient & (color_error >= 48) & ~np.all(preview >= 253, axis=2)
    report["visualMismatchPixels"] = int(np.count_nonzero(mismatched))
    for x, y, w, h, pixels in _bounded_components(np.uint8(mismatched) * 255, minimum, width, height):
        report["visualMismatchRegions"] += 1
        report["issues"].append({"problem": "visualContentMismatch", "elementId": f"mismatch_{x}_{y}", "bbox": [x, y, x + w, y + h], "pixelArea": pixels})
    missing = _bounded_components(np.uint8(lost) * 255, max(40, round(width * height * 0.00003)), width, height)
    for x, y, w, h, pixels in missing:
        patch = source[y:y + h, x:x + w]
        median = np.median(patch.reshape(-1, 3), axis=0)
        flat = float(np.mean(np.max(np.abs(patch.astype(np.float32) - median), axis=2) <= 9)) >= 0.65
        pale = float(np.max(np.abs(median - page_color))) <= 35
        problem = "missingBackplate" if flat and pale else "missingVisualObject"
        report["missingBackplates" if problem == "missingBackplate" else "missingVisualObjects"] += 1
        report["issues"].append({"problem": problem, "elementId": f"unowned_{x}_{y}", "bbox": [x, y, x + w, y + h], "pixelArea": pixels})
    for item in active:
        if item.get("type") != "image":
            continue
        box = _box(item, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        source_detail = contrast[y1:y2, x1:x2] >= 8
        if np.count_nonzero(source_detail) < max(minimum, source_detail.size * 0.05):
            continue
        blank_fraction = float(np.mean(np.all(preview[y1:y2, x1:x2] >= 253, axis=2)[source_detail]))
        if blank_fraction >= 0.80:
            report["blankVisualOwners"] += 1
            report["issues"].append({"problem": "blankVisualOwner", "elementId": item.get("id"), "bbox": list(box), "blankFraction": round(blank_fraction, 3)})
    background_foreground = np.uint8(np.max(np.abs(background.astype(np.int16) - 255), axis=2) >= 6) * 255
    residual = _bounded_components(background_foreground, minimum, width, height)
    report["backgroundResidualRegions"] = len(residual)
    for x, y, w, h, pixels in residual:
        report["issues"].append({"problem": "assetBakedIntoBackground", "elementId": f"background_{x}_{y}", "bbox": [x, y, x + w, y + h], "pixelArea": pixels})
    if debug_path is not None:
        debug = np.full_like(source, 255)
        colors = {"editable_text": (75, 150, 45), "movable_image": (220, 120, 40), "native_shape": (150, 70, 160)}
        for owner in report["ownerRegions"]:
            x1, y1, x2, y2 = owner["bbox"]
            cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), colors[owner["owner"]], 2)
        for issue in report["issues"]:
            if "bbox" in issue:
                x1, y1, x2, y2 = issue["bbox"]
                cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), (40, 40, 230), 2)
        cv2.putText(debug, "GREEN text  BLUE image  PURPLE shape  WHITE base  RED missing", (8, max(15, height - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (45, 45, 45), 1)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), debug)
    return report


def repair_missing_regions(source_path: Path, layout: dict, issues: list[dict], asset_dir: Path,
                           project_id: str, revision_round: int, asset_prefix: str | None = None) -> list[dict]:
    """Stage only reported unowned regions; leave existing objects untouched."""
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        return []
    height, width = source.shape[:2]
    page_color = np.median(np.concatenate((source[0], source[-1], source[:, 0], source[:, -1])), axis=0).astype(np.int16)
    asset_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for issue in issues:
        if issue.get("problem") not in {"missingBackplate", "missingVisualObject", "blankVisualOwner", "visualContentMismatch"}:
            continue
        box = issue.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [round(float(value)) for value in box]
        x1, y1, x2, y2 = max(0, x1 - 1), max(0, y1 - 1), min(width, x2 + 1), min(height, y2 + 1)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = source[y1:y2, x1:x2].copy()
        existing = [item for item in layout.get("elements", []) if item.get("type") == "text" and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy")) and _box(item, width, height) and _overlaps((x1, y1, x2, y2), _box(item, width, height))]
        group = next((str(item.get("groupId")) for item in existing if item.get("groupId")), None)
        visual_below = [item for item in layout.get("elements", []) if item.get("type") in VISUAL_TYPES and _box(item, width, height) and _overlaps((x1, y1, x2, y2), _box(item, width, height))]
        z_index = max((int(item.get("zIndex") or 0) for item in visual_below), default=8) + 1
        if existing:
            z_index = min(z_index, min(int(item.get("zIndex") or 20) for item in existing) - 1)
        identifier = f"{asset_prefix or f'revision_{revision_round}'}_object_{len(created) + 1:03d}"
        metadata = {"reconstructionStrategySource": "objectization_audit", "layerRole": "container" if issue["problem"] == "missingBackplate" else "residual", "qaIssue": issue["problem"]}
        median = np.median(crop.reshape(-1, 3), axis=0).astype(np.uint8)
        flat = float(np.mean(np.max(np.abs(crop.astype(np.int16) - median.astype(np.int16)), axis=2) <= 9)) >= 0.75
        common = {"id": identifier, "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "zIndex": z_index, "rotation": 0, "groupId": group, "metadata": metadata}
        if issue["problem"] == "missingBackplate" and flat:
            color = f"#{median[2]:02X}{median[1]:02X}{median[0]:02X}"
            common.update({"type": "rectangle", "style": {"fill": color, "stroke": color, "strokeWidth": 0, "opacity": 1}})
            metadata["reconstructionStrategy"] = "native_shape"
        else:
            mask = np.uint8(np.max(np.abs(crop.astype(np.int16) - page_color), axis=2) >= 5) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            for item in existing:
                tx1, ty1, tx2, ty2 = _box(item, width, height)
                left, top, right, bottom = max(0, tx1 - x1), max(0, ty1 - y1), min(x2 - x1, tx2 - x1), min(y2 - y1, ty2 - y1)
                if right > left and bottom > top:
                    text_mask = np.zeros(mask.shape, np.uint8)
                    text_mask[top:bottom, left:right] = 255
                    crop = cv2.inpaint(crop, text_mask, 3, cv2.INPAINT_TELEA)
            if np.count_nonzero(mask) < 24:
                continue
            path = asset_dir / f"{identifier}.png"
            if not cv2.imwrite(str(path), np.dstack((crop, mask))):
                continue
            common.update({"type": "image", "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1}})
            metadata["reconstructionStrategy"] = "cutout_image"
        layout.setdefault("elements", []).append(common)
        created.append(common)
    return created


def recover_initial_missing_regions(source_path: Path, background_path: Path, preview_path: Path,
                                    layout: dict, asset_dir: Path, project_id: str, page_index: int) -> int:
    """Add only missing local visuals whose rendered result measurably improves."""
    from app.services.visual_qa.analyzer import render_preview

    before = audit_objectization(source_path, background_path, preview_path, layout)
    issues = [issue for issue in before["issues"] if issue.get("problem") in {"missingBackplate", "missingVisualObject"}]
    if not issues:
        return 0
    candidate = copy.deepcopy(layout)
    created = repair_missing_regions(source_path, candidate, issues, asset_dir, project_id, 0,
                                     asset_prefix=f"initial_page_{page_index}")
    if not created:
        return 0
    candidate_preview = preview_path.with_name(f"{preview_path.stem}_object_recovery.png")
    accepted = False
    try:
        render_preview(background_path, candidate, candidate_preview)
        after = audit_objectization(source_path, background_path, candidate_preview, candidate)
        valid = not any(issue.get("problem") == "objectizationAuditUnavailable" for issue in after["issues"])
        improved = (valid and after["missingVisualPixels"] < before["missingVisualPixels"]
                    and after["visualMismatchPixels"] <= before["visualMismatchPixels"] + 24)
        if improved:
            candidate_preview.replace(preview_path)
            layout["elements"] = candidate["elements"]
            accepted = True
            return len(created)
        return 0
    finally:
        candidate_preview.unlink(missing_ok=True)
        if not accepted:
            for item in created:
                if item.get("type") == "image":
                    (asset_dir / Path(str(item.get("src") or "")).name).unlink(missing_ok=True)


def _overlaps(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(left[1], right[1])


def _visual_mask(item: dict, box: tuple[int, int, int, int], source_path: Path) -> np.ndarray:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    kind = item.get("type")
    if kind == "image":
        src = str(item.get("src") or "")
        path = Path(src)
        if not path.is_file():
            path = source_path.parent / "assets" / Path(src).name
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
        if image is not None and image.ndim == 3 and image.shape[2] == 4:
            alpha = cv2.resize(image[:, :, 3], (width, height), interpolation=cv2.INTER_LINEAR)
            return np.uint8(alpha > 32) * 255
        return np.full((height, width), 255, np.uint8) if image is not None else np.zeros((height, width), np.uint8)
    if kind == "ellipse":
        mask = np.zeros((height, width), np.uint8)
        cv2.ellipse(mask, (width // 2, height // 2), (max(1, width // 2), max(1, height // 2)), 0, 0, 360, 255, -1)
        return mask
    if kind in {"line", "arrow"}:
        mask = np.zeros((height, width), np.uint8)
        cv2.line(mask, (0, height // 2), (width - 1, height // 2), 255, max(2, int(float((item.get("style") or {}).get("strokeWidth") or 2)) + 2))
        return mask
    return np.full((height, width), 255, np.uint8)


def _bounded_components(mask: np.ndarray, minimum: int, width: int, height: int) -> list[tuple[int, int, int, int, int]]:
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = []
    for index in range(1, count):
        x, y, w, h, pixels = [int(value) for value in stats[index]]
        if pixels < minimum or ((w < 5 or h < 4) and not is_meaningful_stroke(w, h, pixels, width, height)) or w * h > width * height * 0.62:
            continue
        result.append((x, y, w, h, pixels))
    return result


def _box(item: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    x, y = float(item.get("x") or 0), float(item.get("y") or 0)
    x1, y1 = max(0, min(width, round(x))), max(0, min(height, round(y)))
    x2, y2 = max(0, min(width, round(x + float(item.get("width") or 0)))), max(0, min(height, round(y + float(item.get("height") or 0))))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None
