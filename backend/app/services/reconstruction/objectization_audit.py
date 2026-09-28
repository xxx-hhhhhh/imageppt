"""Pixel and ownership audit for white, objectized slide reconstructions."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


VISUAL_TYPES = {"image", "rectangle", "roundedRectangle", "ellipse", "line", "arrow"}


def audit_objectization(source_path: Path, background_path: Path, preview_path: Path, layout: dict,
                        debug_path: Path | None = None) -> dict:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    preview = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
    report = {"whiteBackground": False, "missingBackplates": 0, "missingVisualObjects": 0, "blankVisualOwners": 0,
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
            owner_mask[y1:y2, x1:x2] = 255
    page_color = np.median(np.concatenate((source[0], source[-1], source[:, 0], source[:, -1])), axis=0).astype(np.int16)
    contrast = np.max(np.abs(source.astype(np.int16) - page_color), axis=2)
    lost = (contrast >= 5) & np.all(preview >= 253, axis=2) & (owner_mask == 0)
    # Editable glyphs need not land on precisely the same raster pixels.
    for item in layout.get("elements", []):
        if item.get("type") == "text":
            box = _box(item, width, height)
            if box:
                x1, y1, x2, y2 = box
                lost[y1:y2, x1:x2] = False
    minimum = max(24, round(width * height * 0.0001))
    missing = _bounded_components(np.uint8(lost) * 255, minimum, width, height)
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
                           project_id: str, revision_round: int) -> list[dict]:
    """Stage only reported unowned regions; leave existing objects untouched."""
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        return []
    height, width = source.shape[:2]
    page_color = np.median(np.concatenate((source[0], source[-1], source[:, 0], source[:, -1])), axis=0).astype(np.int16)
    asset_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for issue in issues:
        if issue.get("problem") not in {"missingBackplate", "missingVisualObject", "blankVisualOwner"}:
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
        z_index = min((int(item.get("zIndex") or 20) for item in existing), default=10) - 1
        identifier = f"revision_{revision_round}_object_{len(created) + 1:03d}"
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


def _overlaps(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(left[1], right[1])


def _bounded_components(mask: np.ndarray, minimum: int, width: int, height: int) -> list[tuple[int, int, int, int, int]]:
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = []
    for index in range(1, count):
        x, y, w, h, pixels = [int(value) for value in stats[index]]
        if pixels < minimum or w < 5 or h < 4 or w * h > width * height * 0.62:
            continue
        result.append((x, y, w, h, pixels))
    return result[:50]


def _box(item: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    x, y = float(item.get("x") or 0), float(item.get("y") or 0)
    x1, y1 = max(0, min(width, round(x))), max(0, min(height, round(y)))
    x2, y2 = max(0, min(width, round(x + float(item.get("width") or 0)))), max(0, min(height, round(y + float(item.get("height") or 0))))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None
