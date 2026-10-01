"""Pixel and ownership audit for white, objectized slide reconstructions."""

from __future__ import annotations

import copy
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.services.reconstruction.residual_objects import is_meaningful_stroke, visual_candidate_mask
from app.services.reconstruction.visual_asset_ownership import count_duplicate_planned_visual_pixels
from app.services.reconstruction.white_objectization import detect_flat_page_surface


VISUAL_TYPES = {"image", "rectangle", "roundedRectangle", "ellipse", "line", "arrow"}


def audit_objectization(source_path: Path, background_path: Path, preview_path: Path, layout: dict,
                        debug_path: Path | None = None) -> dict:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    preview = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
    report = {"whiteBackground": False, "missingBackplates": 0, "missingVisualObjects": 0, "blankVisualOwners": 0,
              "visualMismatchRegions": 0, "visualMismatchPixels": 0, "salientVisualPixels": 0,
              "missingVisualPixels": 0, "coveredMissingVisualPixels": 0, "largestMissingVisualRegion": 0, "retainedVisualCoverage": 1.0,
              "paleAssetGapPixels": 0, "duplicatePlannedVisualPixels": 0, "pageSurfaceMismatchPixels": 0,
              "backgroundResidualRegions": 0, "ownerRegions": [], "issues": []}
    if source is None or background is None or preview is None or source.shape != background.shape or source.shape != preview.shape:
        report["issues"].append({"problem": "objectizationAuditUnavailable"})
        return report
    height, width = source.shape[:2]
    page_surface = detect_flat_page_surface(source)
    if page_surface is not None:
        source_difference = np.max(np.abs(source.astype(np.int16) - page_surface.astype(np.int16)), axis=2)
        preview_difference = np.max(np.abs(preview.astype(np.int16) - page_surface.astype(np.int16)), axis=2)
        report["pageSurfaceMismatchPixels"] = int(np.count_nonzero((source_difference <= 10) & (preview_difference >= 20)))
        if report["pageSurfaceMismatchPixels"] > max(100, round(width * height * 0.02)):
            report["issues"].append({"problem": "pageSurfaceLost", "bbox": [0, 0, width, height],
                                     "pixelArea": report["pageSurfaceMismatchPixels"]})
    report["duplicatePlannedVisualPixels"] = count_duplicate_planned_visual_pixels(layout, source_path.parent / "assets")
    if report["duplicatePlannedVisualPixels"] >= max(100, round(width * height * 0.0001)):
        report["issues"].append({"problem": "duplicatePlannedVisualOwnership",
                                 "pixelArea": report["duplicatePlannedVisualPixels"]})
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
    local_asset_regions = np.zeros((height, width), np.bool_)
    for item in active:
        if item.get("type") != "image":
            continue
        box = _box(item, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) <= width * height * 0.55:
            local_asset_regions[y1:y2, x1:x2] = True
    report["paleAssetGapPixels"] = int(np.count_nonzero(_pale_gap_pixels(source, preview) & local_asset_regions))
    page_color = np.median(np.concatenate((source[0], source[-1], source[:, 0], source[:, -1])), axis=0).astype(np.int16)
    salient = visual_candidate_mask(source)
    contrast = np.max(np.abs(source.astype(np.int16) - page_color), axis=2)
    # White details inside a badge or card can be intentional. Only count pixels
    # that actually changed to white, rather than every white source pixel.
    erased = np.all(preview >= 253, axis=2) & (np.max(np.abs(source.astype(np.int16) - preview.astype(np.int16)), axis=2) >= 2)
    lost = salient & erased
    # OCR boxes often include a badge, colored plate, or chart mark beside the
    # letters. Keep distinct source color that has no nearby rendered owner.
    source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    preview_hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
    source_color = (source_hsv[:, :, 1] >= 55) & (source_hsv[:, :, 2] >= 35)
    preview_color = (preview_hsv[:, :, 1] >= 40) & (preview_hsv[:, :, 2] >= 35)
    source_channel = np.argmax(source, axis=2)
    preview_channel = np.argmax(preview, axis=2)
    nearby_same_color = np.zeros((height, width), np.bool_)
    for channel in range(3):
        rendered = np.uint8(preview_color & (preview_channel == channel)) * 255
        nearby = cv2.dilate(rendered, np.ones((13, 13), np.uint8)) != 0
        nearby_same_color |= (source_channel == channel) & nearby
    lost_inside_text = source_color & erased & ~nearby_same_color
    text_coverage_area = np.zeros((height, width), np.bool_)
    editable_glyph_color = np.zeros((height, width), np.bool_)
    # Editable glyphs need not land on precisely the same raster pixels.
    # Use the observed OCR bounds, not the often enlarged editor textbox.
    # A suppressed false OCR detection must never conceal a lost visual.
    for item in layout.get("elements", []):
        if item.get("type") != "text":
            continue
        metadata = item.get("metadata") or {}
        if item not in active and (metadata.get("ownedBy") or metadata.get("duplicateSuppressed")):
            continue
        box = _text_source_box(item, width, height, padding=2)
        if box:
            x1, y1, x2, y2 = box
            text_coverage_area[y1:y2, x1:x2] = True
            color = str((item.get("style") or {}).get("color") or "").lstrip("#")
            if len(color) == 6:
                try:
                    bgr = np.frombuffer(bytes.fromhex(color)[::-1], dtype=np.uint8).astype(np.int16)
                    region = source[y1:y2, x1:x2].astype(np.int16)
                    editable_glyph_color[y1:y2, x1:x2] |= np.max(np.abs(region - bgr), axis=2) <= 90
                except ValueError:
                    pass
            lost[y1:y2, x1:x2] = False
            salient[y1:y2, x1:x2] = False
    uncovered_colored_visual = lost_inside_text & text_coverage_area & ~editable_glyph_color
    count, labels, component_stats, _ = cv2.connectedComponentsWithStats(np.uint8(uncovered_colored_visual), 8)
    accepted_colored_visual = np.zeros_like(uncovered_colored_visual)
    minimum_colored_area = max(150, round(width * height * 0.00008))
    for label in range(1, count):
        _, _, component_width, component_height, pixels = [int(value) for value in component_stats[label]]
        area = max(1, component_width * component_height)
        if pixels >= 400 or (pixels >= minimum_colored_area and min(component_width, component_height) >= 12
                             and pixels / area >= 0.60):
            accepted_colored_visual[labels == label] = True
    lost |= accepted_colored_visual
    salient |= accepted_colored_visual
    missing_mask = np.uint8(salient & erased)
    missing_pixels = int(np.count_nonzero(missing_mask))
    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(missing_mask, 8)
    report["largestMissingVisualRegion"] = int(np.max(component_stats[1:, cv2.CC_STAT_AREA])) if component_count > 1 else 0
    report["salientVisualPixels"] = int(np.count_nonzero(salient))
    report["missingVisualPixels"] = missing_pixels
    report["coveredMissingVisualPixels"] = int(np.count_nonzero(missing_mask & (owner_mask != 0)))
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
        # Judge the missing pixels, not their often-white bounding rectangle.
        # A thin chart line inside a large white box is still a visual object.
        colors = patch[lost[y:y + h, x:x + w]]
        median = np.median(colors, axis=0)
        flat = float(np.mean(np.max(np.abs(colors.astype(np.float32) - median), axis=1) <= 9)) >= 0.65
        pale = float(np.max(np.abs(median - page_color))) <= 35
        dense = len(colors) / max(1, w * h) >= 0.70
        problem = "missingBackplate" if flat and pale and dense else "missingVisualObject"
        report["missingBackplates" if problem == "missingBackplate" else "missingVisualObjects"] += 1
        report["issues"].append({"problem": problem, "elementId": f"unowned_{x}_{y}", "bbox": [x, y, x + w, y + h], "pixelArea": pixels})
    for item in active:
        if item.get("type") != "image":
            continue
        box = _box(item, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        # A transparent crop may occupy only a small part of its bounding box.
        # Judge the pixels it actually owns, not unrelated white space nearby.
        owner_pixels = _visual_mask(item, box, source_path)
        source_detail = contrast[y1:y2, x1:x2] >= 8
        if np.any(owner_pixels):
            source_detail &= owner_pixels != 0
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
                           project_id: str, revision_round: int, asset_prefix: str | None = None,
                           preview_path: Path | None = None) -> list[dict]:
    """Stage only reported unowned regions; leave existing objects untouched."""
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        return []
    preview = cv2.imread(str(preview_path), cv2.IMREAD_COLOR) if preview_path else None
    if preview is not None and preview.shape != source.shape:
        preview = None
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
            source_detail = np.max(np.abs(crop.astype(np.int16) - page_color), axis=2) >= 2
            if preview is not None:
                shown = preview[y1:y2, x1:x2]
                difference = np.max(np.abs(crop.astype(np.int16) - shown.astype(np.int16)), axis=2)
                if issue["problem"] == "visualContentMismatch":
                    missing_pixels = difference >= 48
                else:
                    missing_pixels = np.all(shown >= 253, axis=2) & (difference >= 2)
                mask = np.uint8(source_detail & missing_pixels) * 255
            else:
                mask = np.uint8(source_detail) * 255
            for item in existing:
                text_box = _text_source_box(item, width, height)
                if text_box is None:
                    continue
                tx1, ty1, tx2, ty2 = text_box
                left, top, right, bottom = max(0, tx1 - x1 - 1), max(0, ty1 - y1 - 1), min(x2 - x1, tx2 - x1 + 1), min(y2 - y1, ty2 - y1 + 1)
                if right > left and bottom > top:
                    if issue["problem"] != "missingVisualObject":
                        mask[top:bottom, left:right] = 0
                        continue
                    patch = crop[top:bottom, left:right]
                    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                    distinct_visual = (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 35)
                    text_color = str((item.get("style") or {}).get("color") or "").lstrip("#")
                    if len(text_color) == 6:
                        try:
                            bgr = np.frombuffer(bytes.fromhex(text_color)[::-1], dtype=np.uint8).astype(np.int16)
                            distinct_visual &= np.max(np.abs(patch.astype(np.int16) - bgr), axis=2) > 90
                        except ValueError:
                            pass
                    mask[top:bottom, left:right][~distinct_visual] = 0
            if np.count_nonzero(mask) < 24:
                continue
            mx, my, mw, mh = cv2.boundingRect(mask)
            crop = crop[my:my + mh, mx:mx + mw]
            mask = mask[my:my + mh, mx:mx + mw]
            common.update({"x": x1 + mx, "y": y1 + my, "width": mw, "height": mh})
            path = asset_dir / f"{identifier}.png"
            if not cv2.imwrite(str(path), np.dstack((crop, mask))):
                continue
            common.update({"type": "image", "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1}})
            metadata.update({"reconstructionStrategy": "cutout_image", "sourceMaskPixels": int(np.count_nonzero(mask))})
        layout.setdefault("elements", []).append(common)
        created.append(common)
    return created


def recover_initial_missing_regions(source_path: Path, background_path: Path, preview_path: Path,
                                    layout: dict, asset_dir: Path, project_id: str, page_index: int) -> int:
    """Add only missing local visuals whose rendered result measurably improves."""
    from app.services.visual_qa.analyzer import render_preview

    before = audit_objectization(source_path, background_path, preview_path, layout)
    issues = [issue for issue in before["issues"] if issue.get("problem") in {"missingBackplate", "missingVisualObject"}]
    candidate = copy.deepcopy(layout)
    created = repair_missing_regions(source_path, candidate, issues, asset_dir, project_id, 0,
                                     asset_prefix=f"initial_page_{page_index}", preview_path=preview_path)
    created.extend(_recover_pale_asset_gaps(source_path, preview_path, candidate, asset_dir, project_id, page_index))
    if not created:
        return 0
    from app.services.reconstruction.white_objectization import layer_objectized_elements
    layer_objectized_elements(candidate.get("elements", []))
    candidate_preview = preview_path.with_name(f"{preview_path.stem}_object_recovery.png")
    accepted = False
    try:
        render_preview(background_path, candidate, candidate_preview)
        after = audit_objectization(source_path, background_path, candidate_preview, candidate)
        valid = not any(issue.get("problem") == "objectizationAuditUnavailable" for issue in after["issues"])
        improved = (valid and after["missingVisualPixels"] < before["missingVisualPixels"]
                    and after["visualMismatchPixels"] <= before["visualMismatchPixels"] + 24)
        if not improved and any((item.get("metadata") or {}).get("qaIssue") in {"paleAssetGap", "paleTextSupportGap"} for item in created):
            source = cv2.imread(str(source_path))
            previous = cv2.imread(str(preview_path))
            current = cv2.imread(str(candidate_preview))
            if source is not None and previous is not None and current is not None:
                pale = _pale_gap_pixels(source, previous)
                before_gap = int(np.count_nonzero(pale))
                after_gap = int(np.count_nonzero(_pale_gap_pixels(source, current) & pale))
                improved = (valid and before_gap >= 200 and after_gap <= before_gap * 0.8
                            and after["missingVisualPixels"] <= before["missingVisualPixels"] + 24
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


def _pale_gap_pixels(source: np.ndarray, preview: np.ndarray) -> np.ndarray:
    """Low contrast support that has become page white after reconstruction."""
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    difference = np.max(np.abs(source.astype(np.int16) - preview.astype(np.int16)), axis=2)
    return ((hsv[:, :, 2] >= 225) & (hsv[:, :, 1] <= 32)
            & np.all(preview >= 253, axis=2) & (difference >= 3))


def _recover_pale_asset_gaps(source_path: Path, preview_path: Path, layout: dict,
                             asset_dir: Path, project_id: str, page_index: int) -> list[dict]:
    """Restore bounded pale support omitted by an otherwise valid movable asset."""
    source = cv2.imread(str(source_path))
    preview = cv2.imread(str(preview_path))
    if source is None or preview is None or source.shape != preview.shape:
        return []
    height, width = source.shape[:2]
    missing = _pale_gap_pixels(source, preview)
    if not np.any(missing):
        return []
    text_support_gaps = missing.copy()
    for item in layout.get("elements", []):
        if item.get("type") != "text" or any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        box = _box(item, width, height)
        if box:
            x1, y1, x2, y2 = box
            missing[y1:y2, x1:x2] = False
    created: list[dict] = []
    for owner in list(layout.get("elements", [])):
        if owner.get("type") != "image" or any((owner.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        box = _box(owner, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area > width * height * 0.55:
            continue
        owner_alpha = _visual_mask(owner, box, source_path)
        gap = np.uint8(missing[y1:y2, x1:x2] & (owner_alpha == 0)) * 255
        stable_gap = cv2.morphologyEx(gap, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        if np.count_nonzero(stable_gap) < max(120, round(area * 0.004)):
            continue
        # A residual crop and its missing pale support are one visual module.
        # Fold the support into a staged copy of that same transparent asset so
        # moving the ribbon/card does not leave a detached pale shadow behind.
        if (owner.get("metadata") or {}).get("reconstructionStrategySource") == "residual_detection":
            prior_path = asset_dir / Path(str(owner.get("src") or "")).name
            prior = cv2.imread(str(prior_path), cv2.IMREAD_UNCHANGED) if prior_path.is_file() else None
            if prior is not None and prior.ndim == 3 and prior.shape[2] == 4 and prior.shape[:2] == gap.shape:
                merged = prior.copy()
                selected = (gap != 0) & (merged[:, :, 3] <= 32)
                merged[selected, :3] = source[y1:y2, x1:x2][selected]
                merged[selected, 3] = 255
                if np.count_nonzero(selected) >= max(120, round(area * 0.004)):
                    identifier = f"initial_page_{page_index}_pale_merged_{uuid4().hex[:10]}"
                    path = asset_dir / f"{identifier}.png"
                    if cv2.imwrite(str(path), merged):
                        owner["src"] = f"/media/assets/{project_id}/{path.name}"
                        owner.setdefault("metadata", {}).update({"paleGapMergedPixels": int(np.count_nonzero(selected)),
                                                                 "qaIssue": "paleAssetGap"})
                        created.append(owner)
                        missing[y1:y2, x1:x2][selected] = False
                        continue
        identifier = f"initial_page_{page_index}_pale_gap_{len(created) + 1:03d}"
        path = asset_dir / f"{identifier}.png"
        if not cv2.imwrite(str(path), np.dstack((source[y1:y2, x1:x2], gap))):
            continue
        item = {"id": identifier, "type": "image", "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
                "rotation": 0, "zIndex": int(owner.get("zIndex") or 0) + 1, "groupId": owner.get("groupId"),
                "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1},
                "metadata": {"reconstructionStrategy": "cutout_image", "reconstructionStrategySource": "objectization_audit",
                             "layerRole": "container_detail", "qaIssue": "paleAssetGap", "parentId": owner.get("id")}}
        layout.setdefault("elements", []).append(item)
        created.append(item)
        missing[y1:y2, x1:x2][gap != 0] = False
    created.extend(_recover_text_support_gaps(source, text_support_gaps, layout, asset_dir, project_id, page_index))
    return created


def _recover_text_support_gaps(source: np.ndarray, missing: np.ndarray, layout: dict,
                               asset_dir: Path, project_id: str, page_index: int) -> list[dict]:
    """Restore pale support beneath editable text without duplicating source glyphs."""
    height, width = source.shape[:2]
    images = [item for item in layout.get("elements", []) if item.get("type") == "image" and
              not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    created: list[dict] = []
    for text in layout.get("elements", []):
        if text.get("type") != "text" or any((text.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        raw = (text.get("metadata") or {}).get("rawOCRBBox")
        box = _box(text, width, height) if not isinstance(raw, list) or len(raw) != 4 else _box(
            {"x": raw[0], "y": raw[1], "width": float(raw[2]) - float(raw[0]), "height": float(raw[3]) - float(raw[1])}, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area < 100 or area > width * height * 0.15:
            continue
        parents = [item for item in images if _box(item, width, height) and _overlaps(box, _box(item, width, height))]
        if not parents:
            continue
        crop = source[y1:y2, x1:x2].copy()
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pale = (hsv[:, :, 2] >= 225) & (hsv[:, :, 1] <= 32)
        gap = missing[y1:y2, x1:x2]
        if np.count_nonzero(gap) < max(80, round(area * 0.12)) or np.mean(pale) < 0.40:
            continue
        support = np.median(crop[pale], axis=0).astype(np.int16)
        ink = np.uint8(np.max(np.abs(crop.astype(np.int16) - support), axis=2) >= 28) * 255
        if np.mean(ink != 0) > 0.38:
            continue
        ink = cv2.dilate(ink, np.ones((3, 3), np.uint8), iterations=1)
        cleaned = cv2.inpaint(crop, ink, 3, cv2.INPAINT_TELEA) if np.any(ink) else crop
        alpha = np.uint8(pale | (ink != 0)) * 255
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        identifier = f"initial_page_{page_index}_text_support_{len(created) + 1:03d}"
        path = asset_dir / f"{identifier}.png"
        if not cv2.imwrite(str(path), np.dstack((cleaned, alpha))):
            continue
        parent = max(parents, key=lambda item: (_box(item, width, height)[2] - _box(item, width, height)[0]) *
                     (_box(item, width, height)[3] - _box(item, width, height)[1]))
        item = {"id": identifier, "type": "image", "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
                "rotation": 0, "zIndex": int(text.get("zIndex") or 20) - 1, "groupId": text.get("groupId") or parent.get("groupId"),
                "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1},
                "metadata": {"reconstructionStrategy": "cutout_image", "reconstructionStrategySource": "objectization_audit",
                             "layerRole": "container_detail", "qaIssue": "paleTextSupportGap", "parentId": parent.get("id"),
                             "editableTextIds": [text.get("id")], "textCleaned": True}}
        created.append(item)
    layout.setdefault("elements", []).extend(created)
    return created


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


def _text_source_box(item: dict, width: int, height: int, *, padding: int = 0) -> tuple[int, int, int, int] | None:
    raw = (item.get("metadata") or {}).get("rawOCRBBox")
    if isinstance(raw, list) and len(raw) == 4:
        try:
            x1, y1, x2, y2 = (float(value) for value in raw)
        except (TypeError, ValueError):
            return _box(item, width, height)
        box = _box({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}, width, height)
    else:
        box = _box(item, width, height)
    if box is None or not padding:
        return box
    return (max(0, box[0] - padding), max(0, box[1] - padding),
            min(width, box[2] + padding), min(height, box[3] + padding))
