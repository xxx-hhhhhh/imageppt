"""Pixel and ownership audit for white, objectized slide reconstructions."""

from __future__ import annotations

import copy
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.services.reconstruction.residual_objects import _small_solid_decoration, is_meaningful_stroke, visual_candidate_mask
from app.services.reconstruction.visual_asset_ownership import count_duplicate_planned_visual_pixels
from app.services.reconstruction.white_objectization import _shape_source_supported, detect_flat_page_surface


VISUAL_TYPES = {"image", "rectangle", "roundedRectangle", "ellipse", "line", "arrow"}


def audit_objectization(source_path: Path, background_path: Path, preview_path: Path, layout: dict,
                        debug_path: Path | None = None) -> dict:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    preview = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
    report = {"whiteBackground": False, "missingBackplates": 0, "missingVisualObjects": 0, "blankVisualOwners": 0,
              "visualMismatchRegions": 0, "visualMismatchPixels": 0, "salientVisualPixels": 0,
              "falseVisualAdditionPixels": 0, "fadedPaleSupportPixels": 0, "washedColoredAssetPixels": 0,
              "missingVisualPixels": 0, "coveredMissingVisualPixels": 0, "largestMissingVisualRegion": 0, "retainedVisualCoverage": 1.0,
              "paleAssetGapPixels": 0, "unownedPaleSupportPixels": 0, "textBoundPaleGapPixels": 0,
              "duplicatePlannedVisualPixels": 0, "pageSurfaceMismatchPixels": 0,
              "monolithicPageImageCount": 0,
              "backgroundResidualRegions": 0, "ownerRegions": [], "issues": []}
    if source is None or background is None or preview is None or source.shape != background.shape or source.shape != preview.shape:
        report["issues"].append({"problem": "objectizationAuditUnavailable"})
        return report
    height, width = source.shape[:2]
    page_surface = detect_flat_page_surface(source)
    if page_surface is not None:
        if float(np.mean(np.all(background >= 250, axis=2))) < 0.995:
            report["issues"].append({"problem": "pageSurfaceBakedIntoBackground", "elementId": "page_surface",
                                     "bbox": [0, 0, width, height]})
        source_difference = np.max(np.abs(source.astype(np.int16) - page_surface.astype(np.int16)), axis=2)
        preview_difference = np.max(np.abs(preview.astype(np.int16) - page_surface.astype(np.int16)), axis=2)
        surface_mismatch = (source_difference <= 10) & (preview_difference >= 20)
        for item in layout.get("elements", []):
            if item.get("type") != "text" or any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
                continue
            text_box = _text_source_box(item, width, height, padding=2)
            if text_box is not None:
                tx1, ty1, tx2, ty2 = text_box
                surface_mismatch[ty1:ty2, tx1:tx2] = False
        report["pageSurfaceMismatchPixels"] = int(np.count_nonzero(surface_mismatch))
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
    residual_owner_mask = np.zeros((height, width), np.uint8)
    for item in active:
        box = _box(item, width, height)
        if box is None:
            continue
        if (item.get("type") in {"rectangle", "roundedRectangle"}
                and (item.get("style") or {}).get("fill")
                and not (item.get("metadata") or {}).get("pageSurface")
                and not _shape_source_supported(source, box, item.get("style") or {})):
            report["issues"].append({"problem": "unsupportedNativeShape", "elementId": item.get("id"),
                                     "bbox": list(box)})
        role = "editable_text" if item.get("type") == "text" else "movable_image" if item.get("type") == "image" else "native_shape"
        report["ownerRegions"].append({"elementId": item.get("id"), "owner": role, "bbox": list(box)})
        if item.get("type") in VISUAL_TYPES:
            x1, y1, x2, y2 = box
            visible = _visual_mask(item, box, source_path)
            owner_mask[y1:y2, x1:x2] = cv2.max(owner_mask[y1:y2, x1:x2], visible)
            if item.get("type") == "image" and (item.get("metadata") or {}).get("layerRole") == "residual":
                residual_owner_mask[y1:y2, x1:x2] = cv2.max(residual_owner_mask[y1:y2, x1:x2], visible)
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
    pale_missing = _pale_gap_pixels(source, preview)
    # A slightly off-white page surface is not a missing local plate. It is
    # handled by the page-surface owner and may show a one-pixel crop seam.
    border_color = np.median(np.concatenate((source[0], source[-1], source[:, 0], source[:, -1])), axis=0).astype(np.int16)
    page_distance = np.max(np.abs(source.astype(np.int16) - border_color), axis=2)
    if float(np.mean(border_color)) < 253 and float(np.mean(page_distance <= 1)) >= 0.65:
        pale_missing[page_distance <= 1] = False
    text_bound_pale = np.zeros((height, width), np.bool_)
    # A page may be only two or three RGB levels below white. That diffuse
    # surface is not a missing local support beneath OCR text; counting it
    # would make revision rollback react to harmless page texture changes.
    visible_pale_support = np.min(source, axis=2) <= 250
    for item in active:
        if item.get("type") != "text":
            continue
        box = _text_source_box(item, width, height, padding=2)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        local_gap = pale_missing[y1:y2, x1:x2] & visible_pale_support[y1:y2, x1:x2]
        text_bound_pale[y1:y2, x1:x2] |= _coherent_text_support(
            source[y1:y2, x1:x2], local_gap)
        pale_missing[y1:y2, x1:x2] = False
    report["textBoundPaleGapPixels"] = int(np.count_nonzero(text_bound_pale))
    unowned_pale = np.zeros((height, width), np.bool_)
    for item in active:
        if item.get("type") != "image":
            continue
        box = _box(item, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area > width * height * 0.55:
            continue
        alpha = _visual_mask(item, box, source_path)
        gap = np.uint8(pale_missing[y1:y2, x1:x2] & (alpha <= 32)) * 255
        stable = cv2.morphologyEx(gap, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        pixels = int(np.count_nonzero(stable))
        # Judge a local support relative to its owner. A fixed page-scale floor
        # discards complete backplates of small labels and corner decorations.
        if pixels < max(16, round(area * 0.01)):
            continue
        unowned_pale[y1:y2, x1:x2] |= stable != 0
        report["issues"].append({"problem": "paleAssetGap", "elementId": item.get("id"),
                                 "bbox": list(box), "pixelArea": pixels})
    local_faded = _faded_local_support_mask(source, preview)
    local_faded[(owner_mask > 32) | ~local_asset_regions] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(local_faded, 8)
    faded_parents: dict[str, dict] = {}
    for index in range(1, count):
        x, y, w, h, pixels = [int(value) for value in stats[index]]
        if pixels < max(120, round(width * height * 0.00006)) or w < 8 or h < 5 or w * h > width * height * 0.12:
            continue
        box = (x, y, x + w, y + h)
        eligible = [item for item in active if item.get("type") == "image"
                    and (parent_box := _box(item, width, height)) is not None
                    and _box_area(parent_box) <= width * height * 0.55
                    and _overlap_of_first(box, parent_box) >= 0.75]
        if not eligible:
            continue
        parent = min(eligible, key=lambda item: _box_area(_box(item, width, height)))
        key = str(parent.get("id"))
        issue = faded_parents.setdefault(key, {"problem": "paleAssetGap", "elementId": key,
                                              "bbox": list(_box(parent, width, height)), "pixelArea": 0,
                                              "reason": "unowned_local_pale_support"})
        issue["pixelArea"] += pixels
        unowned_pale[labels == index] = True
    report["issues"].extend(faded_parents.values())
    report["unownedPaleSupportPixels"] = int(np.count_nonzero(unowned_pale))
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
    for index in range(1, component_count):
        x, y, w, h, pixels = [int(value) for value in component_stats[index]]
        if pixels >= width * height * 0.62:
            report["missingVisualObjects"] += 1
            report["issues"].append({"problem": "largeVisualLoss", "elementId": f"large_unowned_{x}_{y}",
                                     "bbox": [x, y, x + w, y + h], "pixelArea": pixels})
    report["salientVisualPixels"] = int(np.count_nonzero(salient))
    report["missingVisualPixels"] = missing_pixels
    report["coveredMissingVisualPixels"] = int(np.count_nonzero(missing_mask & (owner_mask != 0)))
    report["retainedVisualCoverage"] = round(1 - missing_pixels / max(1, int(np.count_nonzero(salient))), 4)
    minimum = max(24, round(width * height * 0.0001))
    color_error = np.max(np.abs(source.astype(np.int16) - preview.astype(np.int16)), axis=2)
    mismatched = salient & (color_error >= 48) & ~np.all(preview >= 253, axis=2)
    report["visualMismatchPixels"] = int(np.count_nonzero(mismatched))
    mismatch_count, _, mismatch_stats, _ = cv2.connectedComponentsWithStats(np.uint8(mismatched), 8)
    for index in range(1, mismatch_count):
        x, y, w, h, pixels = [int(value) for value in mismatch_stats[index]]
        if pixels >= width * height * 0.62:
            report["visualMismatchRegions"] += 1
            report["issues"].append({"problem": "visualContentMismatch", "elementId": f"mismatch_{x}_{y}",
                                     "bbox": [x, y, x + w, y + h], "pixelArea": pixels})
    for x, y, w, h, pixels in _bounded_components(np.uint8(mismatched) * 255, minimum, width, height):
        report["visualMismatchRegions"] += 1
        report["issues"].append({"problem": "visualContentMismatch", "elementId": f"mismatch_{x}_{y}", "bbox": [x, y, x + w, y + h], "pixelArea": pixels})
    # A dark residual asset can spread into originally pale gaps during text
    # inpainting. These source pixels are not salient, so the ordinary
    # source-detail mismatch check above cannot see the added visual content.
    false_addition = ((source_hsv[:, :, 2] >= 240) & (source_hsv[:, :, 1] <= 30)
                      & (preview_hsv[:, :, 2] <= 150) & (residual_owner_mask > 32))
    false_addition_pixels = 0
    for x, y, w, h, pixels in _bounded_components(np.uint8(false_addition) * 255,
                                                   max(300, round(width * height * 0.00018)), width, height):
        # Shifted editable glyphs are usually narrow and separate. A broad,
        # dense added band is evidence of asset bleed rather than font drift.
        if w < max(36, round(width * 0.025)) or h < 5 or w / h < 2.5 or pixels / max(1, w * h) < 0.35:
            continue
        false_addition_pixels += pixels
        report["visualMismatchRegions"] += 1
        report["visualMismatchPixels"] += pixels
        report["issues"].append({"problem": "visualContentMismatch", "elementId": f"false_addition_{x}_{y}",
                                 "bbox": [x, y, x + w, y + h], "pixelArea": pixels,
                                 "reason": "dark_residual_over_pale_source"})
    report["falseVisualAdditionPixels"] = false_addition_pixels
    # A tinted pale plate can be hidden by an almost-white residual crop. It
    # is still a lost movable support even though the preview is not pure white.
    faded_support = ((source_hsv[:, :, 2] >= 225) & (source_hsv[:, :, 1] >= 12)
                     & (source_hsv[:, :, 1] <= 55) & (preview_hsv[:, :, 2] >= 248)
                     & (preview_hsv[:, :, 1] <= 10) & (color_error >= 15)
                     & (residual_owner_mask > 32))
    faded_pixels = 0
    for x, y, w, h, pixels in _bounded_components(np.uint8(faded_support) * 255,
                                                   max(120, round(width * height * 0.00008)), width, height):
        if w < 20 or h < 5 or pixels / max(1, w * h) < 0.35:
            continue
        faded_pixels += pixels
        report["visualMismatchRegions"] += 1
        report["visualMismatchPixels"] += pixels
        report["issues"].append({"problem": "visualContentMismatch", "elementId": f"faded_plate_{x}_{y}",
                                 "bbox": [x, y, x + w, y + h], "pixelArea": pixels,
                                 "reason": "pale_support_overwritten"})
    report["fadedPaleSupportPixels"] = faded_pixels
    # A text-cleaned residual can accidentally replace a colored ribbon with
    # a broad white patch. Small glyphs are excluded: editable text may move
    # slightly without meaning that the underlying visual asset was lost.
    washed_color = ((source_hsv[:, :, 1] >= 60) & (source_hsv[:, :, 2] >= 35)
                    & (preview_hsv[:, :, 1] <= 30) & (preview_hsv[:, :, 2] >= 230)
                    & (residual_owner_mask > 32))
    washed_pixels = 0
    for x, y, w, h, pixels in _bounded_components(np.uint8(washed_color) * 255,
                                                   max(2000, round(width * height * 0.001)), width, height):
        if w < max(80, round(width * 0.05)) or h < 10 or pixels / max(1, w * h) < 0.4:
            continue
        washed_pixels += pixels
        report["visualMismatchRegions"] += 1
        report["visualMismatchPixels"] += pixels
        report["issues"].append({"problem": "visualContentMismatch", "elementId": f"washed_{x}_{y}",
                                 "bbox": [x, y, x + w, y + h], "pixelArea": pixels,
                                 "reason": "colored_residual_washed_out"})
    report["washedColoredAssetPixels"] = washed_pixels
    missing = _bounded_components(np.uint8(lost) * 255, 9, width, height)
    for x, y, w, h, pixels in missing:
        if pixels < max(40, round(width * height * 0.00003)) and not _small_solid_decoration(
                source, lost[y:y + h, x:x + w], (x, y, w, h), page_color):
            continue
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
        if (x2 - x1) * (y2 - y1) >= width * height * 0.90:
            asset_alpha = _visual_mask(item, box, source_path)
            occupied_fraction = float(np.mean(asset_alpha > 32))
            if occupied_fraction >= 0.95:
                report["monolithicPageImageCount"] += 1
                report["issues"].append({"problem": "monolithicPageImage", "elementId": item.get("id"),
                                         "bbox": list(box), "occupiedFraction": round(occupied_fraction, 3)})
        # A transparent crop may occupy only a small part of its bounding box.
        # Judge the pixels it actually owns, not unrelated white space nearby.
        owner_pixels = _visual_mask(item, box, source_path)
        source_detail = (contrast[y1:y2, x1:x2] >= 8) & ~np.all(source[y1:y2, x1:x2] >= 253, axis=2)
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
    background_count, _, background_stats, _ = cv2.connectedComponentsWithStats(background_foreground, 8)
    for index in range(1, background_count):
        x, y, w, h, pixels = [int(value) for value in background_stats[index]]
        if pixels >= width * height * 0.62 and w * h < width * height * 0.95:
            residual.append((x, y, w, h, pixels))
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
        if issue.get("problem") not in {"missingBackplate", "missingVisualObject", "largeVisualLoss", "blankVisualOwner", "visualContentMismatch", "unsupportedNativeShape"}:
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
        if issue["problem"] in {"missingBackplate", "largeVisualLoss"} and flat:
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
                elif issue["problem"] == "unsupportedNativeShape":
                    missing_pixels = difference >= 2
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
                    if issue["problem"] not in {"missingVisualObject", "unsupportedNativeShape"}:
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
                            distinct_visual &= ~_connected_glyph_blends(patch, bgr)
                        except ValueError:
                            pass
                    mask[top:bottom, left:right][~distinct_visual] = 0
            if np.count_nonzero(mask) < 24:
                continue
            if issue["problem"] != "unsupportedNativeShape":
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


def _connected_glyph_blends(patch: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Exclude antialiased ink connected to text, without removing its plate.

    Source glyph edges blend with the local support color and can be far from
    the estimated ink color. A color corridor alone could also erase a pale
    blue plate, so only components connected to an actual ink seed qualify.
    """
    pixels = patch.astype(np.float32)
    surface = np.median(pixels.reshape(-1, 3), axis=0)
    direction = ink.astype(np.float32) - surface
    length_squared = float(np.dot(direction, direction))
    if length_squared < 1600:
        return np.zeros(patch.shape[:2], np.bool_)
    fraction = np.sum((pixels - surface) * direction, axis=2) / length_squared
    blended = surface + fraction[:, :, None] * direction
    corridor = ((fraction >= 0.08) & (fraction <= 1.15)
                & (np.max(np.abs(pixels - blended), axis=2) <= 18))
    seeds = np.max(np.abs(pixels - ink), axis=2) <= 38
    count, labels = cv2.connectedComponents(np.uint8(corridor), 8)
    seeded_labels = np.unique(labels[seeds & corridor])
    seeded_labels = seeded_labels[seeded_labels != 0]
    return np.isin(labels, seeded_labels) if count > 1 else np.zeros(patch.shape[:2], np.bool_)


def recover_initial_missing_regions(source_path: Path, background_path: Path, preview_path: Path,
                                    layout: dict, asset_dir: Path, project_id: str, page_index: int) -> int:
    """Add only missing local visuals whose rendered result measurably improves."""
    from app.services.visual_qa.analyzer import render_preview

    before = audit_objectization(source_path, background_path, preview_path, layout)
    issues = [issue for issue in before["issues"] if issue.get("problem") in {"missingBackplate", "missingVisualObject", "largeVisualLoss"}]
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
                target = np.zeros(pale.shape, np.bool_)
                for item in created:
                    if (item.get("metadata") or {}).get("qaIssue") not in {"paleAssetGap", "paleTextSupportGap"}:
                        continue
                    box = _box(item, source.shape[1], source.shape[0])
                    if box is None:
                        continue
                    x1, y1, x2, y2 = box
                    target[y1:y2, x1:x2] |= _visual_mask(item, box, source_path) > 32
                before_gap = int(np.count_nonzero(pale & target))
                after_gap = int(np.count_nonzero(_pale_gap_pixels(source, current) & pale & target))
                global_after_gap = int(np.count_nonzero(_pale_gap_pixels(source, current)))
                global_before_gap = int(np.count_nonzero(pale))
                improved = (valid and before_gap >= 80 and after_gap <= before_gap * 0.8
                            and global_after_gap <= global_before_gap + 24
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


def _coherent_text_support(source_crop: np.ndarray, gap: np.ndarray) -> np.ndarray:
    """Separate a missing plate from pale antialiasing around editable glyphs."""
    if not np.any(gap) or min(gap.shape) < 10:
        return np.zeros(gap.shape, np.bool_)
    hsv = cv2.cvtColor(source_crop, cv2.COLOR_BGR2HSV)
    dark_ink = np.uint8((hsv[:, :, 2] < 225) | (hsv[:, :, 1] > 32))
    glyph_halo = cv2.dilate(dark_ink, np.ones((5, 5), np.uint8)) != 0
    candidate = np.uint8(gap & ~glyph_halo) * 255
    opened = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    coherent = np.zeros(gap.shape, np.uint8)
    for index in range(1, count):
        _, _, width, height, pixels = [int(value) for value in stats[index]]
        if width >= 10 and height >= 10 and pixels >= 80:
            coherent[labels == index] = 255
    return (cv2.dilate(coherent, np.ones((5, 5), np.uint8)) != 0) & gap


def _recover_pale_asset_gaps(source_path: Path, preview_path: Path, layout: dict,
                             asset_dir: Path, project_id: str, page_index: int,
                             *, target_ids: set[str] | None = None,
                             asset_prefix: str | None = None) -> list[dict]:
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
        box = _text_source_box(item, width, height, padding=2)
        if box:
            x1, y1, x2, y2 = box
            missing[y1:y2, x1:x2] = False
    created: list[dict] = []
    for owner in list(layout.get("elements", [])):
        if owner.get("type") != "image" or any((owner.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        if target_ids is not None and str(owner.get("id")) not in target_ids:
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
        if np.count_nonzero(stable_gap) < max(16, round(area * 0.004)):
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
                if np.count_nonzero(selected) >= max(16, round(area * 0.004)):
                    identifier = f"{asset_prefix or f'initial_page_{page_index}'}_pale_merged_{uuid4().hex[:10]}"
                    path = asset_dir / f"{identifier}.png"
                    if cv2.imwrite(str(path), merged):
                        owner["src"] = f"/media/assets/{project_id}/{path.name}"
                        owner.setdefault("metadata", {}).update({"paleGapMergedPixels": int(np.count_nonzero(selected)),
                                                                 "qaIssue": "paleAssetGap"})
                        created.append(owner)
                        missing[y1:y2, x1:x2][selected] = False
                        continue
        identifier = f"{asset_prefix or f'initial_page_{page_index}'}_pale_gap_{len(created) + 1:03d}"
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
    created.extend(_recover_faded_local_supports(source_path, source, preview, layout,
                                                 asset_dir, project_id, page_index, target_ids))
    if target_ids is None:
        created.extend(_recover_text_support_gaps(source, text_support_gaps, layout, asset_dir, project_id, page_index))
    return created


def _recover_faded_local_supports(source_path: Path, source: np.ndarray, preview: np.ndarray,
                                  layout: dict, asset_dir: Path, project_id: str, page_index: int,
                                  target_ids: set[str] | None) -> list[dict]:
    """Restore bounded pale card surfaces missed by per-asset alpha repair."""
    height, width = source.shape[:2]
    candidate = _faded_local_support_mask(source, preview)
    parents = [item for item in layout.get("elements", []) if item.get("type") == "image"
               and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))
               and (target_ids is None or str(item.get("id")) in target_ids)
               and (box := _box(item, width, height)) is not None
               and (box[2] - box[0]) * (box[3] - box[1]) <= width * height * 0.55]
    if not parents:
        return []
    # Existing visible pixels already have an owner. Only add support into
    # actual transparent holes, never another copy over a valid image.
    owned = np.zeros((height, width), np.bool_)
    for item in layout.get("elements", []):
        if item.get("type") not in VISUAL_TYPES or any((item.get("metadata") or {}).get(key)
                for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        box = _box(item, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        owned[y1:y2, x1:x2] |= _visual_mask(item, box, source_path) > 32
    candidate[owned] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    created: list[dict] = []
    merged_assets: dict[str, tuple[dict, np.ndarray, int]] = {}
    minimum = max(120, round(width * height * 0.00006))
    for index in range(1, count):
        x, y, w, h, pixels = [int(value) for value in stats[index]]
        if pixels < minimum or w < 8 or h < 5 or w * h > width * height * 0.12:
            continue
        component = np.uint8(labels[y:y + h, x:x + w] == index) * 255
        box = (x, y, x + w, y + h)
        eligible = [item for item in parents if (parent_box := _box(item, width, height))
                    and _overlap_of_first(box, parent_box) >= 0.75]
        if not eligible:
            continue
        parent = min(eligible, key=lambda item: _box_area(_box(item, width, height)))
        parent_box = _box(parent, width, height)
        px1, py1, px2, py2 = parent_box
        key = str(parent.get("id"))
        prior_path = asset_dir / Path(str(parent.get("src") or "")).name
        prior = (merged_assets[key][1] if key in merged_assets else
                 cv2.imread(str(prior_path), cv2.IMREAD_UNCHANGED) if prior_path.is_file() else None)
        if (prior is not None and prior.ndim == 3 and prior.shape[2] == 4
                and prior.shape[:2] == (py2 - py1, px2 - px1)
                and _overlap_of_first(box, parent_box) == 1):
            merged = prior.copy()
            selected = component != 0
            region = merged[y - py1:y + h - py1, x - px1:x + w - px1]
            region[selected, :3] = source[y:y + h, x:x + w][selected]
            region[selected, 3] = 255
            total = pixels + (merged_assets[key][2] if key in merged_assets else 0)
            merged_assets[key] = (parent, merged, total)
            continue
        identifier = f"initial_page_{page_index}_faded_support_{uuid4().hex[:10]}"
        path = asset_dir / f"{identifier}.png"
        if not cv2.imwrite(str(path), np.dstack((source[y:y + h, x:x + w], component))):
            continue
        item = {"id": identifier, "type": "image", "x": x, "y": y, "width": w, "height": h,
                "rotation": 0, "zIndex": int(parent.get("zIndex") or 0) + 1,
                "groupId": parent.get("groupId"),
                "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1},
                "metadata": {"reconstructionStrategy": "cutout_image", "reconstructionStrategySource": "objectization_audit",
                             "layerRole": "container_detail", "qaIssue": "paleAssetGap", "parentId": parent.get("id"),
                             "sourceMaskPixels": pixels}}
        layout.setdefault("elements", []).append(item)
        created.append(item)
    for parent, merged, pixels in merged_assets.values():
        path = asset_dir / f"faded_support_merged_{uuid4().hex[:12]}.png"
        if not cv2.imwrite(str(path), merged):
            continue
        parent["src"] = f"/media/assets/{project_id}/{path.name}"
        parent.setdefault("metadata", {}).update({"qaIssue": "paleAssetGap",
                                                 "fadedSupportMergedPixels": pixels})
        created.append(parent)
    return created


def _faded_local_support_mask(source: np.ndarray, preview: np.ndarray) -> np.ndarray:
    border = np.concatenate((source[0], source[-1], source[:, 0], source[:, -1]))
    page_color = np.median(border, axis=0).astype(np.int16)
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    difference = np.max(np.abs(source.astype(np.int16) - preview.astype(np.int16)), axis=2)
    page_difference = np.max(np.abs(source.astype(np.int16) - page_color), axis=2)
    candidate = np.uint8((hsv[:, :, 2] >= 225) & (hsv[:, :, 1] <= 32)
                           & np.all(preview >= 253, axis=2) & (difference >= 5)
                           & (page_difference >= 6)) * 255
    return cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def _box_area(box: tuple[int, int, int, int]) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def _overlap_of_first(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]))
    return intersection / max(1, _box_area(first))


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
        gap = _coherent_text_support(crop, missing[y1:y2, x1:x2])
        if np.count_nonzero(gap) < max(80, round(area * 0.12)) or np.mean(pale) < 0.40:
            continue
        support = np.median(crop[pale], axis=0).astype(np.int16)
        ink = np.uint8(np.max(np.abs(crop.astype(np.int16) - support), axis=2) >= 28) * 255
        # Bold headings can legitimately occupy nearly half a pale label.
        # Keep their supporting surface when the surrounding pale color is
        # still dominant, then erase only the glyph pixels in the new asset.
        ink_limit = 0.48 if float(np.mean(pale)) >= 0.60 else 0.38
        if np.mean(ink != 0) > ink_limit:
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
        if pixels < minimum or ((w < 3 or h < 3) and not is_meaningful_stroke(w, h, pixels, width, height)) or w * h > width * height * 0.62:
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
