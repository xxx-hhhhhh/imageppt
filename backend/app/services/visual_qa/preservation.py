from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from app.services.reconstruction.object_first import important_mask, read_image
from app.services.scene.ownership import resolve_asset, valid_replacement


def preservation_qa(source_path: Path, preview_path: Path, background_path: Path, layout: dict, output_root: Path) -> dict:
    source = read_image(source_path)
    preview = read_image(preview_path)
    if preview.shape != source.shape:
        preview = cv2.resize(preview, (source.shape[1], source.shape[0]))
    important = important_mask(source) > 0
    delta = np.max(np.abs(source.astype(np.int16) - preview.astype(np.int16)), axis=2)
    covered = np.zeros(important.shape, bool)
    movable = np.zeros_like(covered)
    text_pixels = np.zeros_like(covered)
    missing_assets = missing_visual = duplicates = 0
    images = shapes = editable = 0
    seen = set()
    records = []
    for item in layout.get("elements", []):
        owner = item.get("owner")
        if owner in {"background", "intentional_ignore"}:
            continue
        if item.get("type") == "image":
            images += 1
        if owner == "native_shape":
            shapes += 1
        if owner == "editable_text":
            editable += 1
        valid = valid_replacement(item, output_root)
        if owner == "movable_image" and not valid:
            missing_assets += 1
        path = resolve_asset(item.get("mask"), output_root)
        mask = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) > 0 if path and path.is_file() else None
        if mask is not None and mask.shape == covered.shape:
            if valid:
                covered |= mask
                if owner in {"movable_image", "native_shape"}:
                    movable |= mask
                elif owner == "editable_text":
                    text_pixels |= mask
            # Test non-text source evidence in the actual recomposition, not just bbox coverage.
            selected = mask & important & ~text_pixels
            if owner != "editable_text" and np.any(selected):
                present = float(np.mean(delta[selected] <= 45))
                if not valid or present < 0.7:
                    missing_visual += 1
                records.append({"id": item["id"], "sourcePixels": int(selected.sum()), "renderedPixelRetention": round(present, 4), "validOwner": valid})
        elif owner != "editable_text" and not valid:
            missing_visual += 1
        signature = (item.get("type"), item.get("text") or item.get("src"), *(round(float(item.get(k, 0)), 1) for k in ("x", "y", "width", "height")))
        if signature in seen:
            duplicates += 1
        seen.add(signature)
    nontext = important & ~text_pixels
    # Pixels must have both a live owner and visible evidence, so lost assets cannot pass.
    preserved = (covered & nontext & (delta <= 45)).sum() / max(1, nontext.sum())
    background = read_image(background_path)
    residual = important_mask(background) > 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(residual.astype(np.uint8), 8)
    residual_count = sum(stats[i, cv2.CC_STAT_AREA] >= 16 for i in range(1, n))
    ocr_count = int((layout.get("metadata") or {}).get("ocrRegionCount", editable))
    return {
        "editableTextCoverage": round(min(1.0, editable / max(1, ocr_count)) if ocr_count else 1.0, 4),
        "objectExtractionCoverage": round(float((covered & important).sum() / max(1, important.sum())), 4),
        "movableVisualCoverage": round(float((movable & nontext).sum() / max(1, nontext.sum())) if nontext.any() else 1.0, 4),
        "missingVisualCount": int(missing_visual), "missingAssetCount": missing_assets,
        "backgroundResidualCount": int(residual_count),
        "ghostingCount": int((layout.get("metadata") or {}).get("ghostingCount", 0)),
        "ghostingMeasurement": "source ink residual after single local patch; not independent OCR",
        "duplicateCount": duplicates, "shapeCount": shapes, "imageAssetCount": images,
        "visualAreaPreserved": round(float(preserved) if nontext.any() else 1.0, 4),
        "backgroundWhiteArea": round(float(np.all(background >= 248, axis=2).mean()), 4),
        "unownedVisualPixelCount": int((important & ~covered).sum()),
        "objects": records,
    }


def revision_gate(previous: dict, candidate: dict, old_layout: dict, new_layout: dict) -> tuple[bool, list[str]]:
    reasons = []
    if candidate.get("visualAreaPreserved", 0) < previous.get("visualAreaPreserved", 0) - 0.005:
        reasons.append("visual_area_decreased")
    for key in ("missingVisualCount", "missingAssetCount", "backgroundResidualCount", "duplicateCount", "ghostingCount", "unownedVisualPixelCount"):
        if candidate.get(key, 0) > previous.get(key, 0):
            reasons.append(key + "_increased")
    for key in ("objectExtractionCoverage", "movableVisualCoverage", "editableTextCoverage"):
        if candidate.get(key, 0) < previous.get(key, 0) - 0.001:
            reasons.append(key + "_decreased")
    if candidate.get("imageAssetCount", 0) < previous.get("imageAssetCount", 0):
        reasons.append("image_assets_removed")
    if candidate.get("backgroundWhiteArea", 0) > previous.get("backgroundWhiteArea", 0) + 0.02:
        reasons.append("background_whitening")
    old = {item["id"]: item for item in old_layout.get("elements", [])}
    new = {item["id"]: item for item in new_layout.get("elements", [])}
    canvas = new_layout.get("slide") or {}
    for item in new.values():
        values = [float(item.get(key, 0)) for key in ("x", "y", "width", "height")]
        if not all(math.isfinite(value) for value in values) or values[2] <= 0 or values[3] <= 0:
            reasons.append("invalid_layout")
        elif canvas and (values[0] < 0 or values[1] < 0 or values[0]+values[2] > canvas["width"]+1 or values[1]+values[3] > canvas["height"]+1):
            reasons.append("out_of_bounds")
    if not old.keys() <= new.keys():
        reasons.append("objects_removed")
    for element_id, item in old.items():
        revised = new.get(element_id) or {}
        if item.get("owner") == "editable_text" and revised.get("text") != item.get("text"):
            reasons.append("ocr_text_rewritten")
        if revised.get("owner") != item.get("owner") or revised.get("src") != item.get("src"):
            reasons.append("ownership_or_asset_replaced")
    if candidate.get("overall", 0) <= previous.get("overall", 0) + 0.0001:
        reasons.append("no_measured_improvement")
    return not reasons, sorted(set(reasons))
