from __future__ import annotations
from app.utils import image_io

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.services.pptx.renderer import _default_strategy, _path_from_src


def _font(style: dict[str, Any], size_scale: float = 1.0) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(8, int(float(style.get("fontSize", 20)) * size_scale))
    family = style.get("fontFamily", "Microsoft YaHei")
    candidates = [
        Path("C:/Windows/Fonts") / f"{family}.ttf",
        Path("C:/Windows/Fonts/msyh.ttc") if "YaHei" in family else Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_preview(background_path: Path, layout: dict[str, Any], output_path: Path) -> Path:
    base = Image.open(background_path).convert("RGBA") if background_path.exists() else Image.new("RGBA", (int(layout["slide"]["width"]), int(layout["slide"]["height"])), "white")
    draw = ImageDraw.Draw(base, "RGBA")
    for element in sorted(layout.get("elements", []), key=lambda item: item.get("zIndex", 0)):
        kind = element.get("type")
        metadata = element.get("metadata") or {}
        if metadata.get("suppressed") or metadata.get("suppressRender") or metadata.get("ownedBy"):
            continue
        strategy = metadata.get("reconstructionStrategy") or _default_strategy(kind)
        if strategy == "group":
            continue
        x, y = float(element.get("x", 0)), float(element.get("y", 0))
        w, h = max(1.0, float(element.get("width", 1))), max(1.0, float(element.get("height", 1)))
        style = element.get("style") or {}
        if strategy in {"transparent_image", "local_image", "cutout_image", "background_image"}:
            image_path = _path_from_src(element.get("src"))
            if image_path and image_path.exists():
                with Image.open(image_path) as source:
                    asset = source.convert("RGBA").resize((max(1, int(round(w))), max(1, int(round(h)))), Image.Resampling.LANCZOS)
                base.alpha_composite(asset, (int(round(x)), int(round(y))))
                draw = ImageDraw.Draw(base, "RGBA")
            continue
        if strategy == "editable_text":
            draw.multiline_text((x, y), element.get("text") or "", font=_font(style), fill=style.get("color", "#111827"), spacing=max(0, int(float(style.get("lineSpacing", 1.1)) * 4)), align=style.get("align", "left"))
        elif strategy == "native_shape" and kind in {"rectangle", "roundedRectangle"}:
            fill = style.get("fill", "#DCE6F1")
            outline = style.get("stroke", fill)
            alpha = int(float(style.get("opacity", 1)) * 255)
            draw.rounded_rectangle((x, y, x + w, y + h), radius=min(w, h) * 0.16 if kind == "roundedRectangle" else 0, fill=fill + f"{alpha:02X}" if isinstance(fill, str) and len(fill) == 7 else fill, outline=outline, width=max(1, int(float(style.get("strokeWidth", 1)))))
        elif strategy == "native_shape" and kind in {"ellipse", "circle"}:
            draw.ellipse((x, y, x + w, y + h), fill=style.get("fill", "#DCE6F1"), outline=style.get("stroke", style.get("fill", "#DCE6F1")), width=max(1, int(float(style.get("strokeWidth", 1)))))
        elif strategy == "native_shape" and kind in {"line", "arrow"}:
            draw.line((x, y + h / 2, x + w, y + h / 2), fill=style.get("stroke", "#17365D"), width=max(1, int(float(style.get("strokeWidth", 1)))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(output_path)
    return output_path


def _similarity(original: np.ndarray, preview: np.ndarray) -> tuple[float, float, float]:
    original = cv2.resize(original, (preview.shape[1], preview.shape[0]))
    diff = cv2.absdiff(original, preview)
    pixel = max(0.0, 1.0 - float(np.mean(diff)) / 255.0)
    original_edges = cv2.Canny(original, 60, 160)
    preview_edges = cv2.Canny(preview, 60, 160)
    edge = 1.0 - float(np.mean(cv2.absdiff(original_edges, preview_edges))) / 255.0
    gray_a = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    mse = float(np.mean((gray_a.astype(np.float32) - gray_b.astype(np.float32)) ** 2))
    ssim_like = 1.0 / (1.0 + mse / (255.0 ** 2))
    return pixel, edge, ssim_like


def run_visual_qa(original_path: Path, preview_path: Path, output_dir: Path, layout: dict[str, Any]) -> dict[str, Any]:
    original = image_io.imread(str(original_path), cv2.IMREAD_COLOR)
    preview = image_io.imread(str(preview_path), cv2.IMREAD_COLOR)
    if original is None or preview is None:
        score = {"overall": 0.0, "textRegionScore": 0.0, "layoutScore": 0.0, "componentScore": 0.0, "colorSimilarity": 0.0, "backgroundScore": 0.0, "ghostingPenalty": 0.0, "duplicatePenalty": 0.0, "regions": []}
        return score
    original = cv2.resize(original, (preview.shape[1], preview.shape[0]))
    pixel, edge, ssim_like = _similarity(original, preview)
    text_mask = _element_mask(preview.shape[:2], layout, {"text"})
    component_mask = _element_mask(preview.shape[:2], layout, {"image", "ellipse", "rectangle", "roundedRectangle", "line", "arrow"})
    layout_mask = cv2.bitwise_or(text_mask, component_mask)
    background_mask = cv2.bitwise_not(layout_mask)
    text_pixel, text_edge = _masked_similarity(original, preview, text_mask)
    component_pixel, component_edge = _masked_similarity(original, preview, component_mask)
    layout_pixel, layout_edge = _masked_similarity(original, preview, layout_mask)
    background_pixel, background_edge = _masked_similarity(original, preview, background_mask)
    text_region = text_pixel * 0.35 + text_edge * 0.65
    layout_score = layout_pixel * 0.25 + layout_edge * 0.75
    component_score = component_pixel * 0.45 + component_edge * 0.55
    background_score = background_pixel * 0.7 + background_edge * 0.3
    color_similarity = _color_similarity(original, preview, layout_mask)
    ghosting_penalty = _ghosting_penalty(original, preview, text_mask)
    duplicate_penalty = _duplicate_penalty(layout)
    regions, issues = _critical_regions(original, preview, layout)
    geometry, structural_issues = _structural_checks(layout, preview.shape[:2], regions)
    issues.extend(structural_issues)
    critical_score = sum(item["score"] * item["weight"] for item in regions) / max(1.0, sum(item["weight"] for item in regions)) if regions else layout_score
    overlap_penalty = min(0.2, sum(0.04 for issue in issues if issue["problem"] == "textOverlap"))
    overall = (
        0.23 * critical_score
        + 0.18 * text_region
        + 0.14 * component_score
        + 0.10 * layout_score
        + 0.05 * color_similarity
        + 0.05 * background_score
        + 0.25 * geometry["structuralScore"]
        - ghosting_penalty
        - duplicate_penalty
        - overlap_penalty
    )
    score = {
        "overall": round(max(0.0, min(1.0, overall)), 4),
        "textRegionScore": round(text_region, 4),
        "layoutScore": round(layout_score, 4),
        "componentScore": round(component_score, 4),
        "colorSimilarity": round(color_similarity, 4),
        "backgroundScore": round(background_score, 4),
        "ghostingPenalty": round(ghosting_penalty, 4),
        "duplicatePenalty": round(duplicate_penalty, 4),
        "layout": round(layout_score, 4),
        "text": round(text_region, 4),
        "color": round(color_similarity, 4),
        "structure": round(component_score, 4),
        "pixelSimilarity": round(pixel, 4),
        "ssim": round(ssim_like, 4),
        "edgeSimilarity": round(edge, 4),
        "criticalRegionScore": round(critical_score, 4),
        **geometry,
        "textOverlapPenalty": round(overlap_penalty, 4),
        "regions": regions,
        "issues": issues,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    difference = cv2.absdiff(cv2.resize(original, (preview.shape[1], preview.shape[0])), preview)
    image_io.imwrite(str(output_dir / "difference.png"), difference)
    return score


def enrich_quality_score(score: dict[str, Any], *, editable_coverage: float, movable_coverage: float, ghosting_count: int, background_residual_count: int, professional_pending: int = 0) -> dict[str, Any]:
    """Apply ownership and editability gates after the page assets have been measured."""
    visual = float(score.get("overall") or 0)
    coverage = max(0.0, min(1.0, editable_coverage))
    movable = max(0.0, min(1.0, movable_coverage))
    penalties = min(0.35, ghosting_count * 0.08 + background_residual_count * 0.08 + professional_pending * 0.05)
    score.update({
        "visualSceneScore": round(visual, 4),
        "editableTextCoverage": round(coverage, 4),
        "movableVisualCoverage": round(movable, 4),
        "ghostingCount": ghosting_count,
        "backgroundResidualCount": background_residual_count,
        "professionalRepairPending": professional_pending,
        "overall": round(max(0.0, min(1.0, 0.65 * visual + 0.21 * coverage + 0.14 * movable - penalties)), 4),
    })
    if ghosting_count:
        score.setdefault("issues", []).append({"problem": "ghosting", "count": ghosting_count})
    if background_residual_count:
        score.setdefault("issues", []).append({"problem": "backgroundResidual", "count": background_residual_count})
    if professional_pending:
        score.setdefault("issues", []).append({"problem": "professionalInpaintingPending", "count": professional_pending})
    return score


def _structural_checks(layout: dict[str, Any], shape: tuple[int, int], regions: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    height, width = shape
    issues: list[dict[str, Any]] = []
    active = [item for item in layout.get("elements", []) if item.get("type") != "background" and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    by_id = {str(item.get("id")): item for item in active}
    placements: list[float] = []
    image_scores: list[float] = []
    boundary_scores: list[float] = []
    for item in active:
        item_id = str(item.get("id"))
        x, y = float(item.get("x") or 0), float(item.get("y") or 0)
        w, h = float(item.get("width") or 0), float(item.get("height") or 0)
        box_area = max(1.0, w * h)
        clipped_area = max(0.0, min(width, x + w) - max(0.0, x)) * max(0.0, min(height, y + h) - max(0.0, y))
        boundary_scores.append(min(1.0, clipped_area / box_area))
        if clipped_area / box_area < 0.9:
            issues.append({"elementId": item_id, "problem": "moduleBoundary"})
        meta = item.get("metadata") or {}
        if item.get("type") == "text":
            raw = meta.get("rawOCRBBox")
            if isinstance(raw, list) and len(raw) == 4:
                displacement = max(abs(x - float(raw[0])), abs(y - float(raw[1]))) / max(8.0, float(raw[3]) - float(raw[1]))
                placements.append(max(0.0, 1.0 - displacement / 1.5))
                if displacement > 0.6:
                    issues.append({"elementId": item_id, "problem": "wrongBBox"})
            owner = by_id.get(str(meta.get("textCleanedFromAsset") or ""))
            if owner and int(item.get("zIndex") or 0) <= int(owner.get("zIndex") or 0):
                issues.append({"elementId": item_id, "problem": "wrongZOrder"})
        if item.get("type") == "image":
            asset_path = _path_from_src(item.get("src"))
            if asset_path and asset_path.is_file() and w > 0 and h > 0:
                try:
                    with Image.open(asset_path) as asset:
                        intrinsic = asset.width / max(1, asset.height)
                    difference = abs(math.log(max(0.01, w / h) / max(0.01, intrinsic)))
                    image_scores.append(max(0.0, 1.0 - difference))
                    if difference > 0.18:
                        issues.append({"elementId": item_id, "problem": "imageDistortion"})
                except OSError:
                    issues.append({"elementId": item_id, "problem": "brokenChartOrModule"})
    for index, left in enumerate(active):
        for right in active[index + 1:]:
            if left.get("type") != right.get("type"):
                continue
            if left.get("type") == "text" and str(left.get("text") or "").strip().casefold() != str(right.get("text") or "").strip().casefold():
                continue
            if left.get("type") == "image" and left.get("src") != right.get("src"):
                continue
            if left.get("type") not in {"text", "image"}:
                continue
            left_box = (float(left.get("x") or 0), float(left.get("y") or 0), float(left.get("width") or 0), float(left.get("height") or 0))
            right_box = (float(right.get("x") or 0), float(right.get("y") or 0), float(right.get("width") or 0), float(right.get("height") or 0))
            overlap = max(0.0, min(left_box[0] + left_box[2], right_box[0] + right_box[2]) - max(left_box[0], right_box[0])) * max(0.0, min(left_box[1] + left_box[3], right_box[1] + right_box[3]) - max(left_box[1], right_box[1]))
            if overlap / max(1.0, min(left_box[2] * left_box[3], right_box[2] * right_box[3])) > 0.85:
                issues.append({"elementId": left.get("id"), "otherElementId": right.get("id"), "problem": "duplicateText" if left.get("type") == "text" else "duplicateElement"})
    for region in regions:
        item = by_id.get(str(region.get("elementId")))
        if item and item.get("type") == "image" and str(item.get("role") or "").lower() in {"chart", "flowchart", "diagram"} and float(region.get("score") or 0) < 0.65:
            issues.append({"elementId": item.get("id"), "problem": "brokenChartOrModule", "bbox": region.get("bbox")})
    placement = sum(placements) / len(placements) if placements else 1.0
    aspect = sum(image_scores) / len(image_scores) if image_scores else 1.0
    boundaries = sum(boundary_scores) / len(boundary_scores) if boundary_scores else 1.0
    duplicate_penalty = min(0.25, sum(issue["problem"] in {"duplicateText", "duplicateElement"} for issue in issues) * 0.05)
    structural = max(0.0, 0.4 * placement + 0.3 * aspect + 0.3 * boundaries - duplicate_penalty)
    return {"textPlacementScore": round(placement, 4), "imageAspectScore": round(aspect, 4), "moduleBoundaryScore": round(boundaries, 4), "structuralScore": round(structural, 4)}, issues


def _critical_regions(original: np.ndarray, preview: np.ndarray, layout: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    height, width = preview.shape[:2]
    active = [item for item in layout.get("elements", []) if item.get("type") != "background" and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    regions: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    text_boxes: list[tuple[dict[str, Any], tuple[int, int, int, int]]] = []
    for item in active:
        x1 = max(0, min(width, round(float(item.get("x", 0)))))
        y1 = max(0, min(height, round(float(item.get("y", 0)))))
        x2 = max(x1, min(width, round(float(item.get("x", 0)) + float(item.get("width", 0)))))
        y2 = max(y1, min(height, round(float(item.get("y", 0)) + float(item.get("height", 0)))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        kind = item.get("type")
        pixel, edge = _similarity(original[y1:y2, x1:x2], preview[y1:y2, x1:x2])[:2]
        score = pixel * 0.35 + edge * 0.65
        weight = 3.0 if item.get("role") == "main_title" else 2.0 if kind == "text" else 1.5 if kind == "image" else 1.0
        regions.append({"elementId": item.get("id"), "kind": kind, "role": item.get("role"), "bbox": [x1, y1, x2, y2], "score": round(score, 4), "weight": weight})
        if score < 0.6 and kind in {"text", "image"}:
            issues.append({"elementId": item.get("id"), "problem": "criticalRegionMismatch", "score": round(score, 4)})
        if kind == "text":
            text_boxes.append((item, (x1, y1, x2, y2)))
    for index, (left_item, left) in enumerate(text_boxes):
        for right_item, right in text_boxes[index + 1:]:
            overlap = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(0, min(left[3], right[3]) - max(left[1], right[1]))
            smaller = min((left[2] - left[0]) * (left[3] - left[1]), (right[2] - right[0]) * (right[3] - right[1]))
            if overlap / max(1, smaller) > 0.25:
                issues.append({"elementId": left_item.get("id"), "otherElementId": right_item.get("id"), "problem": "textOverlap"})
    return regions, issues


def _element_mask(shape: tuple[int, int], layout: dict[str, Any], kinds: set[str]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in layout.get("elements", []):
        metadata = item.get("metadata") or {}
        if item.get("type") not in kinds or any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        x1, y1 = int(max(0, float(item.get("x", 0)))), int(max(0, float(item.get("y", 0))))
        x2 = int(min(width, np.ceil(float(item.get("x", 0)) + float(item.get("width", 0)))))
        y2 = int(min(height, np.ceil(float(item.get("y", 0)) + float(item.get("height", 0)))))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), 255, -1)
    return mask


def _masked_similarity(original: np.ndarray, preview: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    selected = mask > 0
    if not np.any(selected):
        return 1.0, 1.0
    pixel = max(0.0, 1.0 - float(np.mean(cv2.absdiff(original, preview)[selected])) / 255.0)
    original_edges = cv2.Canny(original, 60, 160)
    preview_edges = cv2.Canny(preview, 60, 160)
    edge = max(0.0, 1.0 - float(np.mean(cv2.absdiff(original_edges, preview_edges)[selected])) / 255.0)
    return pixel, edge


def _color_similarity(original: np.ndarray, preview: np.ndarray, mask: np.ndarray) -> float:
    selected = mask > 0
    if not np.any(selected):
        selected = np.ones(mask.shape, dtype=bool)
    mean_a = np.mean(original[selected].astype(np.float32), axis=0)
    mean_b = np.mean(preview[selected].astype(np.float32), axis=0)
    return max(0.0, 1.0 - float(np.linalg.norm(mean_a - mean_b)) / (255.0 * math.sqrt(3)))


def _ghosting_penalty(original: np.ndarray, preview: np.ndarray, text_mask: np.ndarray) -> float:
    selected = text_mask > 0
    if not np.any(selected):
        return 0.0
    original_edges = cv2.Canny(original, 45, 135)[selected] > 0
    preview_edges = cv2.Canny(preview, 45, 135)[selected] > 0
    excess = max(0.0, float(preview_edges.mean()) - float(original_edges.mean()))
    return min(0.3, excess * 1.8)


def _duplicate_penalty(layout: dict[str, Any]) -> float:
    active = [item for item in layout.get("elements", []) if not (item.get("metadata") or {}).get("suppressRender") and not (item.get("metadata") or {}).get("ownedBy")]
    whole_badge_groups = {item.get("groupId") for item in active if (item.get("metadata") or {}).get("wholeBadgeAsset")}
    duplicates = sum(1 for item in active if item.get("groupId") in whole_badge_groups and item.get("type") == "ellipse")
    return min(0.25, duplicates * 0.05)
