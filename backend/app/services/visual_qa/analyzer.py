from __future__ import annotations

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
    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    preview = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
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
    critical_score = sum(item["score"] * item["weight"] for item in regions) / max(1.0, sum(item["weight"] for item in regions)) if regions else layout_score
    overlap_penalty = min(0.2, sum(0.04 for issue in issues if issue["problem"] == "textOverlap"))
    overall = (
        0.30 * critical_score
        + 0.25 * text_region
        + 0.20 * component_score
        + 0.15 * layout_score
        + 0.07 * color_similarity
        + 0.03 * background_score
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
        "textOverlapPenalty": round(overlap_penalty, 4),
        "regions": regions,
        "issues": issues,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    difference = cv2.absdiff(cv2.resize(original, (preview.shape[1], preview.shape[0])), preview)
    cv2.imwrite(str(output_dir / "difference.png"), difference)
    return score


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
        if item.get("type") not in kinds or metadata.get("suppressRender") or metadata.get("ownedBy"):
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
