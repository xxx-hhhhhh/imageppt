from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.pptx.renderer import _path_from_src
from app.services.reconstruction.residual_objects import visual_candidate_mask
from app.services.reconstruction.white_objectization import _occupy_text_glyphs
from app.services.reconstruction.objectization_audit import _connected_glyph_blends


def measure_movable_assets(source_path: Path, background_path: Path, layout: dict, plan: dict, *, debug_path: Path | None = None) -> dict[str, int | float]:
    """Measure independent image ownership and source pixels left beneath it."""
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if source is None or background is None:
        return {"movableAssetCount": 0, "backgroundResidualCount": 0, "movableVisualCoverage": 0.0}
    assets = [item for item in layout.get("elements", []) if item.get("type") == "image" and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy", "sourceTextFallback"))]
    candidates = []
    for module in plan.get("modules", []):
        module_id = str(module.get("moduleId") or "")
        if not module_id.startswith(("detected_visual_", "segmented_visual_", "contour_visual_")) and module.get("reconstructionStrategy") != "cutout_image":
            continue
        box = module.get("bboxPixels")
        if not isinstance(box, list) or len(box) != 4:
            continue
        candidate = tuple(float(value) for value in box)
        if candidate[2] <= candidate[0] or candidate[3] <= candidate[1]:
            continue
        if any(_overlap_fraction(candidate, prior) >= 0.85 for prior in candidates):
            continue
        candidates.append(candidate)
    coverage = _pixel_visual_coverage(source, layout, debug_path=debug_path)
    residual = 0
    source_edges = cv2.Canny(source, 60, 160) > 0
    background_edges = cv2.Canny(background, 60, 160) > 0
    for asset in assets:
        x1, y1, x2, y2 = _clip(_box(asset), source.shape)
        if x2 <= x1 or y2 <= y1:
            continue
        original = source_edges[y1:y2, x1:x2]
        remaining = background_edges[y1:y2, x1:x2]
        if np.count_nonzero(original) < 20:
            continue
        retained = float(np.count_nonzero(original & remaining)) / np.count_nonzero(original)
        if retained >= 0.7:
            residual += 1
    for candidate in candidates:
        if any(_overlap_fraction(candidate, _box(asset)) >= 0.8 for asset in assets):
            continue
        x1, y1, x2, y2 = _clip(candidate, source.shape)
        if x2 > x1 and y2 > y1 and float(np.mean(cv2.absdiff(source[y1:y2, x1:x2], background[y1:y2, x1:x2]))) < 5:
            residual += 1
    return {"movableAssetCount": len(assets), "backgroundResidualCount": residual, "movableVisualCoverage": coverage}


def _pixel_visual_coverage(source: np.ndarray, layout: dict, *, debug_path: Path | None = None) -> float:
    """Measure visible source pixels, not the empty area of an object's box."""
    active = [item for item in layout.get("elements", []) if not any((item.get("metadata") or {}).get(k) for k in ("suppressed", "suppressRender", "ownedBy"))]
    expected = visual_candidate_mask(source)
    glyphs = np.zeros(source.shape[:2], np.uint8)
    for item in active:
        if item.get("type") != "text":
            continue
        raw = (item.get("metadata") or {}).get("rawOCRBBox")
        box = tuple(raw) if isinstance(raw, list) and len(raw) == 4 else _box(item)
        clipped = _clip((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), source.shape)
        _occupy_text_glyphs(source, glyphs, clipped)
        x1, y1, x2, y2 = clipped
        if x2 <= x1 or y2 <= y1:
            continue
        # Bold and tightly packed glyphs can exceed the residual extractor's
        # small-component limits. Exclude their ink, never the whole OCR box:
        # a pale supporting plate inside that box must still have an owner.
        try:
            ink = np.frombuffer(bytes.fromhex(str((item.get("style") or {}).get("color") or "").lstrip("#"))[::-1], dtype=np.uint8)
        except ValueError:
            continue
        if len(ink) == 3:
            blended = _connected_glyph_blends(source[y1:y2, x1:x2], ink)
            edges = cv2.dilate(np.uint8(blended), np.ones((3, 3), np.uint8)) > 0
            glyphs[y1:y2, x1:x2][edges] = 255
    expected &= glyphs == 0
    owned = np.zeros(source.shape[:2], np.bool_)
    for item in active:
        kind = item.get("type")
        if kind not in {"image", "rectangle", "roundedRectangle", "ellipse", "line", "arrow"}:
            continue
        x1, y1, x2, y2 = _clip(_box(item), source.shape)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        original = source[y1:y2, x1:x2]
        tolerance = np.clip(np.max(255 - original.astype(np.int16), axis=2) * .25, 2, 24)
        if kind == "image":
            path = _path_from_src(item.get("src"))
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path and path.is_file() else None
            if image is None or image.ndim != 3:
                continue
            image = cv2.resize(image, (w, h))
            alpha = image[:, :, 3:4].astype(np.float32) / 255 if image.shape[2] == 4 else np.ones((h, w, 1), np.float32)
            alpha *= float((item.get("style") or {}).get("opacity", 1))
            rendered = image[:, :, :3] * alpha + 255 * (1 - alpha)
            matching = (alpha[:, :, 0] > .125) & (np.max(np.abs(rendered - original), axis=2) <= tolerance)
        else:
            matching = np.zeros((h, w), np.bool_)
            style = item.get("style") or {}
            interior = np.full((h, w), 255, np.uint8)
            if kind == "ellipse":
                interior[:] = 0
                cv2.ellipse(interior, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
            elif kind == "roundedRectangle":
                radius = min(18, w // 2, h // 2)
                interior[:] = 0
                cv2.rectangle(interior, (radius, 0), (w - radius - 1, h - 1), 255, -1)
                cv2.rectangle(interior, (0, radius), (w - 1, h - radius - 1), 255, -1)
                for center in ((radius, radius), (w - radius - 1, radius), (radius, h - radius - 1), (w - radius - 1, h - radius - 1)):
                    cv2.circle(interior, center, radius, 255, -1)
            stroke_width = max(0, round(float(style.get("strokeWidth", 1))))
            border = np.zeros((h, w), np.uint8)
            if stroke_width:
                padded = np.pad(interior, stroke_width)
                eroded = cv2.erode(padded, np.ones((stroke_width * 2 + 1, stroke_width * 2 + 1), np.uint8))[stroke_width:-stroke_width, stroke_width:-stroke_width]
                border = np.uint8((interior > 0) & (eroded == 0)) * 255
            if kind in {"line", "arrow"}:
                interior[:] = 0
                border[:] = 0
                cv2.line(border, (0, h // 2), (w - 1, h // 2), 255, max(1, stroke_width))
            for color, geometry in ((style.get("fill"), interior), (style.get("stroke"), border)):
                try:
                    bgr = np.frombuffer(bytes.fromhex(str(color).lstrip("#"))[::-1], dtype=np.uint8)
                except ValueError:
                    continue
                if len(bgr) == 3:
                    opacity = float(style.get("opacity", 1))
                    rendered_color = bgr * opacity + 255 * (1 - opacity)
                    matching |= (geometry > 0) & (np.max(np.abs(original.astype(np.int16) - rendered_color), axis=2) <= tolerance)
        owned[y1:y2, x1:x2] |= matching
    pixels = int(np.count_nonzero(expected))
    if debug_path is not None:
        debug = source.copy()
        for selected, color in ((expected & owned, (70, 180, 70)), (expected & ~owned, (40, 40, 230))):
            debug[selected] = np.uint8(source[selected] * .55 + np.array(color) * .45)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), debug)
    return round(float(np.count_nonzero(expected & owned)) / pixels, 4) if pixels else 1.0


def _box(item: dict) -> tuple[float, float, float, float]:
    x, y = float(item.get("x") or 0), float(item.get("y") or 0)
    return x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)


def _overlap_fraction(candidate: tuple[float, float, float, float], asset: tuple[float, float, float, float]) -> float:
    overlap = max(0, min(candidate[2], asset[2]) - max(candidate[0], asset[0])) * max(0, min(candidate[3], asset[3]) - max(candidate[1], asset[1]))
    return overlap / max(1.0, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))


def _clip(box: tuple[float, float, float, float], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x1, y1, x2, y2 = (round(value) for value in box)
    return max(0, min(width, x1)), max(0, min(height, y1)), max(0, min(width, x2)), max(0, min(height, y2))
