"""Build an editable white slide from source pixels and existing scene owners."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from app.services.reconstruction.residual_objects import _border_color, dense_visual_artwork, extract_residual_objects, visual_candidate_mask


def detect_flat_page_surface(source: np.ndarray) -> np.ndarray | None:
    """Find a genuinely flat page fill so it can be an editable shape."""
    height, width = source.shape[:2]
    border = np.concatenate((source[0], source[-1], source[:, 0], source[:, -1]))
    color = np.median(border, axis=0).astype(np.uint8)
    if int(np.max(255 - color)) < 12:
        return None
    difference = np.max(np.abs(source.astype(np.int16) - color.astype(np.int16)), axis=2)
    border_difference = np.max(np.abs(border.astype(np.int16) - color.astype(np.int16)), axis=1)
    if float(np.mean(border_difference <= 10)) < 0.90 or float(np.mean(difference <= 10)) < 0.85:
        return None
    return color


def objectize_on_white(source_path: Path, background_path: Path, layout: dict, asset_dir: Path, project_id: str, page_index: int) -> dict[str, int]:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(source_path)
    height, width = source.shape[:2]
    asset_dir.mkdir(parents=True, exist_ok=True)
    occupied = np.zeros((height, width), np.uint8)
    elements = layout.setdefault("elements", [])
    for item in elements:
        if item.get("type") == "background":
            item["zIndex"] = -1000
    split_monoliths = _split_monolithic_source_image(source, elements, asset_dir, project_id, page_index)
    active = [item for item in elements if item.get("type") not in {"background", "group"} and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    for item in active:
        metadata = item.setdefault("metadata", {})
        if item.get("type") in {"rectangle", "roundedRectangle", "ellipse", "line", "arrow"} and not item.get("src") and (item.get("style") or {}).get("fill") and metadata.get("reconstructionStrategy") in {None, "local_image", "background_image"}:
            metadata.update({"reconstructionStrategy": "native_shape", "reconstructionStrategySource": "white_objectization"})
    broad_panels = _extract_large_flat_inset_panels(source, active, elements, occupied, page_index)
    round_assets = _extract_round_assets(source, active, elements, occupied, asset_dir, project_id, page_index)
    if dense_visual_artwork(source):
        # Repeated map/photo texture can resemble dozens of flat containers.
        # Let residual extraction keep it as visual assets instead.
        bordered_shapes = bordered_assets = container_count = detail_shapes = detail_assets = 0
    else:
        bordered_shapes, bordered_assets = _extract_bordered_containers(source, active, elements, occupied, asset_dir, project_id, page_index)
        container_count = _extract_flat_containers(source, active, elements, occupied, page_index)
        detail_shapes, detail_assets = _extract_internal_details(source, active, elements, occupied, asset_dir, project_id, page_index)
    for item in active:
        box = _box(item, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        # Only source glyphs belong to the textbox. A plate inside the OCR box
        # remains available for a separate shape or movable image owner.
        if item.get("type") == "text":
            raw = (item.get("metadata") or {}).get("rawOCRBBox")
            if isinstance(raw, list) and len(raw) == 4:
                x1, y1, x2, y2 = _clip(raw, width, height)
            _occupy_text_glyphs(source, occupied, (x1, y1, x2, y2))
            continue
        if item.get("type") in {"rectangle", "roundedRectangle", "ellipse"} and (item.get("style") or {}).get("fill"):
            _occupy_shape_color(source, occupied, (x1, y1, x2, y2), str(item["style"]["fill"]))
            if item["style"].get("stroke"):
                _occupy_shape_color(source, occupied, (x1, y1, x2, y2), str(item["style"]["stroke"]))
        elif item.get("type") == "image":
            _occupy_existing_image(item, occupied, (x1, y1, x2, y2), asset_dir)
        elif item.get("type") in {"line", "arrow"}:
            _occupy_existing_stroke(source, occupied, (x1, y1, x2, y2), item.get("style") or {})
        else:
            occupied[y1:y2, x1:x2] = 255

    residual_assets, residual_stats = extract_residual_objects(source, occupied, asset_dir, project_id, page_index)
    shadow_assets = [item for item in residual_assets if _is_text_shadow_residual(item, active, asset_dir, width, height)]
    shadow_ids = {item["id"] for item in shadow_assets}
    residual_assets = [item for item in residual_assets if item["id"] not in shadow_ids]
    shadow_pixels = sum(int((item.get("metadata") or {}).get("sourcePixelArea") or 0) for item in shadow_assets)
    candidate_pixels = max(0, int(residual_stats["residualCandidateArea"]) - shadow_pixels)
    covered_pixels = sum(int((item.get("metadata") or {}).get("sourcePixelArea") or 0) for item in residual_assets)
    residual_stats["residualObjectsCount"] = len(residual_assets)
    residual_stats["residualCoverageArea"] = covered_pixels
    residual_stats["residualCandidateArea"] = candidate_pixels
    residual_stats["residualObjectizationRate"] = round(covered_pixels / candidate_pixels, 4) if candidate_pixels else 1.0
    residual_stats["textShadowDiscardedCount"] = len(shadow_assets)
    elements.extend(residual_assets)
    page_surface = detect_flat_page_surface(source)
    if page_surface is not None:
        elements.append({"id": f"page_surface_{page_index}", "type": "rectangle", "x": 0, "y": 0,
                         "width": width, "height": height, "rotation": 0, "zIndex": -999,
                         "style": {"fill": _hex_bgr(page_surface), "opacity": 1},
                         "metadata": {"reconstructionStrategy": "native_shape",
                                      "reconstructionStrategySource": "flat_page_surface",
                                      "layerRole": "page_surface", "pageSurface": True}})
    layer_objectized_elements(elements)
    background_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(background_path), np.full_like(source, 255))
    layout["backgroundUrl"] = f"/media/backgrounds/{project_id}/{background_path.name}"
    layout.setdefault("metadata", {})["reconstructionSurfaceMode"] = "white_objectized"
    return {"whiteObjectAssets": len(residual_assets) + round_assets + bordered_assets + detail_assets, "whiteObjectShapes": container_count + bordered_shapes + detail_shapes + broad_panels + int(page_surface is not None), "whiteContainerShapes": container_count + bordered_shapes + bordered_assets + detail_shapes + detail_assets + broad_panels, "whiteInternalDetails": detail_shapes + detail_assets, "whiteBackgroundPixels": width * height, "splitMonolithicImages": split_monoliths, **residual_stats}


def _split_monolithic_source_image(source: np.ndarray, elements: list[dict], asset_dir: Path,
                                   project_id: str, page_index: int) -> int:
    """Retire an opaque screenshot only when independent source regions are verifiable."""
    height, width = source.shape[:2]
    candidates = []
    for item in elements:
        if item.get("type") != "image" or any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        box = _box(item, width, height)
        if box is None or (box[2] - box[0]) * (box[3] - box[1]) < width * height * 0.80:
            continue
        path = asset_dir / Path(str(item.get("src") or "")).name
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
        if image is None:
            continue
        if image.ndim == 3 and image.shape[2] == 4 and float(np.mean(image[:, :, 3] > 32)) < 0.95:
            continue
        candidates.append(item)
    if len(candidates) != 1:
        return 0
    occupied = np.zeros((height, width), np.uint8)
    for item in elements:
        if item.get("type") != "text" or any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        raw = (item.get("metadata") or {}).get("rawOCRBBox")
        box = _clip(raw, width, height) if isinstance(raw, list) and len(raw) == 4 else _box(item, width, height)
        if box is not None:
            _occupy_text_glyphs(source, occupied, box)
    baseline = _border_color(source)
    difference = np.max(np.abs(source.astype(np.int16) - baseline), axis=2)
    candidate = visual_candidate_mask(source)
    unexplained = (difference >= 3) & ~candidate & (occupied == 0)
    if float(np.mean(unexplained)) > 0.01:
        return 0
    with TemporaryDirectory(prefix="imageppt-split-") as staging:
        pieces, stats = extract_residual_objects(source, occupied, Path(staging), project_id, page_index)
    if (len(pieces) < 2 or float(stats["residualObjectizationRate"]) < 0.95
            or any(float(item["width"] * item["height"]) >= width * height * 0.55 for item in pieces)):
        return 0
    target = candidates[0]
    target.setdefault("metadata", {}).update({"suppressed": True, "suppressRender": True,
                                               "splitFromMonolithicImage": True})
    return 1


def _extract_large_flat_inset_panels(source: np.ndarray, active: list[dict], elements: list[dict], occupied: np.ndarray, page_index: int) -> int:
    """Keep a near-page-sized, bounded flat panel as a shape, not a discarded wash."""
    height, width = source.shape[:2]
    border = np.concatenate((source[0], source[-1], source[:, 0], source[:, -1]))
    page_color = np.median(border, axis=0).astype(np.int16)
    contrast = np.max(np.abs(source.astype(np.int16) - page_color), axis=2)
    mask = np.uint8(contrast >= 18) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    created = 0
    for label in range(1, count):
        x, y, w, h, pixels = [int(value) for value in stats[label]]
        box = (x, y, x + w, y + h)
        if w * h < width * height * 0.75 or min(x, y, width - x - w, height - y - h) < 2:
            continue
        if pixels / max(1, w * h) < 0.55:
            continue
        if any(_overlap_of_first(box, _box(item, width, height)) > 0.85
               for item in active if item.get("type") not in {"text", "background"} and _box(item, width, height)):
            continue
        region = source[y:y + h, x:x + w]
        rim_size = max(2, round(min(w, h) * 0.03))
        rim = np.concatenate((region[:rim_size].reshape(-1, 3), region[-rim_size:].reshape(-1, 3),
                              region[:, :rim_size].reshape(-1, 3), region[:, -rim_size:].reshape(-1, 3)))
        color = np.median(rim, axis=0).astype(np.uint8)
        close = np.max(np.abs(region.astype(np.int16) - color.astype(np.int16)), axis=2) <= 10
        rim_close = np.max(np.abs(rim.astype(np.int16) - color.astype(np.int16)), axis=1) <= 10
        if float(np.mean(rim_close)) < 0.85 or float(np.mean(close)) < 0.55:
            continue
        corner_size = max(2, min(w, h) // 50)
        corners = (close[:corner_size, :corner_size], close[:corner_size, -corner_size:],
                   close[-corner_size:, :corner_size], close[-corner_size:, -corner_size:])
        if any(float(np.mean(corner)) < 0.75 for corner in corners):
            continue
        created += 1
        fill = _hex_bgr(color)
        elements.append({"id": f"large_flat_panel_{page_index}_{created}", "type": "rectangle",
                         "x": x, "y": y, "width": w, "height": h, "rotation": 0, "zIndex": 0,
                         "style": {"fill": fill, "strokeWidth": 0, "opacity": 1},
                         "metadata": {"reconstructionStrategy": "native_shape",
                                      "reconstructionStrategySource": "large_inset_panel", "layerRole": "container"}})
        _occupy_shape_color(source, occupied, box, fill)
    return created


def _extract_round_assets(source: np.ndarray, active: list[dict], elements: list[dict], occupied: np.ndarray, asset_dir: Path, project_id: str, page_index: int) -> int:
    """Claim a complete circular badge before its inner marks become shapes."""
    height, width = source.shape[:2]
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    saturated = np.uint8((hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 35)) * 255
    contours, _ = cv2.findContours(saturated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    created = 0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if min(w, h) < 18 or w * h > width * height * 0.12 or max(w / h, h / w) > 1.25:
            continue
        circularity = 4 * np.pi * area / max(1.0, cv2.arcLength(contour, True) ** 2)
        if circularity < 0.70 or area / max(1, w * h) < 0.60:
            continue
        pad = max(2, round(min(w, h) * 0.08))
        x1, y1, x2, y2 = max(0, x - pad), max(0, y - pad), min(width, x + w + pad), min(height, y + h + pad)
        box = (x1, y1, x2, y2)
        if _is_text_glyph_candidate(box, active, width, height):
            continue
        if any(_overlap_of_first(box, _box(item, width, height)) > 0.75 for item in active if item.get("type") not in {"text", "background"} and _box(item, width, height)):
            continue
        if np.mean(occupied[y1:y2, x1:x2] > 0) > 0.1:
            continue
        scale = 4
        alpha_large = np.zeros(((y2 - y1) * scale, (x2 - x1) * scale), np.uint8)
        center = (round((x + w / 2 - x1) * scale), round((y + h / 2 - y1) * scale))
        radius = round((max(w, h) / 2 + pad * 0.5) * scale)
        cv2.circle(alpha_large, center, radius, 255, -1, cv2.LINE_AA)
        alpha = cv2.resize(alpha_large, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
        crop = source[y1:y2, x1:x2]
        path = asset_dir / f"round_visual_page_{page_index}_{created + 1:03d}.png"
        if not cv2.imwrite(str(path), np.dstack((crop, alpha))):
            continue
        created += 1
        elements.append({"id": path.stem, "type": "image", "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "rotation": 0, "zIndex": 1, "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1}, "metadata": {"reconstructionStrategy": "cutout_image", "reconstructionStrategySource": "round_contour", "layerRole": "residual", "preserveWholeAsset": True, "contourQuality": round(float(circularity), 3)}})
        occupied[y1:y2, x1:x2][alpha > 0] = 255
    return created


def _occupy_text_glyphs(source: np.ndarray, occupied: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    region = source[y1:y2, x1:x2]
    # The median represents the local supporting surface for ordinary OCR
    # lines. Large contrasting components are plates/artwork, not glyphs.
    base = np.median(region.reshape(-1, 3), axis=0)
    delta = np.max(np.abs(region.astype(np.float32) - base), axis=2)
    candidate = np.uint8(delta >= 24) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    glyphs = np.zeros(candidate.shape, np.uint8)
    area = candidate.size
    for label in range(1, count):
        _, _, w, h, pixels = [int(value) for value in stats[label]]
        if pixels <= area * 0.18 and w * h <= area * 0.38:
            glyphs[labels == label] = 255
    glyphs = cv2.dilate(glyphs, np.ones((3, 3), np.uint8), iterations=1)
    occupied[y1:y2, x1:x2][glyphs != 0] = 255


def _is_text_shadow_residual(asset: dict, active: list[dict], asset_dir: Path, width: int, height: int) -> bool:
    """Discard faint OCR antialias fragments already owned by editable text."""
    path = asset_dir / Path(str(asset.get("src") or "")).name
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        return False
    alpha = image[:, :, 3] > 32
    count = int(np.count_nonzero(alpha))
    if count < 12 or count / alpha.size >= 0.42:
        return False
    colors = image[:, :, :3][alpha]
    median = np.median(colors, axis=0)
    if float(np.min(median)) < 244 or float(np.max(median) - np.min(median)) > 15:
        return False
    ax, ay = int(asset.get("x") or 0), int(asset.get("y") or 0)
    overlap = np.zeros(alpha.shape, np.bool_)
    has_dark_text = False
    for item in active:
        if item.get("type") != "text":
            continue
        raw = (item.get("metadata") or {}).get("rawOCRBBox")
        box = _clip(raw, width, height) if isinstance(raw, list) and len(raw) == 4 else _box(item, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        left, top, right, bottom = max(ax, x1), max(ay, y1), min(ax + alpha.shape[1], x2), min(ay + alpha.shape[0], y2)
        if right <= left or bottom <= top:
            continue
        color = str((item.get("style") or {}).get("color") or "#111827").lstrip("#")
        try:
            has_dark_text |= len(color) == 6 and max(bytes.fromhex(color)) < 190
        except ValueError:
            pass
        overlap[top - ay:bottom - ay, left - ax:right - ax] = True
    return has_dark_text and float(np.count_nonzero(alpha & overlap)) / count >= 0.95


def layer_objectized_elements(elements: list[dict]) -> None:
    """Keep card fill below its visual details and editable labels above both."""
    containers = [item for item in elements if not (item.get("metadata") or {}).get("pageSurface") and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))
                  and ((item.get("metadata") or {}).get("layerRole") == "container"
                       or item.get("type") in {"rectangle", "roundedRectangle", "ellipse"} and (item.get("style") or {}).get("fill"))]
    visuals = [item for item in elements if item.get("type") == "image"
               and (item.get("metadata") or {}).get("layerRole") != "container"
               and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    texts = [item for item in elements if item.get("type") == "text" and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    for asset in visuals:
        asset_box = _box_unclipped(asset)
        below = [item for item in containers
                 if (item.get("metadata") or {}).get("reconstructionStrategySource") != "colored_text_support"
                 and _intersects(asset_box, _box_unclipped(item))]
        asset["zIndex"] = max([int(item.get("zIndex") or 0) for item in below], default=0) + 1
    # Extracted visual crops own their source pixels; residual crops only fill
    # surrounding gaps and must stay underneath the movable object.
    for planned in visuals:
        if (planned.get("metadata") or {}).get("reconstructionStrategySource") not in {"planner", "round_contour", "merged_contour_owner", "leading_text_icon"}:
            continue
        box = _box_unclipped(planned)
        residuals = [item for item in visuals if item is not planned
                     and (item.get("metadata") or {}).get("layerRole") == "residual"
                     and _intersects(box, _box_unclipped(item))]
        planned["zIndex"] = max(int(planned.get("zIndex") or 0),
                                max([int(item.get("zIndex") or 0) for item in residuals], default=0) + 1)
    # A text support replaces pixels in older residual assets. Keep it above
    # those assets so the editor can select its visible surface, while the
    # editable label remains above the support.
    for support in containers:
        if (support.get("metadata") or {}).get("reconstructionStrategySource") != "colored_text_support":
            continue
        box = _box_unclipped(support)
        overlapping = [item for item in visuals if _intersects(box, _box_unclipped(item))]
        support["zIndex"] = max(int(support.get("zIndex") or 0),
                                max([int(item.get("zIndex") or 0) for item in overlapping], default=0) + 1)
    for text in texts:
        box = _box_unclipped(text)
        below = [item for item in containers + visuals if _intersects(box, _box_unclipped(item))]
        if below:
            text["zIndex"] = max(int(text.get("zIndex") or 0), max(int(item.get("zIndex") or 0) for item in below) + 1)


def _box_unclipped(item: dict) -> tuple[float, float, float, float]:
    x, y = float(item.get("x") or 0), float(item.get("y") or 0)
    return x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)


def _intersects(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    return min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(left[1], right[1])


def _occupy_shape_color(source: np.ndarray, occupied: np.ndarray, box: tuple[int, int, int, int], fill: str) -> None:
    x1, y1, x2, y2 = box
    try:
        rgb = bytes.fromhex(fill.lstrip("#"))
        if len(rgb) != 3:
            raise ValueError
        color = np.frombuffer(rgb[::-1], dtype=np.uint8).astype(np.int16)
    except ValueError:
        occupied[y1:y2, x1:x2] = 255
        return
    region = source[y1:y2, x1:x2]
    matching = np.max(np.abs(region.astype(np.int16) - color), axis=2) <= 18
    occupied[y1:y2, x1:x2][matching] = 255


def _occupy_existing_image(item: dict, occupied: np.ndarray, box: tuple[int, int, int, int], asset_dir: Path) -> None:
    """Reserve rendered pixels, not the transparent part of an image's bbox."""
    x1, y1, x2, y2 = box
    src = str(item.get("src") or "")
    path = Path(src)
    if not path.is_file():
        path = asset_dir / Path(src).name
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
    if image is None:
        return
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = cv2.resize(image[:, :, 3], (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        occupied[y1:y2, x1:x2][alpha > 32] = 255
    else:
        occupied[y1:y2, x1:x2] = 255


def _occupy_existing_stroke(source: np.ndarray, occupied: np.ndarray, box: tuple[int, int, int, int], style: dict) -> None:
    """Claim the connected stroke, leaving unrelated artwork inside its bbox."""
    x1, y1, x2, y2 = box
    try:
        rgb = bytes.fromhex(str(style.get("stroke") or "#17365D").lstrip("#"))
        if len(rgb) != 3:
            return
        color = np.frombuffer(rgb[::-1], dtype=np.uint8).astype(np.int16)
    except ValueError:
        return
    matching = np.uint8(np.max(np.abs(source[y1:y2, x1:x2].astype(np.int16) - color), axis=2) <= 25) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(matching, 8)
    center = (y2 - y1) // 2
    band = max(2, round(float(style.get("strokeWidth") or 2) * 1.5))
    seed = labels[max(0, center - band):min(y2 - y1, center + band + 1)]
    selected = set(np.unique(seed)) - {0}
    for label in selected:
        if label < count and stats[label, cv2.CC_STAT_AREA] >= 3:
            occupied[y1:y2, x1:x2][labels == label] = 255


def _extract_bordered_containers(source: np.ndarray, active: list[dict], elements: list[dict], occupied: np.ndarray, asset_dir: Path, project_id: str, page_index: int) -> tuple[int, int]:
    height, width = source.shape[:2]
    edges = cv2.Canny(source, 45, 130)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 10 or w * h < max(90, width * height * 0.0001) or w * h > width * height * 0.80:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if not 4 <= len(polygon) <= 12 or cv2.contourArea(contour) / max(1, w * h) < 0.68:
            continue
        if np.mean(occupied[y:y + h, x:x + w] > 0) > 0.35:
            continue
        border = edges[y:y + h, x:x + w]
        if not _has_card_corners(np.uint8(cv2.drawContours(np.zeros((h, w), np.uint8), [contour - np.array([[[x, y]]])], -1, 255, -1))):
            continue
        band = max(2, min(w, h) // 18)
        if min(np.mean(border[:band] > 0), np.mean(border[-band:] > 0), np.mean(border[:, :band] > 0), np.mean(border[:, -band:] > 0)) < 0.045:
            continue
        candidates.append((x, y, w, h, contour, len(polygon) > 4))
    candidates.sort(key=lambda value: value[2] * value[3], reverse=True)
    shapes = assets = 0
    selected: list[tuple[int, int, int, int]] = []
    for x, y, w, h, contour, rounded in candidates[:80]:
        box = (x, y, x + w, y + h)
        if _is_text_glyph_candidate(box, active, width, height):
            continue
        if any(_overlap_min(box, prior) > 0.75 for prior in selected):
            continue
        if any(_overlap_of_first(box, _box(item, width, height)) > 0.80 for item in active if item.get("type") != "text" and _box(item, width, height)):
            continue
        members = _module_members(active, box)
        texts = [item for item in members if item.get("type") == "text"]
        crop = source[y:y + h, x:x + w]
        inset = max(2, min(w, h) // 10)
        inner = crop[inset:h - inset, inset:w - inset]
        if inner.size == 0:
            continue
        # Border-only cards are native when their interior is flat. Text and
        # textured interiors use a text-cleaned movable crop instead.
        text_mask = np.zeros((h, w), np.uint8)
        for item in texts:
            raw_box = (item.get("metadata") or {}).get("rawOCRBBox")
            text_box = _clip(raw_box, width, height) if isinstance(raw_box, list) and len(raw_box) == 4 else _box(item, width, height)
            if text_box:
                tx1, ty1, tx2, ty2 = text_box
                cv2.rectangle(text_mask, (max(0, tx1 - x - 2), max(0, ty1 - y - 2)), (min(w - 1, tx2 - x + 2), min(h - 1, ty2 - y + 2)), 255, -1)
        clean_pixels = inner[text_mask[inset:h - inset, inset:w - inset] == 0]
        if len(clean_pixels) < 12:
            continue
        dominant_fill = np.median(clean_pixels, axis=0).astype(np.uint8)
        dominant_fraction = float(np.mean(np.max(np.abs(clean_pixels.astype(np.int16) - dominant_fill.astype(np.int16)), axis=1) <= 12))
        flat = float(np.max(np.std(clean_pixels.astype(np.float32), axis=0))) < 12 or dominant_fraction >= 0.78
        z_index = min([int(item.get("zIndex") or 20) for item in members], default=10) - 1
        identifier = f"white_border_page_{page_index}_{shapes + assets + 1:03d}"
        group_id = _bind_container_module(members, identifier)
        if flat:
            fill_color = dominant_fill
            border_color = np.median(crop[edges[y:y + h, x:x + w] > 0], axis=0).astype(np.uint8)
            shapes += 1
            elements.append({"id": identifier, "type": "roundedRectangle" if rounded else "rectangle", "x": x, "y": y, "width": w, "height": h, "rotation": 0, "zIndex": z_index, "groupId": group_id, "style": {"fill": _hex_bgr(fill_color), "stroke": _hex_bgr(border_color), "strokeWidth": 1, "opacity": 1}, "metadata": {"reconstructionStrategy": "native_shape", "reconstructionStrategySource": "white_objectization", "layerRole": "container", "groupId": group_id, "moduleMemberIds": [item["id"] for item in members if item.get("id")]}})
        else:
            assets += 1
            clean = _clean_container_text(crop, text_mask)
            alpha = np.zeros((h, w), np.uint8)
            local_contour = contour - np.array([[[x, y]]])
            cv2.drawContours(alpha, [local_contour], -1, 255, -1)
            path = asset_dir / f"white_border_page_{page_index}_{assets:03d}.png"
            cv2.imwrite(str(path), np.dstack((clean, alpha)))
            elements.append({"id": path.stem, "type": "image", "x": x, "y": y, "width": w, "height": h, "rotation": 0, "zIndex": z_index, "groupId": group_id, "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1}, "metadata": {"reconstructionStrategy": "cutout_image", "reconstructionStrategySource": "white_objectization", "layerRole": "container", "groupId": group_id, "moduleMemberIds": [item["id"] for item in members if item.get("id")], "textCleaned": bool(texts), "editableTextIds": [item["id"] for item in texts if item.get("id")]}})
        if flat:
            _occupy_shape_color(source, occupied, box, _hex_bgr(fill_color))
            _occupy_shape_color(source, occupied, box, _hex_bgr(border_color))
        else:
            occupied[y:y + h, x:x + w] = 255
        selected.append(box)
    return shapes, assets


def _is_text_glyph_candidate(box: tuple[int, int, int, int], active: list[dict], width: int, height: int) -> bool:
    """Keep small contour fragments of a long OCR line owned by its textbox."""
    area = (box[2] - box[0]) * (box[3] - box[1])
    for item in active:
        if item.get("type") != "text":
            continue
        text_box = _box(item, width, height)
        if text_box is None:
            continue
        text_area = (text_box[2] - text_box[0]) * (text_box[3] - text_box[1])
        if text_area <= 0 or (text_box[2] - text_box[0]) < 3 * (text_box[3] - text_box[1]):
            continue
        if area <= text_area * 0.18 and _overlap_of_first(box, text_box) >= 0.82:
            return True
    return False


def _clean_container_text(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Restore broad OCR boxes from same-row support when the surface is continuous.

    Inpainting a box spanning a colored header and a white card can pull white
    pixels up into the header. Rows with matching support on both sides retain
    their own surface; irregular rows still use the usual local inpainting.
    """
    if not np.any(mask):
        return crop
    cleaned = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
    height, width = mask.shape
    for y in range(height):
        runs = np.flatnonzero(np.diff(np.pad(mask[y] > 0, (1, 1)).astype(np.int8)))
        for left, right in zip(runs[::2], runs[1::2]):
            if left < 3 or right + 3 > width:
                continue
            before = crop[y, left - 3:left].astype(np.float32)
            after = crop[y, right:right + 3].astype(np.float32)
            if max(float(np.ptp(before, axis=0).max()), float(np.ptp(after, axis=0).max())) > 25:
                continue
            start, end = np.median(before, axis=0), np.median(after, axis=0)
            if float(np.max(np.abs(start - end))) > 55:
                continue
            interpolation = np.linspace(start, end, right - left + 2)[1:-1]
            cleaned[y, left:right] = np.uint8(np.round(interpolation))
    return cleaned


def _hex_bgr(color: np.ndarray) -> str:
    return f"#{color[2]:02X}{color[1]:02X}{color[0]:02X}"


def _extract_internal_details(source: np.ndarray, active: list[dict], elements: list[dict], occupied: np.ndarray, asset_dir: Path, project_id: str, page_index: int) -> tuple[int, int]:
    """Recover bounded secondary surfaces relative to their parent, not the page.

    A white inset on a colored bar is invisible to a page-white residual mask.
    Its ownership instead comes from the bar's own fill and geometry.
    """
    height, width = source.shape[:2]
    parents = [item for item in elements if item.get("type") in {"rectangle", "roundedRectangle"} and (item.get("style") or {}).get("fill") and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    shapes = assets = 0
    for parent in parents:
        box = _box(parent, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w < 24 or h < 20 or w * h > width * height * 0.65:
            continue
        try:
            rgb = bytes.fromhex(str(parent["style"]["fill"]).lstrip("#"))
            if len(rgb) != 3:
                continue
            fill = np.frombuffer(rgb[::-1], dtype=np.uint8).astype(np.int16)
        except ValueError:
            continue
        crop = source[y1:y2, x1:x2]
        different = np.uint8(np.max(np.abs(crop.astype(np.int16) - fill), axis=2) >= 28) * 255
        for item in active:
            if item is parent or item.get("type") == "background":
                continue
            child_box = _box(item, width, height)
            if child_box is None or not _intersects(_box_unclipped(item), box):
                continue
            cx1, cy1, cx2, cy2 = child_box
            pad = 3 if item.get("type") == "text" else 2
            different[max(0, cy1 - y1 - pad):min(h, cy2 - y1 + pad), max(0, cx1 - x1 - pad):min(w, cx2 - x1 + pad)] = 0
        different = cv2.morphologyEx(different, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(different, 8)
        for label in range(1, count):
            dx, dy, dw, dh, pixels = [int(value) for value in stats[label]]
            if dx < 2 or dy < 2 or dx + dw > w - 2 or dy + dh > h - 2:
                continue
            if dw < max(8, round(w * 0.12)) or dh < max(5, round(h * 0.10)) or pixels < max(12, round(w * h * 0.002)):
                continue
            detail_box = (x1 + dx, y1 + dy, x1 + dx + dw, y1 + dy + dh)
            if any(_overlap_of_first(detail_box, _box(item, width, height)) > 0.75 for item in elements if item is not parent and item.get("type") not in {"text", "background", "group"} and _box(item, width, height)):
                continue
            region_mask = np.uint8(labels[dy:dy + dh, dx:dx + dw] == label) * 255
            detail_contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not detail_contours or cv2.contourArea(max(detail_contours, key=cv2.contourArea)) / max(1, dw * dh) < 0.85:
                # Round icons and irregular artwork belong to contour/residual
                # extraction; a parent surface must not claim their pixels.
                continue
            region = crop[dy:dy + dh, dx:dx + dw]
            selected = region[region_mask > 0]
            if len(selected) == 0:
                continue
            median = np.median(selected, axis=0).astype(np.uint8)
            color = _hex_bgr(median)
            members = _module_members(active, box)
            group_id = parent.get("groupId") or _bind_container_module(members, str(parent.get("id") or "container"))
            parent["groupId"] = group_id
            parent.setdefault("metadata", {})["groupId"] = group_id
            detail_id = f"internal_detail_page_{page_index}_{shapes + assets + 1:03d}"
            parent["metadata"].setdefault("moduleMemberIds", [item["id"] for item in members if item.get("id")])
            parent["metadata"]["moduleMemberIds"].append(detail_id)
            z_index = int(parent.get("zIndex") or 0) + 1
            metadata = {"reconstructionStrategySource": "internal_surface", "layerRole": "container_detail", "groupId": group_id, "parentId": parent.get("id")}
            solid = pixels / max(1, dw * dh) >= 0.78 and float(np.mean(np.max(np.abs(selected.astype(np.int16) - median.astype(np.int16)), axis=1) <= 9)) >= 0.80
            if solid:
                shapes += 1
                metadata["reconstructionStrategy"] = "native_shape"
                elements.append({"id": detail_id, "type": "rectangle", "x": detail_box[0], "y": detail_box[1], "width": dw, "height": dh, "rotation": 0, "zIndex": z_index, "groupId": group_id, "style": {"fill": color, "stroke": color, "strokeWidth": 0, "opacity": 1}, "metadata": metadata})
            else:
                assets += 1
                metadata["reconstructionStrategy"] = "cutout_image"
                path = asset_dir / f"{detail_id}.png"
                cv2.imwrite(str(path), np.dstack((region, region_mask)))
                elements.append({"id": detail_id, "type": "image", "x": detail_box[0], "y": detail_box[1], "width": dw, "height": dh, "rotation": 0, "zIndex": z_index, "groupId": group_id, "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1}, "metadata": metadata})
            occupied[detail_box[1]:detail_box[3], detail_box[0]:detail_box[2]][region_mask > 0] = 255
            for member in members:
                if member is not parent:
                    member["zIndex"] = max(int(member.get("zIndex") or 0), z_index + 1)
    return shapes, assets


def _extract_flat_containers(source: np.ndarray, active: list[dict], elements: list[dict], occupied: np.ndarray, page_index: int) -> int:
    """Recover bounded flat panels before text masks split their source pixels."""
    height, width = source.shape[:2]
    quantized = ((source.astype(np.uint16) + 8) // 16).clip(0, 15).astype(np.uint8)
    packed = (quantized[:, :, 0].astype(np.uint16) << 8) | (quantized[:, :, 1].astype(np.uint16) << 4) | quantized[:, :, 2].astype(np.uint16)
    counts = np.bincount(packed.ravel(), minlength=4096)
    order = np.argsort(counts)[::-1]
    source_int = source.astype(np.int16)
    candidates: list[tuple[int, int, int, int, str, str]] = []
    minimum_area = max(70, round(width * height * 0.00008))
    for index in order[:40]:
        color = np.array([(index >> 8) & 15, (index >> 4) & 15, index & 15], dtype=np.int16) * 16
        if np.max(255 - color) < 9 or counts[index] < minimum_area:
            continue
        delta = np.max(np.abs(source_int - color), axis=2)
        mask = np.uint8(delta <= 13) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = _trim_surface_protrusions(mask, (x, y, w, h))
            area = w * h
            if w < 10 or h < 8 or area < minimum_area or area > width * height * 0.85:
                continue
            fill_ratio = cv2.contourArea(contour) / area
            if fill_ratio < 0.65:
                continue
            if not _has_card_corners(mask[y:y + h, x:x + w]):
                continue
            region = source[y:y + h, x:x + w]
            matching = delta[y:y + h, x:x + w] <= 13
            if np.mean(matching) < 0.92:
                continue
            median = np.median(region[matching], axis=0).astype(np.uint8)
            fill = f"#{median[2]:02X}{median[1]:02X}{median[0]:02X}"
            radius = _rounded_corner_hint(mask[y:y + h, x:x + w])
            candidates.append((x, y, w, h, fill, "roundedRectangle" if radius else "rectangle"))
    # Quantization can put a pale panel and its near-white page background in
    # the same bucket. Find locally bounded panels by contrast with the page
    # border as well, then require a flat interior and associated foreground.
    page_color = np.median(np.concatenate((source[0], source[-1], source[:, 0], source[:, -1])), axis=0).astype(np.int16)
    contrast = np.max(np.abs(source_int - page_color), axis=2)
    local_mask = np.uint8(contrast >= 5) * 255
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    local_contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in local_contours:
        x, y, w, h = cv2.boundingRect(contour)
        x, y, w, h = _trim_surface_protrusions(local_mask, (x, y, w, h))
        area = w * h
        if w < 10 or h < 8 or area < minimum_area or area > width * height * 0.60:
            continue
        if cv2.contourArea(contour) / area < 0.65:
            continue
        if not _has_card_corners(local_mask[y:y + h, x:x + w]):
            continue
        box = (x, y, x + w, y + h)
        if not _module_members(active, box):
            continue
        region = source[y:y + h, x:x + w]
        median = np.median(region.reshape(-1, 3), axis=0).astype(np.int16)
        if np.max(np.abs(median - page_color)) < 5:
            continue
        if np.mean(np.max(np.abs(region.astype(np.int16) - median), axis=2) <= 5) < 0.92:
            continue
        fill = _hex_bgr(median.astype(np.uint8))
        radius = _rounded_corner_hint(local_mask[y:y + h, x:x + w])
        candidates.append((x, y, w, h, fill, "roundedRectangle" if radius else "rectangle"))
    # Prefer larger panels; nested differently colored blocks remain separate.
    candidates.sort(key=lambda item: item[2] * item[3], reverse=True)
    created = 0
    created_boxes: list[tuple[int, int, int, int]] = []
    for x, y, w, h, fill, kind in candidates:
        box = (x, y, x + w, y + h)
        if np.mean(occupied[y:y + h, x:x + w] > 0) > 0.35:
            continue
        if any(_overlap_of_first(box, text_box) > 0.7 and w * h < (text_box[2] - text_box[0]) * (text_box[3] - text_box[1]) * 0.5
               for item in active if item.get("type") == "text" for text_box in [_box(item, width, height)] if text_box):
            continue
        if any(_box_iou(box, previous) > 0.78 for previous in created_boxes):
            continue
        if any(_overlap_of_first(box, _box(item, width, height)) > 0.80 for item in active if item.get("type") not in {"text", "background"} and _box(item, width, height)):
            continue
        if any(_overlap_min(box, _box(item, width, height)) > 0.80 and item.get("style", {}).get("fill") == fill for item in elements if item.get("metadata", {}).get("reconstructionStrategySource") == "white_objectization" and _box(item, width, height)):
            continue
        members = _module_members(active, box)
        if not members and (w * h > width * height * 0.60 or x == 0 or y == 0 or x + w >= width or y + h >= height):
            continue
        created += 1
        identifier = f"white_container_page_{page_index}_{created:03d}"
        group_id = _bind_container_module(members, identifier)
        z_index = min([int(item.get("zIndex") or 20) for item in members], default=10) - 1
        elements.append({"id": identifier, "type": kind, "x": x, "y": y, "width": w, "height": h, "rotation": 0, "zIndex": z_index, "groupId": group_id, "style": {"fill": fill, "stroke": fill, "strokeWidth": 0, "opacity": 1}, "metadata": {"reconstructionStrategy": "native_shape", "reconstructionStrategySource": "white_objectization", "layerRole": "container", "groupId": group_id, "moduleMemberIds": [item["id"] for item in members if item.get("id")]}})
        _occupy_shape_color(source, occupied, box, fill)
        created_boxes.append(box)
    return created


def _module_members(active: list[dict], box: tuple[int, int, int, int]) -> list[dict]:
    """Foreground with its center on a local plate, excluding page-size owners."""
    area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    return [item for item in active if item.get("type") in {"text", "image", "ellipse", "line", "arrow"}
            and float(item.get("width") or 0) * float(item.get("height") or 0) <= area * 0.85
            and _text_inside(item, box)]


def _bind_container_module(members: list[dict], identifier: str) -> str:
    groups = [str(item.get("groupId") or (item.get("metadata") or {}).get("groupId")) for item in members if item.get("groupId") or (item.get("metadata") or {}).get("groupId")]
    group_id = max(set(groups), key=groups.count) if groups else f"module_{identifier}"
    for item in members:
        if not item.get("groupId") and not (item.get("metadata") or {}).get("groupId"):
            item["groupId"] = group_id
            item.setdefault("metadata", {})["groupId"] = group_id
    return group_id


def _overlap_min(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    overlap = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(0, min(left[3], right[3]) - max(left[1], right[1]))
    return overlap / max(1, min((left[2] - left[0]) * (left[3] - left[1]), (right[2] - right[0]) * (right[3] - right[1])))


def _box_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    overlap = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(0, min(left[3], right[3]) - max(left[1], right[1]))
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return overlap / max(1, left_area + right_area - overlap)


def _overlap_of_first(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    overlap = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(0, min(left[3], right[3]) - max(left[1], right[1]))
    return overlap / max(1, (left[2] - left[0]) * (left[3] - left[1]))


def _text_inside(item: dict, box: tuple[int, int, int, int]) -> bool:
    x = float(item.get("x") or 0) + float(item.get("width") or 0) / 2
    y = float(item.get("y") or 0) + float(item.get("height") or 0) / 2
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _rounded_corner_hint(mask: np.ndarray) -> bool:
    h, w = mask.shape
    size = max(2, min(w, h) // 8)
    return h >= 20 and w >= 20 and np.mean(mask[:size, :size]) < 180 and np.mean(mask[size:2 * size, size:2 * size]) > 180


def _has_card_corners(mask: np.ndarray) -> bool:
    """Reject circular silhouettes before converting a local surface to a box."""
    h, w = mask.shape
    if min(h, w) < 10:
        return False
    band = max(2, round(min(h, w) * 0.12))
    corners = (mask[:band, :band], mask[:band, -band:], mask[-band:, :band], mask[-band:, -band:])
    return sum(float(np.mean(corner > 0)) >= 0.20 for corner in corners) >= 3


def _trim_surface_protrusions(mask: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Remove a touching badge's narrow protrusion from a rectangular plate."""
    x, y, w, h = box
    if min(w, h) < 10:
        return box
    region = mask[y:y + h, x:x + w] > 0
    rows = np.count_nonzero(region, axis=1)
    cols = np.count_nonzero(region, axis=0)
    good_rows = np.flatnonzero(rows >= max(3, np.max(rows) * 0.55))
    good_cols = np.flatnonzero(cols >= max(3, np.max(cols) * 0.55))
    if len(good_rows) < h * 0.55 or len(good_cols) < w * 0.55:
        return box
    left, right = int(good_cols[0]), int(good_cols[-1]) + 1
    top, bottom = int(good_rows[0]), int(good_rows[-1]) + 1
    return x + left, y + top, right - left, bottom - top


def _clip(values: list, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [round(float(value)) for value in values]
    return max(0, min(width, x1)), max(0, min(height, y1)), max(0, min(width, x2)), max(0, min(height, y2))


def _box(item: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    x = float(item.get("x") or 0)
    y = float(item.get("y") or 0)
    box = _clip([x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)], width, height)
    return box if box[2] > box[0] and box[3] > box[1] else None
