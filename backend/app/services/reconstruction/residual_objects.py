"""Extract unowned source detail as bounded, movable transparent image assets."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extract_residual_objects(source: np.ndarray, occupied: np.ndarray, asset_dir: Path, project_id: str,
                             page_index: int, *, asset_prefix: str | None = None) -> tuple[list[dict], dict]:
    height, width = source.shape[:2]
    baseline = _border_color(source)
    detailed_page = dense_visual_artwork(source)
    # A pale label/card may differ from the page by only a few levels. Keep
    # bounded low-contrast regions; page-sized washes are rejected below.
    raw = np.uint8(visual_candidate_mask(source) & (occupied == 0)) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    raw[occupied != 0] = 0
    potential_area = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    minimum = max(12, round(width * height * 0.000015))
    components = []
    for label in range(1, count):
        x, y, w, h, pixels = [int(value) for value in stats[label]]
        small_visual = _small_solid_decoration(source, labels[y:y + h, x:x + w] == label,
                                               (x, y, w, h), baseline)
        if (pixels < minimum and not small_visual) or ((w < 2 or h < 2) and not is_meaningful_stroke(w, h, pixels, width, height)):
            continue
        if _page_environment((x, y, x + w, y + h), pixels, width, height, detailed_page):
            continue
        local = np.uint8(labels[y:y + h, x:x + w] == label)
        if w * h <= width * height * 0.25 and x > 0 and y > 0 and x + w < width and y + h < height and _card_like_outline(local):
            local = _fill_enclosed_surface(local)
        for px, py, piece in _split_component(local, x, y, width, height, fallback_tiling=detailed_page):
            piece_h, piece_w = piece.shape
            piece_pixels = int(np.count_nonzero(piece))
            if piece_pixels < minimum and not _small_solid_decoration(source, piece != 0,
                                                                       (px, py, piece_w, piece_h), baseline):
                continue
            potential_area += piece_pixels
            values = source[py:py + piece_h, px:px + piece_w][piece != 0]
            color = np.median(values, axis=0) if len(values) else baseline
            components.append({"mask": piece, "box": (px, py, px + piece_w, py + piece_h), "pixels": piece_pixels, "color": color})
    components.sort(key=lambda item: item["pixels"], reverse=True)
    groups = _group_components(components, width, height)
    groups.sort(key=lambda group: sum(components[index]["pixels"] for index in group), reverse=True)
    assets: list[dict] = []
    covered = 0
    asset_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        x1 = min(components[index]["box"][0] for index in group)
        y1 = min(components[index]["box"][1] for index in group)
        x2 = max(components[index]["box"][2] for index in group)
        y2 = max(components[index]["box"][3] for index in group)
        pixels = sum(components[index]["pixels"] for index in group)
        # A diffuse page-sized component is environmental background, not a
        # selectable visual object. Do not repackage the slide as one image.
        if _page_environment((x1, y1, x2, y2), pixels, width, height, detailed_page):
            continue
        pad = 2
        x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)
        selected = np.zeros((y2 - y1, x2 - x1), np.uint8)
        for index in group:
            component = components[index]
            cx1, cy1, cx2, cy2 = component["box"]
            selected[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1][component["mask"] != 0] = 255
        alpha = cv2.dilate(selected, np.ones((3, 3), np.uint8), iterations=1)
        alpha[occupied[y1:y2, x1:x2] != 0] = 0
        if np.count_nonzero(alpha) < minimum and pixels < 9:
            continue
        path = asset_dir / f"{asset_prefix or f'residual_page_{page_index}'}_{len(assets) + 1:03d}.png"
        cv2.imwrite(str(path), np.dstack((source[y1:y2, x1:x2], alpha)))
        asset_id = path.stem
        assets.append({"id": asset_id, "type": "image", "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "rotation": 0, "zIndex": 1, "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1}, "metadata": {"reconstructionStrategy": "cutout_image", "reconstructionStrategySource": "residual_detection", "layerRole": "residual", "preserveWholeAsset": True, "doNotVectorize": True, "sourcePixelArea": pixels}})
        covered += pixels
    return assets, {"residualObjectsCount": len(assets), "residualCoverageArea": covered, "residualCandidateArea": potential_area, "residualObjectizationRate": round(covered / potential_area, 4) if potential_area else 1.0}


def _small_solid_decoration(source: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int],
                            baseline: np.ndarray) -> bool:
    x, y, w, h = box
    pixels = int(np.count_nonzero(mask))
    if w < 3 or h < 3 or pixels < 9 or pixels / max(1, w * h) < 0.75:
        return False
    colors = source[y:y + h, x:x + w][mask]
    median = np.median(colors, axis=0)
    hsv = cv2.cvtColor(np.uint8([[median]]), cv2.COLOR_BGR2HSV)[0, 0]
    return (int(hsv[1]) >= 45 and float(np.max(np.abs(median - baseline))) >= 40
            and float(np.mean(np.max(np.abs(colors.astype(np.float32) - median), axis=1) <= 12)) >= 0.8)


def visual_candidate_mask(source: np.ndarray) -> np.ndarray:
    """Find visual content without merging a subtly tinted page into one region."""
    baseline = _border_color(source)
    difference = np.max(np.abs(source.astype(np.int16) - baseline), axis=2)
    candidate = difference >= 5
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    background_hsv = cv2.cvtColor(np.uint8([[baseline.astype(np.uint8)]]), cv2.COLOR_BGR2HSV)[0, 0]
    pale_background = int(background_hsv[2]) >= 235 and int(background_hsv[1]) <= 25
    saturation_floor = max(6, int(background_hsv[1]) + 2)
    diffuse_candidate = float(np.mean(candidate)) >= 0.70
    neutral_plates = (_neutral_local_surfaces(hsv, saturation_floor, min_contrast=5 if diffuse_candidate else 1)
                      if pale_background else np.zeros(candidate.shape, np.bool_))
    if not diffuse_candidate:
        return candidate | neutral_plates
    # A near-white gradient can differ from the sampled border by five levels
    # across most of a page. Preserve chromatic and dark artwork while dropping
    # that diffuse wash. Bounded pale surfaces remain detectable on simpler pages.
    if not pale_background:
        return candidate
    dark_ceiling = min(235, int(background_hsv[2]) - 16)
    chromatic_or_dark = (hsv[:, :, 1] >= saturation_floor) | (hsv[:, :, 2] <= dark_ceiling)
    return (candidate & chromatic_or_dark) | neutral_plates


def _neutral_local_surfaces(hsv: np.ndarray, saturation_floor: int, *, min_contrast: int = 5) -> np.ndarray:
    """Recover bounded pale gray plates without claiming a page-wide wash."""
    height, width = hsv.shape[:2]
    band = max(4, round(width * 0.04))
    edge_values = np.concatenate((hsv[:, :band, 2], hsv[:, -band:, 2]), axis=1)
    row_value = np.median(edge_values, axis=1).astype(np.int16)
    darker = row_value[:, None] - hsv[:, :, 2].astype(np.int16) >= min_contrast
    mask = np.uint8((hsv[:, :, 1] < saturation_floor) & darker) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    accepted = np.zeros((height, width), np.bool_)
    minimum = max(60, round(width * height * 0.00008))
    for label in range(1, count):
        x, y, w, h, pixels = [int(value) for value in stats[label]]
        area = w * h
        if pixels < minimum or w < 10 or h < 8 or area > width * height * 0.4:
            continue
        # A bounded title plate may touch a page edge. Reject only components
        # spanning the whole page; the area cap above excludes broad washes.
        margin_x, margin_y = max(2, round(width * 0.02)), max(2, round(height * 0.02))
        if x < margin_x and y < margin_y and x + w > width - margin_x and y + h > height - margin_y:
            continue
        if pixels / max(1, area) < 0.72:
            continue
        accepted[labels == label] = True
    return accepted


def dense_visual_artwork(source: np.ndarray) -> bool:
    """Distinguish a detailed full-slide visual from a smooth page wash."""
    candidate = visual_candidate_mask(source)
    candidate_fraction = float(np.mean(candidate))
    if candidate_fraction < 0.85:
        return False
    edges = cv2.Canny(source, 50, 130)
    color_variation = float(np.max(np.std(source[candidate].astype(np.float32), axis=0)))
    return float(np.mean(edges != 0)) >= 0.025 or color_variation >= 18


def _page_environment(box: tuple[int, int, int, int], pixels: int, width: int, height: int,
                      detailed_page: bool = False) -> bool:
    """Only broad, dense, edge-reaching washes qualify as page environment.

    Large maps and artwork can occupy most of a slide while remaining a
    bounded, irregular visual object. Their area alone must not erase them.
    """
    x1, y1, x2, y2 = box
    box_area = max(1, (x2 - x1) * (y2 - y1))
    margin_x, margin_y = max(5, round(width * 0.03)), max(5, round(height * 0.03))
    near_all_edges = x1 <= margin_x and y1 <= margin_y and x2 >= width - margin_x and y2 >= height - margin_y
    # A dense irregular silhouette can touch every margin while still being a
    # meaningful movable object. Reserve this exclusion for nearly solid fills.
    return not detailed_page and near_all_edges and box_area >= width * height * 0.80 and pixels / box_area >= 0.97


def is_meaningful_stroke(w: int, h: int, pixels: int, width: int, height: int) -> bool:
    """Keep long narrow accents while rejecting isolated single-pixel noise."""
    short, long = min(w, h), max(w, h)
    return short <= 3 and long >= max(24, round(max(width, height) * 0.06)) and long / max(1, short) >= 8 and pixels >= max(12, round(long * 0.55))


def _border_color(source: np.ndarray) -> np.ndarray:
    height, width = source.shape[:2]
    band = max(2, min(height, width) // 100)
    border = np.concatenate((source[:band].reshape(-1, 3), source[-band:].reshape(-1, 3), source[:, :band].reshape(-1, 3), source[:, -band:].reshape(-1, 3)))
    colors = ((border.astype(np.uint16) + 8) // 16).clip(0, 15)
    packed = (colors[:, 0] << 8) | (colors[:, 1] << 4) | colors[:, 2]
    dominant = int(np.bincount(packed.astype(np.int32), minlength=4096).argmax())
    selected = border[packed == dominant]
    return np.median(selected, axis=0).astype(np.int16)


def _group_components(components: list[dict], width: int, height: int) -> list[list[int]]:
    parent = list(range(len(components)))
    group_boxes = [item["box"] for item in components]

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    gap_limit = max(5, round(min(width, height) * 0.018))
    cell_size = max(32, gap_limit * 4)
    spatial_cells: dict[tuple[int, int], list[int]] = {}
    for index, component in enumerate(components):
        x1, y1, x2, y2 = component["box"]
        for cell_y in range(y1 // cell_size, (y2 - 1) // cell_size + 1):
            for cell_x in range(x1 // cell_size, (x2 - 1) // cell_size + 1):
                spatial_cells.setdefault((cell_x, cell_y), []).append(index)
    for index, first in enumerate(components):
        x1, y1, x2, y2 = first["box"]
        nearby: set[int] = set()
        for cell_y in range(max(0, y1 - gap_limit) // cell_size, (y2 + gap_limit - 1) // cell_size + 1):
            for cell_x in range(max(0, x1 - gap_limit) // cell_size, (x2 + gap_limit - 1) // cell_size + 1):
                nearby.update(spatial_cells.get((cell_x, cell_y), ()))
        for other in sorted(candidate for candidate in nearby if candidate > index):
            second = components[other]
            a, b = first["box"], second["box"]
            dx = max(0, a[0] - b[2], b[0] - a[2])
            dy = max(0, a[1] - b[3], b[1] - a[3])
            if dx > gap_limit or dy > gap_limit:
                continue
            if float(np.max(np.abs(first["color"] - second["color"]))) > 48:
                continue
            first_root, second_root = root(index), root(other)
            if first_root == second_root:
                continue
            a, b = group_boxes[first_root], group_boxes[second_root]
            merged = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
            union_area = (merged[2] - merged[0]) * (merged[3] - merged[1])
            if union_area > width * height * 0.12:
                continue
            parent[second_root] = first_root
            group_boxes[first_root] = merged
    grouped: dict[int, list[int]] = {}
    for index in range(len(components)):
        grouped.setdefault(root(index), []).append(index)
    return list(grouped.values())


def _split_component(mask: np.ndarray, x: int, y: int, page_width: int, page_height: int,
                     depth: int = 0, *, fallback_tiling: bool = False) -> list[tuple[int, int, np.ndarray]]:
    """Split connected modules at long sparse seams without losing pixels."""
    ys, xs = np.where(mask != 0)
    if len(xs) == 0:
        return []
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    mask = mask[top:bottom, left:right]
    x, y = x + left, y + top
    h, w = mask.shape
    if depth >= 3 or w * h < page_width * page_height * 0.08:
        return [(x, y, mask)]
    pixels = int(np.count_nonzero(mask))
    best: tuple[float, int, int] | None = None
    for axis, span in ((0, h), (1, w)):
        if span < max(80, round((page_height if axis == 0 else page_width) * 0.22)):
            continue
        counts = np.count_nonzero(mask, axis=1 - axis)
        positive = counts[counts > 0]
        threshold = max(3, round(float(np.percentile(positive, 85)) * 0.26))
        low = counts <= threshold
        starts = np.flatnonzero(low & ~np.r_[False, low[:-1]])
        ends = np.flatnonzero(low & ~np.r_[low[1:], False]) + 1
        for start, end in zip(starts, ends):
            if end - start < max(4, round(span * 0.012)):
                continue
            cut = (int(start) + int(end)) // 2
            if cut < span * 0.15 or span - cut < span * 0.15:
                continue
            before = int(np.count_nonzero(mask[:cut] if axis == 0 else mask[:, :cut]))
            after = pixels - before
            if min(before, after) < max(80, pixels * 0.13):
                continue
            score = (end - start) * min(before, after) / pixels
            if best is None or score > best[0]:
                best = (score, axis, cut)
    if best is None:
        page_area = page_width * page_height
        minimum_fraction = 0.75 if depth == 0 else 0.30
        if (not fallback_tiling or depth >= 2 or w * h < page_area * minimum_fraction
                or pixels < page_area * (0.35 if depth == 0 else 0.15)):
            return [(x, y, mask)]
        axis = 1 if (depth == 0 and w >= h) or (depth == 1 and w < h) else 0
        best = (0.0, axis, (w if axis == 1 else h) // 2)
    _, axis, cut = best
    if axis == 0:
        return _split_component(mask[:cut], x, y, page_width, page_height, depth + 1, fallback_tiling=fallback_tiling) + _split_component(mask[cut:], x, y + cut, page_width, page_height, depth + 1, fallback_tiling=fallback_tiling)
    return _split_component(mask[:, :cut], x, y, page_width, page_height, depth + 1, fallback_tiling=fallback_tiling) + _split_component(mask[:, cut:], x + cut, y, page_width, page_height, depth + 1, fallback_tiling=fallback_tiling)


def _fill_enclosed_surface(mask: np.ndarray) -> np.ndarray:
    """Keep pale interiors enclosed by an extracted card or badge outline."""
    padded = np.pad(np.uint8(mask != 0) * 255, 1)
    inverse = cv2.bitwise_not(padded)
    flood_mask = np.zeros((inverse.shape[0] + 2, inverse.shape[1] + 2), np.uint8)
    cv2.floodFill(inverse, flood_mask, (0, 0), 0)
    holes = inverse[1:-1, 1:-1] != 0
    return np.uint8((mask != 0) | holes)


def _card_like_outline(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if min(h, w) < 20:
        return False
    band = max(2, round(min(h, w) * 0.12))
    corners = (mask[:band, :band], mask[:band, -band:], mask[-band:, :band], mask[-band:, -band:])
    return sum(float(np.mean(corner != 0)) >= 0.20 for corner in corners) >= 3
