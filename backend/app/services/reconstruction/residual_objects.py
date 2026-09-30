"""Extract unowned source detail as bounded, movable transparent image assets."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extract_residual_objects(source: np.ndarray, occupied: np.ndarray, asset_dir: Path, project_id: str, page_index: int) -> tuple[list[dict], dict]:
    height, width = source.shape[:2]
    baseline = _border_color(source)
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
        if pixels < minimum or ((w < 2 or h < 2) and not is_meaningful_stroke(w, h, pixels, width, height)):
            continue
        if _page_environment((x, y, x + w, y + h), pixels, width, height):
            continue
        local = np.uint8(labels[y:y + h, x:x + w] == label)
        if w * h <= width * height * 0.25 and x > 0 and y > 0 and x + w < width and y + h < height and _card_like_outline(local):
            local = _fill_enclosed_surface(local)
        for px, py, piece in _split_component(local, x, y, width, height):
            piece_h, piece_w = piece.shape
            piece_pixels = int(np.count_nonzero(piece))
            if piece_pixels < minimum:
                continue
            potential_area += piece_pixels
            values = source[py:py + piece_h, px:px + piece_w][piece != 0]
            color = np.median(values, axis=0) if len(values) else baseline
            components.append({"mask": piece, "box": (px, py, px + piece_w, py + piece_h), "pixels": piece_pixels, "color": color})
    components.sort(key=lambda item: item["pixels"], reverse=True)
    components = components[:300]
    groups = _group_components(components, width, height)
    groups.sort(key=lambda group: sum(components[index]["pixels"] for index in group), reverse=True)
    assets: list[dict] = []
    covered = 0
    max_objects = 150
    asset_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        x1 = min(components[index]["box"][0] for index in group)
        y1 = min(components[index]["box"][1] for index in group)
        x2 = max(components[index]["box"][2] for index in group)
        y2 = max(components[index]["box"][3] for index in group)
        pixels = sum(components[index]["pixels"] for index in group)
        # A diffuse page-sized component is environmental background, not a
        # selectable visual object. Do not repackage the slide as one image.
        if _page_environment((x1, y1, x2, y2), pixels, width, height):
            continue
        if len(assets) >= max_objects:
            break
        pad = 2
        x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)
        selected = np.zeros((y2 - y1, x2 - x1), np.uint8)
        for index in group:
            component = components[index]
            cx1, cy1, cx2, cy2 = component["box"]
            selected[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1][component["mask"] != 0] = 255
        alpha = cv2.dilate(selected, np.ones((3, 3), np.uint8), iterations=1)
        alpha[occupied[y1:y2, x1:x2] != 0] = 0
        if np.count_nonzero(alpha) < minimum:
            continue
        path = asset_dir / f"residual_page_{page_index}_{len(assets) + 1:03d}.png"
        cv2.imwrite(str(path), np.dstack((source[y1:y2, x1:x2], alpha)))
        asset_id = path.stem
        assets.append({"id": asset_id, "type": "image", "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "rotation": 0, "zIndex": 1, "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1}, "metadata": {"reconstructionStrategy": "cutout_image", "reconstructionStrategySource": "residual_detection", "layerRole": "residual", "preserveWholeAsset": True, "doNotVectorize": True, "sourcePixelArea": pixels}})
        covered += pixels
    return assets, {"residualObjectsCount": len(assets), "residualCoverageArea": covered, "residualCandidateArea": potential_area, "residualObjectizationRate": round(covered / potential_area, 4) if potential_area else 1.0}


def visual_candidate_mask(source: np.ndarray) -> np.ndarray:
    """Find visual content without merging a subtly tinted page into one region."""
    baseline = _border_color(source)
    difference = np.max(np.abs(source.astype(np.int16) - baseline), axis=2)
    candidate = difference >= 5
    if float(np.mean(candidate)) < 0.70:
        return candidate
    # A near-white gradient can differ from the sampled border by five levels
    # across most of a page. Preserve chromatic and dark artwork while dropping
    # that diffuse wash. Bounded pale surfaces remain detectable on simpler pages.
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    background_hsv = cv2.cvtColor(np.uint8([[baseline.astype(np.uint8)]]), cv2.COLOR_BGR2HSV)[0, 0]
    if int(background_hsv[2]) < 235 or int(background_hsv[1]) > 25:
        return candidate
    saturation_floor = max(6, int(background_hsv[1]) + 2)
    dark_ceiling = min(235, int(background_hsv[2]) - 16)
    return candidate & ((hsv[:, :, 1] >= saturation_floor) | (hsv[:, :, 2] <= dark_ceiling))


def _page_environment(box: tuple[int, int, int, int], pixels: int, width: int, height: int) -> bool:
    """Only broad, dense, edge-reaching washes qualify as page environment.

    Large maps and artwork can occupy most of a slide while remaining a
    bounded, irregular visual object. Their area alone must not erase them.
    """
    x1, y1, x2, y2 = box
    box_area = max(1, (x2 - x1) * (y2 - y1))
    margin_x, margin_y = max(5, round(width * 0.03)), max(5, round(height * 0.03))
    near_all_edges = x1 <= margin_x and y1 <= margin_y and x2 >= width - margin_x and y2 >= height - margin_y
    return near_all_edges and box_area >= width * height * 0.80 and pixels / box_area >= 0.82


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
    for index, first in enumerate(components):
        for other in range(index + 1, len(components)):
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
                     depth: int = 0) -> list[tuple[int, int, np.ndarray]]:
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
        return [(x, y, mask)]
    _, axis, cut = best
    if axis == 0:
        return _split_component(mask[:cut], x, y, page_width, page_height, depth + 1) + _split_component(mask[cut:], x, y + cut, page_width, page_height, depth + 1)
    return _split_component(mask[:, :cut], x, y, page_width, page_height, depth + 1) + _split_component(mask[:, cut:], x + cut, y, page_width, page_height, depth + 1)


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
