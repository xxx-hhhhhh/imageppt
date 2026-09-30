"""Extract unowned source detail as bounded, movable transparent image assets."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extract_residual_objects(source: np.ndarray, occupied: np.ndarray, asset_dir: Path, project_id: str, page_index: int) -> tuple[list[dict], dict]:
    height, width = source.shape[:2]
    baseline = _border_color(source)
    difference = np.max(np.abs(source.astype(np.int16) - baseline), axis=2)
    # A pale label/card may differ from the page by only a few levels. Keep
    # bounded low-contrast regions; page-sized washes are rejected below.
    raw = np.uint8((difference >= 5) & (occupied == 0)) * 255
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
        potential_area += pixels
        values = source[labels == label]
        color = np.median(values, axis=0) if len(values) else baseline
        components.append({"label": label, "box": (x, y, x + w, y + h), "pixels": pixels, "color": color})
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
        local_labels = labels[y1:y2, x1:x2]
        selected = np.isin(local_labels, [components[index]["label"] for index in group]).astype(np.uint8) * 255
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
            union_area = (max(a[2], b[2]) - min(a[0], b[0])) * (max(a[3], b[3]) - min(a[1], b[1]))
            if union_area > width * height * 0.5:
                continue
            parent[root(other)] = root(index)
    grouped: dict[int, list[int]] = {}
    for index in range(len(components)):
        grouped.setdefault(root(index), []).append(index)
    return list(grouped.values())
