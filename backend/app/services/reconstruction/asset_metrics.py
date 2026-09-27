from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def measure_movable_assets(source_path: Path, background_path: Path, layout: dict, plan: dict) -> dict[str, int | float]:
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
    covered = sum(any(_overlap_fraction(candidate, _box(asset)) >= 0.8 for asset in assets) for candidate in candidates)
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
    return {"movableAssetCount": len(assets), "backgroundResidualCount": residual, "movableVisualCoverage": round(covered / len(candidates), 4) if candidates else 1.0}


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
