from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.pptx.renderer import _path_from_src


def separate_foreground(background_path: Path, elements: list[dict]) -> int:
    """Remove independently rendered pixels from the slide's background layer.

    The visual asset is left unchanged. Its position therefore renders exactly
    as before, while moving it no longer reveals a second copy underneath.
    """
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(background_path)
    height, width = background.shape[:2]
    regions: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for element in elements:
        metadata = element.get("metadata") or {}
        if element.get("type") == "background" or any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        strategy = metadata.get("reconstructionStrategy")
        independent_image = element.get("type") == "image" and (metadata.get("preserveWholeAsset") or strategy == "cutout_image")
        planned_shape = strategy == "native_shape" and metadata.get("reconstructionStrategySource") == "planner"
        if not independent_image and not planned_shape:
            continue
        x1, y1 = int(round(float(element.get("x") or 0))), int(round(float(element.get("y") or 0)))
        x2 = x1 + int(round(float(element.get("width") or 0)))
        y2 = y1 + int(round(float(element.get("height") or 0)))
        box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if (box[2] - box[0]) * (box[3] - box[1]) > width * height * 0.65:
            continue
        owned = np.full((box[3] - box[1], box[2] - box[0]), 255, dtype=np.uint8)
        if independent_image:
            path = _path_from_src(element.get("src"))
            asset = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path and path.is_file() else None
            if asset is not None and asset.ndim == 3 and asset.shape[2] == 4:
                owned = cv2.resize(asset[:, :, 3], (owned.shape[1], owned.shape[0]), interpolation=cv2.INTER_LINEAR)
                owned = np.where(owned >= 128, 255, 0).astype(np.uint8)
        if not np.any(owned):
            continue
        regions.append((box, owned))
        metadata["backgroundSeparated"] = True
        metadata["layerRole"] = "cutout_image" if independent_image else "native_shape"
    if not regions:
        return 0
    mask = np.zeros((height, width), dtype=np.uint8)
    for (x1, y1, x2, y2), owned in regions:
        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], owned)
    # Telea handles small icons well. Large photos/charts get a smooth fill
    # sampled outside the owned region so the operation stays bounded.
    small = np.zeros_like(mask)
    for (x1, y1, x2, y2), owned in regions:
        if (x2 - x1) * (y2 - y1) <= width * height * 0.025:
            small[y1:y2, x1:x2] = np.maximum(small[y1:y2, x1:x2], owned)
        else:
            ring = _outer_ring(background, mask, (x1, y1, x2, y2))
            if ring.size:
                patch = background[y1:y2, x1:x2]
                patch[owned > 0] = np.median(ring, axis=0).astype(np.uint8)
    if np.any(small):
        background = cv2.inpaint(background, small, 4, cv2.INPAINT_TELEA)
    cv2.imwrite(str(background_path), background)
    return len(regions)


def _outer_ring(image: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    height, width = mask.shape
    pad = max(6, min(24, min(x2 - x1, y2 - y1) // 6))
    left, top, right, bottom = max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)
    surround = image[top:bottom, left:right]
    available = mask[top:bottom, left:right] == 0
    return surround[available] if np.any(available) else np.empty((0, 3), dtype=np.uint8)
