"""Give an extracted visual one owner across planned and residual assets."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.services.pptx.renderer import _path_from_src


def transfer_planned_visual_pixels(layout: dict, asset_dir: Path, project_id: str) -> dict[str, int]:
    """Remove planned image detail from older residual/plate crops, copy on write.

    The planned crop keeps its pixels. Pale support around its visible detail
    stays in the lower asset so moving the visual does not remove the card.
    """
    active = [item for item in layout.get("elements", []) if item.get("type") == "image"
              and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    planned = [item for item in active if (item.get("metadata") or {}).get("reconstructionStrategySource") == "planner"]
    lower = [item for item in active if (item.get("metadata") or {}).get("reconstructionStrategySource")
             in {"residual_detection", "objectization_qa"}]
    staged: dict[str, tuple[dict, np.ndarray, Path]] = {}
    changed_ids: set[str] = set()
    claimed = 0
    for owner in planned:
        owner_path = _asset_path(owner, asset_dir)
        image = cv2.imread(str(owner_path), cv2.IMREAD_UNCHANGED) if owner_path.is_file() else None
        if image is None or image.ndim != 3:
            continue
        claim = _detail_mask(image)
        if not np.any(claim):
            continue
        ox, oy, ow, oh = _box(owner)
        if min(ow, oh) <= 0:
            continue
        claim = cv2.resize(claim, (ow, oh), interpolation=cv2.INTER_NEAREST)
        for item in lower:
            ix, iy, iw, ih = _box(item)
            left, top, right, bottom = max(ox, ix), max(oy, iy), min(ox + ow, ix + iw), min(oy + oh, iy + ih)
            if right <= left or bottom <= top:
                continue
            key = str(item.get("id"))
            if key not in staged:
                path = _asset_path(item, asset_dir)
                prior = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
                if prior is None or prior.ndim != 3:
                    continue
                if prior.shape[2] == 3:
                    prior = np.dstack((prior, np.full(prior.shape[:2], 255, np.uint8)))
                staged[key] = (item, prior, path)
            _, rgba, _ = staged[key]
            local = np.zeros((ih, iw), np.uint8)
            local[top - iy:bottom - iy, left - ix:right - ix] = claim[top - oy:bottom - oy, left - ox:right - ox]
            local = cv2.resize(local, (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_NEAREST)
            replaced = (local > 0) & (rgba[:, :, 3] > 0)
            claimed += int(np.count_nonzero(replaced))
            if np.any(replaced):
                changed_ids.add(key)
            rgba[:, :, 3][replaced] = 0
    replacements: list[tuple[dict, Path]] = []
    for key, (item, rgba, old_path) in staged.items():
        if key not in changed_ids:
            continue
        new_path = asset_dir / f"{Path(str(item.get('id') or old_path.stem)).stem}_without_planned_{uuid4().hex[:8]}.png"
        if not cv2.imwrite(str(new_path), rgba):
            for _, candidate in replacements:
                candidate.unlink(missing_ok=True)
            return {"plannedVisualPixelsClearedFromOtherAssets": 0, "trimmedOverlappingAssets": 0}
        replacements.append((item, new_path))
    for item, new_path in replacements:
        item["src"] = f"/media/assets/{project_id}/{new_path.name}"
        item.setdefault("metadata", {})["plannedVisualPixelsRemoved"] = True
    return {"plannedVisualPixelsClearedFromOtherAssets": claimed, "trimmedOverlappingAssets": len(replacements)}


def count_duplicate_planned_visual_pixels(layout: dict, asset_dir: Path) -> int:
    """Count planned visual pixels still visible in residual or QA plate assets."""
    active = [item for item in layout.get("elements", []) if item.get("type") == "image"
              and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    total = 0
    for owner in active:
        if (owner.get("metadata") or {}).get("reconstructionStrategySource") != "planner":
            continue
        image = cv2.imread(str(_asset_path(owner, asset_dir)), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3:
            continue
        ox, oy, ow, oh = _box(owner)
        if min(ow, oh) <= 0:
            continue
        claim = cv2.resize(_detail_mask(image), (ow, oh), interpolation=cv2.INTER_NEAREST)
        for other in active:
            if (other.get("metadata") or {}).get("reconstructionStrategySource") not in {"residual_detection", "objectization_qa"}:
                continue
            ix, iy, iw, ih = _box(other)
            left, top, right, bottom = max(ox, ix), max(oy, iy), min(ox + ow, ix + iw), min(oy + oh, iy + ih)
            if right <= left or bottom <= top:
                continue
            lower = cv2.imread(str(_asset_path(other, asset_dir)), cv2.IMREAD_UNCHANGED)
            if lower is None or lower.ndim != 3:
                continue
            alpha = lower[:, :, 3] if lower.shape[2] == 4 else np.full(lower.shape[:2], 255, np.uint8)
            local = np.zeros((ih, iw), np.uint8)
            local[top - iy:bottom - iy, left - ix:right - ix] = claim[top - oy:bottom - oy, left - ox:right - ox]
            local = cv2.resize(local, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_NEAREST)
            total += int(np.count_nonzero((local > 0) & (alpha > 32)))
    return total


def _detail_mask(image: np.ndarray) -> np.ndarray:
    color = image[:, :, :3].astype(np.int16)
    # Near-white pixels are usually the card surface around a satellite,
    # chart, or illustration. Their source support remains independently owned.
    detail = np.max(255 - color, axis=2) >= 18
    detail = cv2.dilate(np.uint8(detail) * 255, np.ones((3, 3), np.uint8), iterations=1)
    if image.shape[2] == 4:
        detail[image[:, :, 3] <= 32] = 0
    return detail


def _box(item: dict) -> tuple[int, int, int, int]:
    return tuple(round(float(item.get(key) or 0)) for key in ("x", "y", "width", "height"))


def _asset_path(item: dict, asset_dir: Path) -> Path:
    local = asset_dir / Path(str(item.get("src") or "")).name
    return local if local.is_file() else (_path_from_src(item.get("src")) or local)
