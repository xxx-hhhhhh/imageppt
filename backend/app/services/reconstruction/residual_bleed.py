"""Restore source pale pixels damaged inside movable residual assets."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.reconstruction.revision_integrity import asset_path


def repair_dark_residual_bleed(source_path: Path, layout: dict, issues: list[dict],
                               asset_dir: Path, project_id: str, revision_round: int) -> list[str]:
    """Replace only proven dark additions, writing new assets before switching src."""
    boxes = [issue.get("bbox") for issue in issues
             if issue.get("reason") == "dark_residual_over_pale_source"
             and isinstance(issue.get("bbox"), list) and len(issue["bbox"]) == 4]
    if not boxes:
        return []
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        return []
    source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    height, width = source.shape[:2]
    changed: list[str] = []
    root = asset_dir.parent
    for item in layout.get("elements", []):
        meta = item.get("metadata") or {}
        if (item.get("type") != "image" or meta.get("layerRole") != "residual"
                or any(meta.get(key) for key in ("suppressed", "suppressRender", "ownedBy"))):
            continue
        x, y = float(item.get("x") or 0), float(item.get("y") or 0)
        w, h = float(item.get("width") or 0), float(item.get("height") or 0)
        if w <= 0 or h <= 0 or not any(x < box[2] and x + w > box[0] and y < box[3] and y + h > box[1]
                                      for box in boxes):
            continue
        path = asset_path(root, item.get("src"))
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path and path.is_file() else None
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            continue
        rows, columns = image.shape[:2]
        global_x = np.floor(x + (np.arange(columns) + 0.5) * w / columns).astype(int)
        global_y = np.floor(y + (np.arange(rows) + 0.5) * h / rows).astype(int)
        valid_x = (global_x >= 0) & (global_x < width)
        valid_y = (global_y >= 0) & (global_y < height)
        global_x = np.clip(global_x, 0, width - 1)
        global_y = np.clip(global_y, 0, height - 1)
        in_issue = np.zeros((rows, columns), np.bool_)
        for x1, y1, x2, y2 in boxes:
            in_issue |= ((global_x[None, :] >= x1) & (global_x[None, :] < x2)
                         & (global_y[:, None] >= y1) & (global_y[:, None] < y2))
        source_pixels = source[global_y[:, None], global_x[None, :]]
        source_hues = source_hsv[global_y[:, None], global_x[None, :]]
        asset_hues = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
        restore = (in_issue & valid_x[None, :] & valid_y[:, None] & (image[:, :, 3] > 32)
                   & (source_hues[:, :, 2] >= 240) & (source_hues[:, :, 1] <= 30)
                   & (asset_hues[:, :, 2] <= 150))
        count = int(np.count_nonzero(restore))
        if count < 24:
            continue
        repaired = image.copy()
        repaired[:, :, :3][restore] = source_pixels[restore]
        target = asset_dir / f"revision_{revision_round}_{item['id']}_pale_restored.png"
        if not cv2.imwrite(str(target), repaired):
            continue
        item["src"] = f"/media/assets/{project_id}/{target.name}"
        item.setdefault("metadata", {}).update({"sourcePaleRestoredPixels": count,
                                                "sourceContentPreserved": True})
        changed.append(str(item["id"]))
    return changed
