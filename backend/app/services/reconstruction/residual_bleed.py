"""Restore source pale pixels damaged inside movable residual assets."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.reconstruction.revision_integrity import asset_path
from app.services.reconstruction.text_erasure import _clean_residual_surface, _mark


def restore_hidden_source_assets(source_path: Path, layout: dict, issues: list[dict],
                                 asset_dir: Path) -> list[str]:
    """Raise source-matching image pixels above a plate that obscures them."""
    boxes = [issue["bbox"] for issue in issues
             if issue.get("reason") == "pale_support_overwritten"
             and isinstance(issue.get("bbox"), list) and len(issue["bbox"]) == 4]
    if not boxes:
        return []
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        return []
    height, width = source.shape[:2]
    active = [item for item in layout.get("elements", []) if not any(
        (item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    containers = [item for item in active if item.get("type") in {"rectangle", "roundedRectangle", "ellipse"}
                  and (item.get("style") or {}).get("fill")]
    raised: list[str] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        area = max(1, (x2 - x1) * (y2 - y1))
        covering = [item for item in containers if _overlaps(item, box)]
        if not covering:
            continue
        top_container = max(int(item.get("zIndex") or 0) for item in covering)
        candidates = []
        for item in active:
            if item.get("type") != "image" or int(item.get("zIndex") or 0) > top_container:
                continue
            ix, iy = float(item.get("x") or 0), float(item.get("y") or 0)
            iw, ih = float(item.get("width") or 0), float(item.get("height") or 0)
            if iw <= 0 or ih <= 0 or iw * ih > area * 30 or not _overlaps(item, box):
                continue
            path = asset_path(asset_dir.parent, item.get("src"))
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path and path.is_file() else None
            if image is None or image.ndim != 3:
                continue
            left, top = max(0, round(max(ix, x1))), max(0, round(max(iy, y1)))
            right, bottom = min(width, round(min(ix + iw, x2))), min(height, round(min(iy + ih, y2)))
            if right <= left or bottom <= top:
                continue
            scaled = cv2.resize(image, (max(1, round(iw)), max(1, round(ih))))
            ax, ay = left - round(ix), top - round(iy)
            patch = scaled[ay:ay + bottom - top, ax:ax + right - left]
            if patch.shape[:2] != (bottom - top, right - left):
                continue
            close = np.max(np.abs(patch[:, :, :3].astype(np.int16)
                                  - source[top:bottom, left:right].astype(np.int16)), axis=2) <= 12
            if patch.shape[2] == 4:
                close &= patch[:, :, 3] > 32
            if np.count_nonzero(close) < max(24, round(area * 0.01)):
                continue
            candidates.append(item)
        for item in sorted(candidates, key=lambda entry: int(entry.get("zIndex") or 0)):
            top_container += 1
            item["zIndex"] = top_container
            item.setdefault("metadata", {})["sourceVisualZOrderRestored"] = True
            raised.append(str(item["id"]))
    return sorted(set(raised))


def _overlaps(item: dict, box: list[float]) -> bool:
    x, y = float(item.get("x") or 0), float(item.get("y") or 0)
    w, h = float(item.get("width") or 0), float(item.get("height") or 0)
    return x < box[2] and x + w > box[0] and y < box[3] and y + h > box[1]


def repair_residual_color_damage(source_path: Path, layout: dict, issues: list[dict],
                                 asset_dir: Path, project_id: str, revision_round: int) -> list[str]:
    """Replace only proven local asset color damage using copy-on-write files."""
    boxes = [(issue.get("bbox"), issue.get("reason")) for issue in issues
             if issue.get("reason") in {"dark_residual_over_pale_source", "pale_support_overwritten",
                                        "colored_residual_washed_out"}
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
                                      for box, _ in boxes):
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
        source_pixels = source[global_y[:, None], global_x[None, :]]
        source_hues = source_hsv[global_y[:, None], global_x[None, :]]
        asset_hues = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
        restore = np.zeros((rows, columns), np.bool_)
        colored_damage = np.zeros((rows, columns), np.bool_)
        for (x1, y1, x2, y2), reason in boxes:
            in_issue = ((global_x[None, :] >= x1) & (global_x[None, :] < x2)
                        & (global_y[:, None] >= y1) & (global_y[:, None] < y2))
            if reason == "dark_residual_over_pale_source":
                wrong_color = ((source_hues[:, :, 2] >= 240) & (source_hues[:, :, 1] <= 30)
                               & (asset_hues[:, :, 2] <= 150))
            elif reason == "colored_residual_washed_out":
                wrong_color = ((source_hues[:, :, 1] >= 60) & (source_hues[:, :, 2] >= 35)
                               & (asset_hues[:, :, 1] <= 30) & (asset_hues[:, :, 2] >= 230))
                if np.count_nonzero(in_issue & wrong_color & (image[:, :, 3] > 32)) >= 200:
                    colored_damage |= in_issue
            else:
                difference = np.max(np.abs(source_pixels.astype(np.int16) - image[:, :, :3].astype(np.int16)), axis=2)
                wrong_color = ((source_hues[:, :, 2] >= 225) & (source_hues[:, :, 1] >= 12)
                               & (source_hues[:, :, 1] <= 55) & (asset_hues[:, :, 2] >= 248)
                               & (asset_hues[:, :, 1] <= 10) & (difference >= 15))
            restore |= in_issue & wrong_color
        restore &= valid_x[None, :] & valid_y[:, None] & (image[:, :, 3] > 32)
        colored_damage &= valid_x[None, :] & valid_y[:, None] & (image[:, :, 3] > 32)
        count = int(np.count_nonzero(restore | colored_damage))
        if count < 24:
            continue
        repaired = image.copy()
        repaired[:, :, :3][restore | colored_damage] = source_pixels[restore | colored_damage]
        if np.any(colored_damage):
            # The source crop includes the old glyphs. Remove only OCR lines
            # that touch this damaged patch before staging the new asset.
            cleaned = repaired[:, :, :3].copy()
            for text in layout.get("elements", []):
                meta = text.get("metadata") or {}
                if (text.get("type") != "text" or any(meta.get(key) for key in ("suppressed", "suppressRender", "ownedBy"))):
                    continue
                raw = meta.get("rawOCRBBox")
                if not isinstance(raw, list) or len(raw) != 4:
                    tx, ty = float(text.get("x") or 0), float(text.get("y") or 0)
                    raw = [tx, ty, tx + float(text.get("width") or 0), ty + float(text.get("height") or 0)]
                text_pixels = ((global_x[None, :] >= raw[0]) & (global_x[None, :] < raw[2])
                               & (global_y[:, None] >= raw[1]) & (global_y[:, None] < raw[3]))
                if np.count_nonzero(text_pixels & colored_damage) < 12:
                    continue
                local = np.zeros((rows, columns), np.uint8)
                _mark(local, (round((raw[0] - x) * columns / w), round((raw[1] - y) * rows / h),
                              round((raw[2] - x) * columns / w), round((raw[3] - y) * rows / h)), 2)
                light_glyph = ((local > 0) & colored_damage & (source_hues[:, :, 1] <= 60)
                               & (source_hues[:, :, 2] >= 190))
                if np.count_nonzero(light_glyph) >= 10:
                    glyph_mask = cv2.dilate(np.uint8(light_glyph) * 255,
                                            np.ones((3, 3), np.uint8), iterations=1)
                    cleaned = cv2.inpaint(cleaned, glyph_mask, 3, cv2.INPAINT_TELEA)
                else:
                    cleaned = _clean_residual_surface(cleaned, repaired[:, :, 3], local, None)
            repaired[:, :, :3] = cleaned
        target = asset_dir / f"revision_{revision_round}_{item['id']}_pale_restored.png"
        if not cv2.imwrite(str(target), repaired):
            continue
        item["src"] = f"/media/assets/{project_id}/{target.name}"
        item.setdefault("metadata", {}).update({"sourcePaleRestoredPixels": count,
                                                "sourceContentPreserved": True})
        changed.append(str(item["id"]))
    return changed
