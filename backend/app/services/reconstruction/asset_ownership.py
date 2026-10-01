"""Keep visual assets complete after final text ownership has been resolved."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.services.pptx.renderer import _path_from_src


def is_badge_owned_text(item: dict, layout: dict) -> bool:
    """An OCR fragment inside a complete badge is visual content, not missing text."""
    if item.get("type") != "text" or not _hidden(item):
        return False
    item_id = str(item.get("id"))
    owner_id = str((item.get("metadata") or {}).get("ownedBy") or "")
    for asset in layout.get("elements", []):
        if asset.get("type") != "image":
            continue
        metadata = asset.get("metadata") or {}
        whole_badge = metadata.get("wholeBadgeAsset") or metadata.get("componentType") == "wholeBadgeImage"
        if str(asset.get("id")) == owner_id and whole_badge:
            return True
        image_text_ids = [*metadata.get("editableTextIds", []), *metadata.get("restoredImageOwnedTextIds", [])]
        if (whole_badge and item_id in image_text_ids
                and _overlap(_text_box(item), _image_box(asset))):
            return True
    return False


def restore_image_owned_text(source_path: Path, layout: dict, asset_dir: Path,
                             project_id: str, *, prefix: str = "owned_text",
                             target_boxes: list[list[float]] | None = None) -> list[str]:
    """Restore source pixels when every textbox cleaned from an image is hidden.

    The image is staged under a new name before its scene reference changes.
    Its original alpha is retained so a circular badge stays circular.
    """
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        return []
    height, width = source.shape[:2]
    by_id = {str(item.get("id")): item for item in layout.get("elements", [])}
    restored: list[str] = []
    for item in layout.get("elements", []):
        metadata = item.get("metadata") or {}
        if item.get("type") != "image" or not metadata.get("textCleaned") or any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        text_ids = [str(value) for value in metadata.get("editableTextIds") or []]
        if not text_ids or any(text_id not in by_id or not _hidden(by_id[text_id]) for text_id in text_ids):
            continue
        x, y = round(float(item.get("x") or 0)), round(float(item.get("y") or 0))
        w, h = round(float(item.get("width") or 0)), round(float(item.get("height") or 0))
        box = (x, y, x + w, y + h)
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            continue
        if target_boxes is not None and not any(_overlap(box, candidate) for candidate in target_boxes):
            continue
        if any(other.get("type") == "text" and not _hidden(other) and _overlap(box, _text_box(other))
               for other in layout.get("elements", [])):
            continue
        prior_path = _path_from_src(item.get("src"))
        if prior_path is None or not prior_path.is_file():
            continue
        prior = cv2.imread(str(prior_path), cv2.IMREAD_UNCHANGED)
        if prior is None or prior.ndim != 3 or prior.shape[2] not in (3, 4):
            continue
        rgb = cv2.resize(source[y:y + h, x:x + w], (prior.shape[1], prior.shape[0]), interpolation=cv2.INTER_LINEAR)
        candidate = np.dstack((rgb, prior[:, :, 3])) if prior.shape[2] == 4 else rgb
        asset_dir.mkdir(parents=True, exist_ok=True)
        path = asset_dir / f"{prefix}_{item['id']}_{uuid4().hex[:8]}.png"
        if not cv2.imwrite(str(path), candidate):
            continue
        item["src"] = f"/media/assets/{project_id}/{path.name}"
        metadata.update({"textCleaned": False, "editableTextIds": [], "sourceContentPreserved": True,
                         "restoredImageOwnedTextIds": text_ids})
        restored.append(str(item["id"]))
    return restored


def _hidden(item: dict) -> bool:
    metadata = item.get("metadata") or {}
    return any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy"))


def _text_box(item: dict) -> tuple[float, float, float, float]:
    raw = (item.get("metadata") or {}).get("rawOCRBBox")
    if isinstance(raw, list) and len(raw) == 4:
        return tuple(float(value) for value in raw)
    x, y = float(item.get("x") or 0), float(item.get("y") or 0)
    return x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)


def _image_box(item: dict) -> tuple[float, float, float, float]:
    x, y = float(item.get("x") or 0), float(item.get("y") or 0)
    return x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)


def _overlap(first: tuple[float, float, float, float], second: tuple[float, float, float, float] | list[float]) -> bool:
    return min(first[2], second[2]) > max(first[0], second[0]) and min(first[3], second[3]) > max(first[1], second[1])
