from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np


def preserve_bad_text_regions(source_path: Path, background_path: Path, preview_path: Path, layout: dict, asset_dir: Path, page: int, minimum_f1: float | None = None) -> dict[str, int]:
    """Preserve a mismatched source line as an independently movable image.

    The decision is recorded per element so the user can see why that line was
    preserved as an image. A whole cleaned module is restored together when
    one of its text lines fails the check; this prevents a half-clean asset.
    """
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    preview = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
    if source is None or background is None or preview is None:
        return {"preservedTextRegions": 0, "restoredModules": 0}
    elements = layout.get("elements", [])
    by_id = {item["id"]: item for item in elements}
    bad_assets: set[str] = set()
    preserved = 0
    for item in elements:
        metadata = item.get("metadata") or {}
        if item.get("type") != "text" or any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        raw = metadata.get("rawOCRBBox")
        if not isinstance(raw, list) or len(raw) != 4:
            continue
        x1, y1, x2, y2 = _clip_box(raw, source.shape)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        fidelity = _edge_f1(source[y1:y2, x1:x2], preview[y1:y2, x1:x2])
        metadata["visualTextF1"] = round(fidelity, 4)
        asset_id = metadata.get("textCleanedFromAsset")
        source_retained = not asset_id and float(np.mean(cv2.absdiff(source[y1:y2, x1:x2], background[y1:y2, x1:x2]))) < 3.0
        owner = _source_asset_owner((x1, y1, x2, y2), elements) if not asset_id else None
        if owner:
            if source_retained:
                mask = np.zeros(background.shape[:2], dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                background = cv2.inpaint(background, mask, 4, cv2.INPAINT_TELEA)
            metadata.update({"suppressRender": True, "sourceTextPreserved": True, "ownedBy": owner, "fallbackReason": "source_asset_owns_text"})
            preserved += 1
            continue
        threshold = minimum_f1 if minimum_f1 is not None else (0.62 if item.get("role") in {"main_title", "section_title", "subtitle"} else 0.55)
        if fidelity >= threshold and not source_retained:
            continue
        if asset_id:
            bad_assets.add(str(asset_id))
            continue
        if source_retained:
            mask = np.zeros(background.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            background = cv2.inpaint(background, mask, 4, cv2.INPAINT_TELEA)
        asset_dir.mkdir(parents=True, exist_ok=True)
        cutout_id = f"fallback_{page}_{item['id']}"
        cutout_path = asset_dir / f"{cutout_id}.png"
        cv2.imwrite(str(cutout_path), source[y1:y2, x1:x2])
        elements.append({
            "id": cutout_id, "type": "image", "x": x1, "y": y1,
            "width": x2 - x1, "height": y2 - y1,
            "zIndex": int(item.get("zIndex") or 0) + 1,
            "src": f"/media/assets/{asset_dir.parent.name}/{cutout_path.name}",
            "style": {"opacity": 1},
            "metadata": {"reconstructionStrategy": "cutout_image", "layerRole": "cutout_image", "sourceTextFallback": True, "preserveWholeAsset": True, "backgroundSeparated": True, "ownedTextId": item["id"]},
        })
        metadata.update({"suppressRender": True, "sourceTextPreserved": True, "fallbackReason": "visual_text_mismatch", "fallbackAssetId": cutout_id})
        preserved += 1
    restored_modules = 0
    for asset_id in bad_assets:
        asset = by_id.get(asset_id)
        if not asset:
            continue
        raw_asset = asset_dir.parent / "module_assets" / f"page_{page}" / f"{asset_id}.png"
        clean_asset = asset_dir / f"{asset_id}.png"
        if not raw_asset.is_file():
            continue
        shutil.copy2(raw_asset, clean_asset)
        asset.setdefault("metadata", {}).update({"textCleaned": False, "fallbackReason": "visual_text_mismatch"})
        restored_modules += 1
        for item in elements:
            metadata = item.get("metadata") or {}
            if metadata.get("textCleanedFromAsset") == asset_id:
                metadata.update({"suppressRender": True, "sourceTextPreserved": True, "fallbackReason": "visual_text_mismatch"})
                preserved += 1
    if preserved:
        cv2.imwrite(str(background_path), background)
    return {"preservedTextRegions": preserved, "restoredModules": restored_modules}


def _clip_box(box: list[float], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x1, y1, x2, y2 = (round(float(value)) for value in box)
    return max(0, min(width, x1)), max(0, min(height, y1)), max(0, min(width, x2)), max(0, min(height, y2))


def _source_asset_owner(box: tuple[int, int, int, int], elements: list[dict]) -> str | None:
    x1, y1, x2, y2 = box
    area = max(1, (x2 - x1) * (y2 - y1))
    for asset in elements:
        metadata = asset.get("metadata") or {}
        if asset.get("type") != "image" or not metadata.get("preserveWholeAsset") or metadata.get("textCleaned"):
            continue
        if any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        ax1, ay1 = float(asset.get("x") or 0), float(asset.get("y") or 0)
        ax2, ay2 = ax1 + float(asset.get("width") or 0), ay1 + float(asset.get("height") or 0)
        overlap = max(0.0, min(x2, ax2) - max(x1, ax1)) * max(0.0, min(y2, ay2) - max(y1, ay1))
        if overlap / area >= 0.85:
            return str(asset["id"])
    return None


def _edge_f1(source: np.ndarray, preview: np.ndarray) -> float:
    source_edges = cv2.Canny(source, 50, 150) > 0
    preview_edges = cv2.Canny(preview, 50, 150) > 0
    if not np.any(source_edges) and not np.any(preview_edges):
        return 1.0
    kernel = np.ones((3, 3), np.uint8)
    source_near = cv2.dilate(source_edges.astype(np.uint8), kernel) > 0
    preview_near = cv2.dilate(preview_edges.astype(np.uint8), kernel) > 0
    precision = float(np.count_nonzero(preview_edges & source_near)) / max(1, np.count_nonzero(preview_edges))
    recall = float(np.count_nonzero(source_edges & preview_near)) / max(1, np.count_nonzero(source_edges))
    return 2 * precision * recall / max(1e-9, precision + recall)
