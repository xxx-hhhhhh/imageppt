from __future__ import annotations
from app.utils import image_io

import shutil
from pathlib import Path

import cv2
import numpy as np

from app.services.reconstruction.residual_objects import visual_candidate_mask


def preserve_bad_text_regions(source_path: Path, background_path: Path, preview_path: Path, layout: dict, asset_dir: Path, page: int, minimum_f1: float | None = None) -> dict[str, int]:
    """Keep ordinary OCR text editable, reporting visual mismatches for review.

    The decision is recorded per element so the user can see why that line was
    preserved as an image. A whole cleaned module is restored together when
    one of its text lines fails the check; this prevents a half-clean asset.
    """
    source = image_io.imread(str(source_path), cv2.IMREAD_COLOR)
    background = image_io.imread(str(background_path), cv2.IMREAD_COLOR)
    preview = image_io.imread(str(preview_path), cv2.IMREAD_COLOR)
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
            if _ordinary_text(item):
                from app.services.reconstruction.text_erasure import erase_editable_text_sources
                erase_editable_text_sources(background_path, layout, target_text_ids={str(item["id"])})
                background = image_io.imread(str(background_path), cv2.IMREAD_COLOR)
                metadata["visualTextMismatch"] = True
                _ensure_readable_text(item, preview[y1:y2, x1:x2])
                continue
            metadata.update({"suppressRender": True, "sourceTextPreserved": True, "ownedBy": owner, "fallbackReason": "source_asset_owns_text"})
            preserved += 1
            continue
        rendered = _rendered_text_present(source[y1:y2, x1:x2], background[y1:y2, x1:x2], preview[y1:y2, x1:x2])
        metadata["renderedTextPresent"] = rendered
        if source_retained:
            mask = np.zeros(background.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            background = cv2.inpaint(background, mask, 4, cv2.INPAINT_TELEA)
            metadata["sourceTextRecleaned"] = True
        if rendered:
            continue
        if _ordinary_text(item):
            confidence = item.get("finalConfidence", item.get("confidence"))
            if confidence is not None and float(confidence) < 0.4 and fidelity < 0.6:
                if _active_asset_overlaps((x1, y1, x2, y2), elements):
                    metadata["visualTextMismatch"] = True
                    continue
                metadata.update({"suppressed": True, "suppressRender": True, "fallbackReason": "low_confidence_ocr_mismatch"})
                if not _other_editable_text_overlaps((x1, y1, x2, y2), elements, item["id"]):
                    cutout = _source_cutout(source, (x1, y1, x2, y2), item, asset_dir, page)
                    if cutout is not None:
                        elements.append(cutout)
                        metadata["fallbackAssetId"] = cutout["id"]
                        metadata["sourceTextPreserved"] = True
                        preserved += 1
                continue
            metadata["visualTextMismatch"] = True
            _ensure_readable_text(item, preview[y1:y2, x1:x2])
            continue
        if asset_id:
            bad_assets.add(str(asset_id))
            continue
        cutout = _source_cutout(source, (x1, y1, x2, y2), item, asset_dir, page)
        if cutout is None:
            continue
        elements.append(cutout)
        metadata.update({"suppressRender": True, "sourceTextPreserved": True, "fallbackReason": "visual_text_mismatch", "fallbackAssetId": cutout["id"]})
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
    if preserved or any((item.get("metadata") or {}).get("sourceTextRecleaned") for item in elements):
        image_io.imwrite(str(background_path), background)
    return {"preservedTextRegions": preserved, "restoredModules": restored_modules}


def _source_cutout(source: np.ndarray, box: tuple[int, int, int, int], item: dict,
                   asset_dir: Path, page: int) -> dict | None:
    x1, y1, x2, y2 = box
    alpha = np.uint8(visual_candidate_mask(source)[y1:y2, x1:x2]) * 255
    if np.count_nonzero(alpha) < 12:
        return None
    alpha = cv2.dilate(alpha, np.ones((3, 3), np.uint8))
    asset_dir.mkdir(parents=True, exist_ok=True)
    cutout_id = f"fallback_{page}_{item['id']}"
    cutout_path = asset_dir / f"{cutout_id}.png"
    if not image_io.imwrite(str(cutout_path), np.dstack((source[y1:y2, x1:x2], alpha))):
        return None
    return {
        "id": cutout_id, "type": "image", "x": x1, "y": y1,
        "width": x2 - x1, "height": y2 - y1,
        "zIndex": int(item.get("zIndex") or 0) + 1,
        "src": f"/media/assets/{asset_dir.parent.name}/{cutout_path.name}",
        "style": {"opacity": 1},
        "metadata": {"reconstructionStrategy": "cutout_image", "layerRole": "cutout_image",
                     "sourceTextFallback": True, "preserveWholeAsset": True,
                     "backgroundSeparated": True, "ownedTextId": item["id"]},
    }


def _ordinary_text(item: dict) -> bool:
    metadata = item.get("metadata") or {}
    return item.get("role") not in {"logo", "decorative_text"} and not metadata.get("decorativeArtText")


def _ensure_readable_text(item: dict, preview_region: np.ndarray) -> None:
    if preview_region.size == 0:
        return
    style = item.setdefault("style", {})
    color = str(style.get("color") or "#111827").lstrip("#")
    if len(color) != 6:
        return
    try:
        red, green, blue = (int(color[index:index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return
    current_luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    backdrop = np.median(preview_region.reshape(-1, 3), axis=0)
    backdrop_luma = float(0.2126 * backdrop[2] + 0.7152 * backdrop[1] + 0.0722 * backdrop[0])
    if abs(current_luma - backdrop_luma) >= 75:
        return
    style["color"] = "#102B5C" if backdrop_luma >= 125 else "#FFFFFF"
    item.setdefault("metadata", {})["visualTextAdjusted"] = True


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


def _active_asset_overlaps(box: tuple[int, int, int, int], elements: list[dict]) -> bool:
    x1, y1, x2, y2 = box
    area = max(1, (x2 - x1) * (y2 - y1))
    for asset in elements:
        metadata = asset.get("metadata") or {}
        if asset.get("type") != "image" or any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        ax1, ay1 = float(asset.get("x") or 0), float(asset.get("y") or 0)
        ax2, ay2 = ax1 + float(asset.get("width") or 0), ay1 + float(asset.get("height") or 0)
        overlap = max(0.0, min(x2, ax2) - max(x1, ax1)) * max(0.0, min(y2, ay2) - max(y1, ay1))
        if overlap / area > 0.2:
            return True
    return False


def _other_editable_text_overlaps(box: tuple[int, int, int, int], elements: list[dict], excluded_id: str) -> bool:
    x1, y1, x2, y2 = box
    area = max(1, (x2 - x1) * (y2 - y1))
    for text in elements:
        metadata = text.get("metadata") or {}
        if text.get("id") == excluded_id or text.get("type") != "text" or any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        raw = metadata.get("rawOCRBBox")
        if not isinstance(raw, list) or len(raw) != 4:
            continue
        tx1, ty1, tx2, ty2 = map(float, raw)
        overlap = max(0.0, min(x2, tx2) - max(x1, tx1)) * max(0.0, min(y2, ty2) - max(y1, ty1))
        if overlap / area > 0.2:
            return True
    return False


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


def _rendered_text_present(source: np.ndarray, background: np.ndarray, preview: np.ndarray) -> bool:
    if background.size == 0:
        return False
    difference = cv2.absdiff(background, preview)
    changed = np.max(difference, axis=2) > 12
    if np.count_nonzero(changed) < max(3, int(changed.size * 0.003)):
        return False
    source_ink = np.max(cv2.absdiff(source, background), axis=2) > 12
    if np.count_nonzero(source_ink) < 3:
        return True
    source_y, source_x = np.where(source_ink)
    render_y, render_x = np.where(changed)
    return (abs(float(np.median(source_x)) - float(np.median(render_x))) <= max(4, changed.shape[1] * 0.18)
            and abs(float(np.median(source_y)) - float(np.median(render_y))) <= max(3, changed.shape[0] * 0.25))
