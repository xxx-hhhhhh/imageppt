from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np


def preserve_bad_text_regions(source_path: Path, background_path: Path, preview_path: Path, layout: dict, asset_dir: Path, page: int, minimum_f1: float | None = None) -> dict[str, int]:
    """Restore source pixels when an editable text line visibly loses fidelity.

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
        threshold = minimum_f1 if minimum_f1 is not None else (0.62 if item.get("role") in {"main_title", "section_title", "subtitle"} else 0.55)
        if fidelity >= threshold:
            continue
        asset_id = metadata.get("textCleanedFromAsset")
        if asset_id:
            bad_assets.add(str(asset_id))
            continue
        background[y1:y2, x1:x2] = source[y1:y2, x1:x2]
        metadata.update({"suppressRender": True, "sourceTextPreserved": True, "fallbackReason": "visual_text_mismatch"})
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
