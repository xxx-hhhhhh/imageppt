from __future__ import annotations

from pathlib import Path
from typing import Callable
from uuid import uuid4

import cv2
import numpy as np

from app.services.pptx.renderer import _path_from_src


ComplexTextCleaner = Callable[[np.ndarray, np.ndarray], np.ndarray]


def erase_editable_text_sources(
    background_path: Path,
    layout: dict,
    *,
    complex_cleaner: ComplexTextCleaner | None = None,
    target_text_ids: set[str] | None = None,
    copy_asset_prefix: str | None = None,
    clean_background: bool = True,
    project_root: Path | None = None,
    protect_background_elements: bool = False,
    force_asset_reclean_ids: set[str] | None = None,
) -> dict[str, int]:
    """Remove source glyphs from every layer beneath an editable OCR line.

    A complex_cleaner may later provide AI inpainting for textured assets. The
    local OpenCV cleaner remains the deterministic default and fallback.
    """
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(background_path)
    height, width = background.shape[:2]
    texts = []
    for item in layout.get("elements", []):
        meta = item.get("metadata") or {}
        raw = meta.get("rawOCRBBox")
        if item.get("type") != "text" or not str(item.get("text") or "").strip():
            continue
        if target_text_ids is not None and str(item.get("id")) not in target_text_ids:
            continue
        if any(meta.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        meta.pop("ghostingDetected", None)
        if not (isinstance(raw, list) and len(raw) == 4):
            x, y = float(item.get("x") or 0), float(item.get("y") or 0)
            raw = [x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)]
        if isinstance(raw, list) and len(raw) == 4:
            x1, y1, x2, y2 = [int(round(float(value))) for value in raw]
            box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
            if box[2] > box[0] and box[3] > box[1]:
                texts.append((item, box))
                meta["textOwner"] = item["id"]
    if not texts:
        return {"backgroundTextErased": 0, "assetTextErased": 0}

    background_mask = np.zeros((height, width), np.uint8)
    for _, box in texts:
        _mark(background_mask, box, 2)
    if protect_background_elements:
        for element in layout.get("elements", []):
            meta = element.get("metadata") or {}
            if element.get("type") not in {"image", "rectangle", "roundedRectangle", "ellipse", "line", "arrow"} or any(meta.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
                continue
            x1, y1 = int(float(element.get("x") or 0)), int(float(element.get("y") or 0))
            x2, y2 = x1 + int(float(element.get("width") or 0)), y1 + int(float(element.get("height") or 0))
            background_mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = 0
    if clean_background:
        background = _clean(background, background_mask, complex_cleaner)
        cv2.imwrite(str(background_path), background)

    cleaned_assets = 0
    for asset in layout.get("elements", []):
        meta = asset.setdefault("metadata", {})
        if asset.get("type") != "image" or any(meta.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        raw_src = str(asset.get("src") or "")
        path = Path(raw_src) if Path(raw_src).is_absolute() else _path_from_src(raw_src)
        if not path or not path.is_file():
            continue
        original = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if original is None or original.ndim != 3:
            continue
        ax, ay = float(asset.get("x") or 0), float(asset.get("y") or 0)
        aw, ah = float(asset.get("width") or 0), float(asset.get("height") or 0)
        if aw <= 0 or ah <= 0:
            continue
        mask = np.zeros(original.shape[:2], np.uint8)
        owned_text = []
        for item, (x1, y1, x2, y2) in texts:
            left, top = max(ax, x1), max(ay, y1)
            right, bottom = min(ax + aw, x2), min(ay + ah, y2)
            if right <= left or bottom <= top:
                continue
            area = (x2 - x1) * (y2 - y1)
            if (right - left) * (bottom - top) / max(1, area) < 0.25:
                continue
            sx, sy = original.shape[1] / aw, original.shape[0] / ah
            _mark(mask, (round((left - ax) * sx), round((top - ay) * sy), round((right - ax) * sx), round((bottom - ay) * sy)), 2)
            owned_text.append(item["id"])
        if not owned_text:
            continue
        if meta.get("textCleaned") and set(owned_text).issubset(set(meta.get("editableTextIds") or [])) and not set(owned_text).intersection(force_asset_reclean_ids or set()):
            continue
        if original.shape[2] == 4 and meta.get("layerRole") == "residual":
            # A residual asset may be a pale card or ribbon carrying editable
            # text. Clearing its whole textbox alpha punches a white rectangle
            # through that support. Reconstruct the local surface instead.
            alpha = original[:, :, 3]
            source_rgb = original[:, :, :3].copy()
            cleaned_rgb = _clean_residual_surface(source_rgb, alpha, mask, complex_cleaner)
            original[:, :, :3] = cleaned_rgb
            # Close glyph-sized holes inside the existing support without
            # expanding its outline into neighboring white space. Only pixels
            # whose source color differs from the repaired surface are glyphs.
            enclosed_support = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            contrast = np.max(np.abs(source_rgb.astype(np.int16) - cleaned_rgb.astype(np.int16)), axis=2)
            # A broad pale plate often has transparent glyph-shaped holes.
            # Fill cavities enclosed by that plate, while leaving exterior
            # transparency untouched even when the OCR box crosses its edge.
            support = np.pad(np.uint8(alpha > 32), (1, 1))
            exterior = support.copy()
            cv2.floodFill(exterior, None, (0, 0), 2)
            holes = exterior[1:-1, 1:-1] == 0
            refill = (mask > 0) & (holes | ((enclosed_support > 32) & (contrast >= 18)))
            alpha[refill] = np.maximum(alpha[refill], enclosed_support[refill])
            alpha[refill & holes] = 255
        elif original.shape[2] == 4:
            rgb = _clean(original[:, :, :3], mask, complex_cleaner)
            original[:, :, :3] = rgb
        else:
            original = _clean(original, mask, complex_cleaner)
        owner_root = project_root or background_path.parent.parent
        output_path = path
        if copy_asset_prefix or owner_root not in path.parents:
            local_assets = owner_root / "assets"
            local_assets.mkdir(parents=True, exist_ok=True)
            output_path = local_assets / f"{copy_asset_prefix or 'text_clean'}_{asset['id']}_{uuid4().hex[:8]}.png"
        if not cv2.imwrite(str(output_path), original):
            raise OSError(f"Could not write cleaned image asset: {output_path}")
        if output_path != path:
            asset["src"] = f"/media/assets/{owner_root.name}/{output_path.name}"
        meta["textCleaned"] = True
        meta["editableTextIds"] = sorted(set((meta.get("editableTextIds") or []) + owned_text))
        cleaned_assets += 1
    return {"backgroundTextErased": len(texts), "assetTextErased": cleaned_assets}


def _clean_residual_surface(rgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray,
                            complex_cleaner: ComplexTextCleaner | None) -> np.ndarray:
    """Prefer a flat local plate color when nearby artwork would bleed inward."""
    ring = cv2.dilate(mask, np.ones((17, 17), np.uint8))
    samples = rgb[(ring > 0) & (mask == 0) & (alpha > 32)]
    if len(samples) >= 30:
        median = np.median(samples, axis=0)
        flat_fraction = float(np.mean(np.max(np.abs(samples.astype(np.float32) - median), axis=1) <= 12))
        if flat_fraction >= 0.65:
            cleaned = rgb.copy()
            cleaned[mask > 0] = np.uint8(np.round(median))
            return cleaned
    return _clean(rgb, mask, complex_cleaner)


def count_text_ghosting(source_path: Path, background_path: Path, layout: dict) -> int:
    """Count editable OCR lines whose original glyph edges remain underneath."""
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if source is None or background is None:
        return 0
    height, width = source.shape[:2]
    count = 0
    assets = [item for item in layout.get("elements", []) if item.get("type") == "image" and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    for item in layout.get("elements", []):
        meta = item.get("metadata") or {}
        raw = meta.get("rawOCRBBox")
        if item.get("type") != "text" or not str(item.get("text") or "").strip():
            continue
        if any(meta.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        meta.pop("ghostingDetected", None)
        if not (isinstance(raw, list) and len(raw) == 4):
            x, y = float(item.get("x") or 0), float(item.get("y") or 0)
            raw = [x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)]
        x1, y1, x2, y2 = [int(round(float(value))) for value in raw]
        box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        original = source[box[1]:box[3], box[0]:box[2]]
        if _source_edges_retained(original, background[box[1]:box[3], box[0]:box[2]]):
            count += 1
            meta["ghostingDetected"] = True
            continue
        for asset in assets:
            ax, ay = float(asset.get("x") or 0), float(asset.get("y") or 0)
            aw, ah = float(asset.get("width") or 0), float(asset.get("height") or 0)
            if aw <= 0 or ah <= 0 or ax > box[0] or ay > box[1] or ax + aw < box[2] or ay + ah < box[3]:
                continue
            path = _path_from_src(asset.get("src"))
            if path is None or not path.is_file():
                continue
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim != 3:
                continue
            sx, sy = image.shape[1] / aw, image.shape[0] / ah
            bx1, by1 = int(round((box[0] - ax) * sx)), int(round((box[1] - ay) * sy))
            bx2, by2 = int(round((box[2] - ax) * sx)), int(round((box[3] - ay) * sy))
            patch = image[by1:by2, bx1:bx2]
            if patch.size == 0:
                continue
            if patch.shape[2] == 4 and np.mean(patch[:, :, 3]) < 128:
                continue
            patch = cv2.resize(patch[:, :, :3], (original.shape[1], original.shape[0]))
            if _source_edges_retained(original, patch):
                count += 1
                meta["ghostingDetected"] = True
                break
    return count


def _source_edges_retained(source: np.ndarray, candidate: np.ndarray) -> bool:
    edges = cv2.Canny(source, 60, 160) > 0
    if np.count_nonzero(edges) < 12:
        return False
    close_pixels = np.max(cv2.absdiff(source, candidate), axis=2) <= 8
    return float(np.count_nonzero(edges & close_pixels)) / np.count_nonzero(edges) >= 0.65


def _mark(mask: np.ndarray, box: tuple[int, int, int, int], padding: int) -> None:
    x1, y1, x2, y2 = box
    h, w = mask.shape
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255


def _clean(image: np.ndarray, mask: np.ndarray, complex_cleaner: ComplexTextCleaner | None) -> np.ndarray:
    if not np.any(mask):
        return image
    if complex_cleaner is not None:
        try:
            result = complex_cleaner(image, mask)
            if isinstance(result, np.ndarray) and result.shape == image.shape:
                return result
        except Exception:
            pass
    return cv2.inpaint(image, mask, 4, cv2.INPAINT_TELEA)
