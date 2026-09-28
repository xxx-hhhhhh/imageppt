from __future__ import annotations

from pathlib import Path
from typing import Callable
import shutil

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
        project_root = background_path.parent.parent
        if copy_asset_prefix or project_root not in path.parents:
            local_assets = project_root / "assets"
            local_assets.mkdir(parents=True, exist_ok=True)
            local_path = local_assets / f"{copy_asset_prefix or 'text_clean'}_{asset['id']}.png"
            shutil.copy2(path, local_path)
            path = local_path
            asset["src"] = f"/media/assets/{project_root.name}/{local_path.name}"
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
        if original.shape[2] == 4 and meta.get("layerRole") == "residual":
            original[:, :, 3][mask > 0] = 0
        elif original.shape[2] == 4:
            rgb = _clean(original[:, :, :3], mask, complex_cleaner)
            original[:, :, :3] = rgb
        else:
            original = _clean(original, mask, complex_cleaner)
        cv2.imwrite(str(path), original)
        meta["textCleaned"] = True
        meta["editableTextIds"] = sorted(set((meta.get("editableTextIds") or []) + owned_text))
        cleaned_assets += 1
    return {"backgroundTextErased": len(texts), "assetTextErased": cleaned_assets}


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
