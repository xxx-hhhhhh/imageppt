from __future__ import annotations

from pathlib import Path
from typing import Callable
from uuid import uuid4

import cv2
import numpy as np

from app.services.pptx.renderer import _path_from_src
from app.services.reconstruction.owner_gate import tight_text_mask


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
        if item.get("confidence") is not None and float(item["confidence"]) < .5:
            continue  # Uncertain OCR never authorizes deleting source pixels.
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
    for text, box in texts:
        region = np.zeros((height, width), np.uint8)
        _mark(region, box, 0)
        background_mask |= tight_text_mask(background, region, (text.get("style") or {}).get("color"))
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
        text_regions: list[tuple[tuple[int, int, int, int], str | None]] = []
        for item, (x1, y1, x2, y2) in texts:
            left, top = max(ax, x1), max(ay, y1)
            right, bottom = min(ax + aw, x2), min(ay + ah, y2)
            if right <= left or bottom <= top:
                continue
            area = (x2 - x1) * (y2 - y1)
            confirmed_assets = (item.get("metadata") or {}).get("ghostingAssetIds") or []
            if (right - left) * (bottom - top) / max(1, area) < 0.25 and asset.get("id") not in confirmed_assets:
                continue
            sx, sy = original.shape[1] / aw, original.shape[0] / ah
            local_box = (round((left - ax) * sx), round((top - ay) * sy),
                         round((right - ax) * sx), round((bottom - ay) * sy))
            _mark(mask, local_box, 2)
            text_regions.append((local_box, (item.get("style") or {}).get("color")))
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
            support = np.pad(np.uint8(alpha > 32), (1, 1))
            exterior = support.copy()
            cv2.floodFill(exterior, None, (0, 0), 2)
            holes = exterior[1:-1, 1:-1] == 0
            cleaned_rgb = source_rgb.copy()
            # A single residual crop may contain several colored surfaces.
            # Repair each OCR line against its own local surface; a union mask
            # would choose one palette color and wash out the other modules.
            for local_box, glyph_color in text_regions:
                line_mask = np.zeros(mask.shape, np.uint8)
                _mark(line_mask, local_box, 2)
                local_alpha = alpha.copy()
                eligible = _light_glyph_on_colored_surface(source_rgb, line_mask, glyph_color)
                if eligible:
                    local_alpha[(line_mask > 0) & holes] = 255
                candidate_rgb = _clean_residual_surface(cleaned_rgb, local_alpha, line_mask, complex_cleaner,
                                                      glyph_color=glyph_color,
                                                      bounded_support=meta.get("reconstructionStrategySource") == "round_contour")
                authorized_ink = tight_text_mask(source_rgb, line_mask, glyph_color)
                cleaned_rgb = np.where(authorized_ink[:, :, None] > 0, candidate_rgb, cleaned_rgb)
            original[:, :, :3] = cleaned_rgb
            # Close glyph-sized holes inside the existing support without
            # expanding its outline into neighboring white space. Only pixels
            # whose source color differs from the repaired surface are glyphs.
            enclosed_support = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            contrast = np.max(np.abs(source_rgb.astype(np.int16) - cleaned_rgb.astype(np.int16)), axis=2)
            # A broad pale plate often has transparent glyph-shaped holes.
            # Fill cavities enclosed by that plate, while leaving exterior
            # transparency untouched even when the OCR box crosses its edge.
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
                            complex_cleaner: ComplexTextCleaner | None,
                            *, glyph_color: str | None = None, bounded_support: bool = False) -> np.ndarray:
    """Prefer a flat local plate color when nearby artwork would bleed inward."""
    local = (mask > 0) & (alpha > 32)
    color = str(glyph_color or "").lstrip("#")
    if len(color) == 6 and np.count_nonzero(local) >= 80:
        try:
            glyph_bgr = np.frombuffer(bytes.fromhex(color)[::-1], dtype=np.uint8).astype(np.int16)
        except ValueError:
            glyph_bgr = None
        if glyph_bgr is not None:
            source = rgb.astype(np.int16)
            color_distance = np.max(np.abs(source - glyph_bgr), axis=2)
            candidate = np.uint8(local & (color_distance <= 38)) * 255
            count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
            glyphs = np.zeros(candidate.shape, np.uint8)
            _, _, box_width, box_height = cv2.boundingRect(mask)
            max_component = max(20, min(round(box_height * box_height * 2),
                                        round(box_width * box_height * (0.45 if bounded_support else 0.12))))
            for index in range(1, count):
                if 2 <= int(stats[index, cv2.CC_STAT_AREA]) <= max_component:
                    glyphs[labels == index] = 255
            background_pixels = source[local & (glyphs == 0)]
            if (np.count_nonzero(glyphs) >= 8 and np.count_nonzero(glyphs) <= np.count_nonzero(local) * (0.48 if bounded_support else 0.38)
                    and len(background_pixels) >= 30
                    and np.max(np.abs(np.median(background_pixels, axis=0) - glyph_bgr)) >= 45):
                glyphs = cv2.dilate(glyphs, np.ones((3, 3), np.uint8), iterations=1)
                return cv2.inpaint(rgb, glyphs, 3, cv2.INPAINT_TELEA)
    if np.count_nonzero(local) >= 80:
        pixels = rgb[local].astype(np.int16)
        median = np.median(pixels, axis=0)
        stable_fraction = float(np.mean(np.max(np.abs(pixels - median), axis=1) <= 18))
        x, y, w, h = cv2.boundingRect(mask)
        boundary = np.zeros(mask.shape, np.bool_)
        boundary[y:y + min(2, h), x:x + w] = True
        boundary[max(y, y + h - 2):y + h, x:x + w] = True
        boundary[y:y + h, x:x + min(2, w)] = True
        boundary[y:y + h, max(x, x + w - 2):x + w] = True
        edge_pixels = rgb[boundary & (alpha > 32)].astype(np.int16)
        edge_agreement = (float(np.mean(np.max(np.abs(edge_pixels - median), axis=1) <= 18))
                          if len(edge_pixels) >= 20 else 0.0)
        if stable_fraction >= 0.58 and edge_agreement >= 0.58:
            cleaned = rgb.copy()
            cleaned[mask > 0] = np.uint8(np.round(median))
            return cleaned
    # OCR boxes often include the empty space around several short labels.
    # Inpainting that entire box can pull a neighboring dark photograph into
    # an intact pale plate. Preserve locally flat, bright source pixels; thin
    # light glyphs fail the neighborhood agreement test and remain erasable.
    if np.any(mask):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        pale = (hsv[:, :, 2] >= 230) & (hsv[:, :, 1] <= 35) & (alpha > 32)
        similar = np.zeros(mask.shape, np.uint8)
        source_pixels = rgb.astype(np.int16)
        padded = cv2.copyMakeBorder(source_pixels, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        for dy in range(3):
            for dx in range(3):
                neighbor = padded[dy:dy + rgb.shape[0], dx:dx + rgb.shape[1]]
                similar += (np.max(np.abs(neighbor - source_pixels), axis=2) <= 12).astype(np.uint8)
        mask = mask.copy()
        mask[pale & (similar >= 7)] = 0
        if not np.any(mask):
            return rgb.copy()
    # When OCR covers almost the entire crop, the outer inpainting ring may
    # contain mostly neighboring glyphs. A stable pale surface within the
    # source crop is stronger evidence for the original card color.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    pale_surface = (hsv[:, :, 2] >= 220) & (hsv[:, :, 1] <= 40) & (alpha > 32)
    pale_samples = rgb[pale_surface]
    if len(pale_samples) >= 30 and len(pale_samples) / max(1, np.count_nonzero(alpha > 32)) >= 0.35:
        median = np.median(pale_samples, axis=0)
        flat_fraction = float(np.mean(np.max(np.abs(pale_samples.astype(np.float32) - median), axis=1) <= 18))
        if flat_fraction >= 0.70:
            cleaned = rgb.copy()
            cleaned[mask > 0] = np.uint8(np.round(median))
            return cleaned
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


def _light_glyph_on_colored_surface(rgb: np.ndarray, mask: np.ndarray,
                                    glyph_color: str | None) -> bool:
    color = str(glyph_color or "").lstrip("#")
    if len(color) != 6 or np.count_nonzero(mask) < 80:
        return False
    try:
        glyph_bgr = np.frombuffer(bytes.fromhex(color)[::-1], dtype=np.uint8).reshape(1, 1, 3)
    except ValueError:
        return False
    glyph_hsv = cv2.cvtColor(glyph_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    if int(glyph_hsv[1]) > 50 or int(glyph_hsv[2]) < 200:
        return False
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    local = mask > 0
    vivid = (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 45)
    return float(np.mean(vivid[local])) >= 0.35


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
        meta.pop("ghostingAssetIds", None)
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
        matched_assets = []
        for asset in assets:
            ax, ay = float(asset.get("x") or 0), float(asset.get("y") or 0)
            aw, ah = float(asset.get("width") or 0), float(asset.get("height") or 0)
            if aw <= 0 or ah <= 0:
                continue
            left, top = max(box[0], round(ax)), max(box[1], round(ay))
            right, bottom = min(box[2], round(ax + aw)), min(box[3], round(ay + ah))
            if right <= left or bottom <= top:
                continue
            path = _path_from_src(asset.get("src"))
            if path is None or not path.is_file():
                continue
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim != 3:
                continue
            sx, sy = image.shape[1] / aw, image.shape[0] / ah
            bx1, by1 = max(0, round((left - ax) * sx)), max(0, round((top - ay) * sy))
            bx2, by2 = min(image.shape[1], round((right - ax) * sx)), min(image.shape[0], round((bottom - ay) * sy))
            patch = image[by1:by2, bx1:bx2]
            if patch.size == 0:
                continue
            local_source = source[top:bottom, left:right]
            rgb = cv2.resize(patch[:, :, :3], (right - left, bottom - top))
            if patch.shape[2] == 4:
                alpha = cv2.resize(patch[:, :, 3], (right - left, bottom - top)).astype(np.float32)[:, :, None] / 255
                underlay = background[top:bottom, left:right]
                rgb = np.uint8(np.round(rgb * alpha + underlay * (1 - alpha)))
            if _source_edges_retained(local_source, rgb):
                matched_assets.append(asset.get("id"))
        if matched_assets:
            count += 1
            meta["ghostingDetected"] = True
            meta["ghostingAssetIds"] = matched_assets
    return count


def _source_edges_retained(source: np.ndarray, candidate: np.ndarray) -> bool:
    edges = cv2.Canny(source, 60, 160) > 0
    if np.count_nonzero(edges) < 12:
        return False
    close_pixels = np.max(cv2.absdiff(source, candidate), axis=2) <= 8
    candidate_edges = cv2.dilate(cv2.Canny(candidate, 60, 160), np.ones((3, 3), np.uint8)) > 0
    return float(np.count_nonzero(edges & close_pixels & candidate_edges)) / np.count_nonzero(edges) >= 0.65


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
    # The external model cannot alter anything outside the authorized ink mask.
    mask = tight_text_mask(image, mask)
    if not np.any(mask):
        return image.copy()
    if complex_cleaner is not None:
        try:
            result = complex_cleaner(image, mask)
            if isinstance(result, np.ndarray) and result.shape == image.shape:
                return np.where(mask[:, :, None] > 0, result, image)
        except Exception:
            pass
    result = cv2.inpaint(image, mask, 4, cv2.INPAINT_TELEA)
    return np.where(mask[:, :, None] > 0, result, image)
