"""Transfer reported background artwork into staged movable cutouts."""

from pathlib import Path

import cv2
from app.utils import image_io
import numpy as np

from app.services.reconstruction.objectization_audit import _box, _connected_glyph_blends, _visual_mask
from app.services.reconstruction.residual_objects import extract_residual_objects
from app.services.reconstruction.white_objectization import _occupy_text_glyphs


def objectize_background_regions(source_path: Path, background_path: Path, layout: dict,
                                 issues: list[dict], asset_dir: Path, project_id: str,
                                 page: int, revision_round: int) -> list[dict]:
    """Clear only pixels already saved in valid new assets; freeze old owners.

    Called on the revision's staged background. Failure leaves that background
    and the scene untouched, so the revision integrity gate can retain the
    previous result. Crops come from the current background, not the original
    page, which would reintroduce already removed text or objects.
    """
    if not issues:
        return []
    background = image_io.imread(str(background_path), cv2.IMREAD_COLOR)
    if background is None:
        return []
    height, width = background.shape[:2]
    occupied = np.full((height, width), 255, np.uint8)
    for issue in issues:
        box = issue.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [round(float(v)) for v in box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            occupied[y1:y2, x1:x2] = 0
    targeted = occupied == 0
    duplicates = np.zeros((height, width), np.bool_)
    glyphs = np.zeros((height, width), np.uint8)
    protected = np.zeros((height, width), np.bool_)
    text_owners = []
    reused = []
    for item in layout.get("elements", []):
        meta = item.get("metadata") or {}
        if item.get("type") in {"background", "group"} or meta.get("pageSurface") or any(meta.get(k) for k in ("suppressed", "suppressRender", "ownedBy")):
            continue
        box = _box(item, width, height)
        if box is None:
            continue
        if item.get("type") == "text":
            raw = meta.get("rawOCRBBox")
            if isinstance(raw, list) and len(raw) == 4:
                box = (max(0, round(raw[0])), max(0, round(raw[1])), min(width, round(raw[2])), min(height, round(raw[3])))
            _occupy_text_glyphs(background, occupied, box)
            x1, y1, x2, y2 = box
            color = str((item.get("style") or {}).get("color") or "").lstrip("#")
            if x2 > x1 and y2 > y1 and len(color) == 6:
                try:
                    ink = np.frombuffer(bytes.fromhex(color)[::-1], dtype=np.uint8)
                except ValueError:
                    continue
                local = _connected_glyph_blends(background[y1:y2, x1:x2], ink)
                local &= targeted[y1:y2, x1:x2]
                if np.any(local):
                    glyphs[y1:y2, x1:x2][local] = 255
                    text_owners.append(item)
        else:
            x1, y1, x2, y2 = box
            mask = _visual_mask(item, box, source_path)
            occupied[y1:y2, x1:x2][mask > 0] = 255
            protected[y1:y2, x1:x2] |= mask > 0
            if item.get("type") == "image" and float((item.get("style") or {}).get("opacity", 1)) == 1:
                from app.services.reconstruction.revision_integrity import asset_path
                path = asset_path(source_path.parent, item.get("src"))
                image = image_io.imread(str(path), cv2.IMREAD_UNCHANGED) if path else None
                if image is not None and image.ndim == 3:
                    image = cv2.resize(image, (x2 - x1, y2 - y1))
                    opaque = image[:, :, 3] == 255 if image.shape[2] == 4 else np.ones(mask.shape, np.bool_)
                    matching = (opaque & targeted[y1:y2, x1:x2]
                                & np.all(image[:, :, :3] == background[y1:y2, x1:x2], axis=2)
                                & np.any(background[y1:y2, x1:x2] != 255, axis=2))
                    if np.any(matching):
                        duplicates[y1:y2, x1:x2] |= matching
                        reused.append((item, int(np.count_nonzero(matching))))
    prefix = f"revision_{revision_round}_background_page_{page}"
    glyphs[protected] = 0
    cleaned_text = cv2.inpaint(background, glyphs, 3, cv2.INPAINT_TELEA) if np.any(glyphs) else background
    assets, _ = extract_residual_objects(cleaned_text, occupied, asset_dir, project_id, page, asset_prefix=prefix)
    if not assets and not np.any(duplicates) and not np.any(glyphs):
        return []
    clear = duplicates.copy()
    for item in assets:
        path = asset_dir / Path(item["src"]).name
        image = image_io.imread(str(path), cv2.IMREAD_UNCHANGED)
        x, y, w, h = [int(item[k]) for k in ("x", "y", "width", "height")]
        if image is None or image.shape != (h, w, 4):
            for staged in assets:
                (asset_dir / Path(staged["src"]).name).unlink(missing_ok=True)
            return []
        clear[y:y + h, x:x + w] |= image[:, :, 3] > 0
        item["metadata"].update({"backgroundSeparated": True, "reconstructionStrategySource": "background_objectization"})
    cleaned = cleaned_text.copy()
    cleaned[clear] = 255
    staged_background = background_path.with_name(f".{background_path.stem}_{prefix}.png")
    try:
        if not image_io.imwrite(str(staged_background), cleaned):
            raise OSError("Could not save staged background")
        staged_background.replace(background_path)
    except OSError:
        staged_background.unlink(missing_ok=True)
        for staged in assets:
            (asset_dir / Path(staged["src"]).name).unlink(missing_ok=True)
        return []
    layout.setdefault("elements", []).extend(assets)
    for item, pixels in reused:
        item.setdefault("metadata", {}).update({"backgroundSeparated": True, "backgroundDuplicateClearedPixels": pixels})
    return assets + [item for item, _ in reused] + text_owners
