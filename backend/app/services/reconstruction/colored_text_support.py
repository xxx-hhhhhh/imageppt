"""Extract bounded colored supports beneath editable OCR lines."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extract_colored_text_supports(source_path: Path, layout: dict, asset_dir: Path,
                                  project_id: str, page_index: int) -> dict[str, int]:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        return {"coloredTextSupports": 0, "coloredSupportPixels": 0}
    height, width = source.shape[:2]
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    active = [item for item in layout.get("elements", []) if not any((item.get("metadata") or {}).get(key)
              for key in ("suppressed", "suppressRender", "ownedBy"))]
    assets = [item for item in active if item.get("type") == "image"]
    created: list[dict] = []
    claimed = np.zeros((height, width), np.uint8)
    for text in (item for item in active if item.get("type") == "text"):
        raw = (text.get("metadata") or {}).get("rawOCRBBox")
        box = _box_from_raw(raw, width, height) if isinstance(raw, list) and len(raw) == 4 else _box(text, width, height)
        if box is None:
            continue
        tx1, ty1, tx2, ty2 = box
        tw, th = tx2 - tx1, ty2 - ty1
        if tw < 24 or th < 10 or tw / th < 2.2:
            continue
        pad_x, pad_y = round(th * 4), round(th * 0.65)
        x1, y1, x2, y2 = max(0, tx1 - pad_x), max(0, ty1 - pad_y), min(width, tx2 + pad_x), min(height, ty2 + pad_y)
        window = hsv[y1:y2, x1:x2]
        vivid = (window[:, :, 1] >= 60) & (window[:, :, 2] >= 45)
        if np.mean(vivid) < 0.28:
            continue
        hue = int(np.median(window[:, :, 0][vivid]))
        hue_gap = np.abs(window[:, :, 0].astype(np.int16) - hue)
        color_mask = np.uint8(vivid & (np.minimum(hue_gap, 180 - hue_gap) <= 10)) * 255
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            cx, cy, cw, ch = cv2.boundingRect(contour)
            if cy == 0 or cy + ch >= y2 - y1:
                # The same color continues beyond the search window: it is
                # likely a large module surface rather than a local title bar.
                continue
            bx1, by1, bx2, by2 = x1 + cx, y1 + cy, x1 + cx + cw, y1 + cy + ch
            area = cw * ch
            if (cw < tw * 0.75 or ch < th * 0.8 or ch > th * 2.8 or cw / ch < 2.5
                    or area > width * height * 0.30 or area < tw * th * 0.5):
                continue
            overlap = max(0, min(tx2, bx2) - max(tx1, bx1)) * max(0, min(ty2, by2) - max(ty1, by1))
            if overlap < tw * th * 0.45 or cv2.contourArea(contour) / max(1, area) < 0.65:
                continue
            candidates.append((overlap, area, contour, (bx1, by1, bx2, by2)))
        if not candidates:
            continue
        _, _, contour, support_box = max(candidates, key=lambda item: (item[0], item[1]))
        bx1, by1, bx2, by2 = support_box
        if np.mean(claimed[by1:by2, bx1:bx2] > 0) > 0.25:
            continue
        alpha = np.zeros((by2 - by1, bx2 - bx1), np.uint8)
        local_contour = contour - np.array([[[bx1 - x1, by1 - y1]]])
        cv2.drawContours(alpha, [local_contour], -1, 255, -1)
        crop = source[by1:by2, bx1:bx2].copy()
        txl, tyt, txr, tyb = max(0, tx1 - bx1), max(0, ty1 - by1), min(bx2 - bx1, tx2 - bx1), min(by2 - by1, ty2 - by1)
        local_hsv = hsv[by1:by2, bx1:bx2]
        ink = np.zeros(alpha.shape, np.uint8)
        if txr > txl and tyb > tyt:
            inside = local_hsv[tyt:tyb, txl:txr]
            distance = np.abs(inside[:, :, 0].astype(np.int16) - hue)
            not_support = ((inside[:, :, 1] < 45) | (np.minimum(distance, 180 - distance) > 13))
            ink[tyt:tyb, txl:txr] = np.uint8(not_support) * 255
        if np.mean(ink > 0) > 0.25:
            continue
        ink = cv2.dilate(ink, np.ones((3, 3), np.uint8), iterations=1)
        cleaned = cv2.inpaint(crop, ink, 3, cv2.INPAINT_TELEA) if np.any(ink) else crop
        identifier = f"colored_support_page_{page_index}_{len(created) + 1:03d}"
        path = asset_dir / f"{identifier}.png"
        if not cv2.imwrite(str(path), np.dstack((cleaned, alpha))):
            continue
        overlapping = [item for item in assets if _overlap(_box(item, width, height), support_box) > 0]
        z_index = max([int(item.get("zIndex") or 0) for item in overlapping], default=0) + 1
        text["zIndex"] = max(int(text.get("zIndex") or 0), z_index + 1)
        created.append({"id": identifier, "type": "image", "x": bx1, "y": by1, "width": bx2 - bx1, "height": by2 - by1,
                        "rotation": 0, "zIndex": z_index, "groupId": text.get("groupId"),
                        "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1},
                        "metadata": {"reconstructionStrategy": "cutout_image", "reconstructionStrategySource": "colored_text_support",
                                     "layerRole": "container", "editableTextIds": [text.get("id")], "textCleaned": True}})
        _remove_replaced_pixels(overlapping, support_box, alpha, asset_dir, width, height)
        for shape in active:
            if shape.get("type") not in {"rectangle", "roundedRectangle", "ellipse"}:
                continue
            shape_box = _box(shape, width, height)
            intersection = _overlap(shape_box, support_box)
            if shape_box and intersection / max(1, (shape_box[2] - shape_box[0]) * (shape_box[3] - shape_box[1])) >= 0.85:
                shape.setdefault("metadata", {}).update({"suppressed": True, "suppressRender": True, "ownedBy": identifier})
        claimed[by1:by2, bx1:bx2][alpha > 0] = 255
    layout.setdefault("elements", []).extend(created)
    return {"coloredTextSupports": len(created), "coloredSupportPixels": int(np.count_nonzero(claimed))}


def _remove_replaced_pixels(assets: list[dict], support_box: tuple[int, int, int, int], alpha: np.ndarray,
                            asset_dir: Path, width: int, height: int) -> None:
    sx1, sy1, sx2, sy2 = support_box
    for item in assets:
        box = _box(item, width, height)
        if box is None:
            continue
        ax1, ay1, ax2, ay2 = box
        x1, y1, x2, y2 = max(sx1, ax1), max(sy1, ay1), min(sx2, ax2), min(sy2, ay2)
        if x2 <= x1 or y2 <= y1:
            continue
        path = asset_dir / Path(str(item.get("src") or "")).name
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
        if image is None or image.ndim != 3:
            continue
        if image.shape[2] == 3:
            image = np.dstack((image, np.full(image.shape[:2], 255, np.uint8)))
        mask = np.zeros((ay2 - ay1, ax2 - ax1), np.uint8)
        mask[y1 - ay1:y2 - ay1, x1 - ax1:x2 - ax1] = alpha[y1 - sy1:y2 - sy1, x1 - sx1:x2 - sx1]
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        image[:, :, 3][mask > 32] = 0
        cv2.imwrite(str(path), image)


def _box(item: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    x, y = float(item.get("x") or 0), float(item.get("y") or 0)
    return _box_from_raw([x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)], width, height)


def _box_from_raw(raw: list, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [round(float(value)) for value in raw]
    box = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    return box if box[2] > box[0] and box[3] > box[1] else None


def _overlap(left: tuple[int, int, int, int] | None, right: tuple[int, int, int, int]) -> int:
    if left is None:
        return 0
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(0, min(left[3], right[3]) - max(left[1], right[1]))
