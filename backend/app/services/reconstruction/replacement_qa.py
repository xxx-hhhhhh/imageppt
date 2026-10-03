"""Local visual check for replacing source pixels with editable objects."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _bounds(item: dict, width: int, height: int, *, source_text: bool = False, padding: int = 0) -> tuple[int, int, int, int]:
    raw = (item.get("metadata") or {}).get("rawOCRBBox") if source_text else None
    try:
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            x, y, right, bottom = map(float, raw)
        else:
            x, y = float(item.get("x") or 0), float(item.get("y") or 0)
            right, bottom = x + float(item.get("width") or 0), y + float(item.get("height") or 0)
    except (TypeError, ValueError):
        return _bounds({**item, "metadata": {}}, width, height) if source_text else (0, 0, 0, 0)
    return max(0, round(x) - padding), max(0, round(y) - padding), min(width, round(right) + padding), min(height, round(bottom) + padding)


def check_replacement_regions(source_path: Path, before_path: Path, after_path: Path, layout: dict, report_path: Path, comparison_path: Path, ghosting_before: int, ghosting_after: int) -> dict:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    before = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    after = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
    if source is None or before is None or after is None or source.shape != before.shape or source.shape != after.shape:
        raise ValueError("Replacement QA images are missing or have mismatched dimensions")
    height, width = source.shape[:2]
    checked = 0
    worsened = []
    for item in layout.get("elements", []):
        metadata = item.get("metadata") or {}
        if item.get("type") not in {"image", "rectangle", "roundedRectangle", "ellipse", "line", "arrow"} or any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        x1, y1, x2, y2 = _bounds(item, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        mask = np.ones((y2 - y1, x2 - x1), bool)
        for text in layout.get("elements", []):
            if text.get("type") != "text" or any((text.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
                continue
            # Match the two-pixel source erasure margin. Glyph antialiasing just
            # outside OCR bounds is text, while the wider editor box is not.
            tx1, ty1, tx2, ty2 = _bounds(text, width, height, source_text=True, padding=2)
            left, top, right, bottom = max(x1, tx1), max(y1, ty1), min(x2, tx2), min(y2, ty2)
            if right > left and bottom > top:
                mask[top - y1:bottom - y1, left - x1:right - x1] = False
        if np.count_nonzero(mask) < 20:
            continue
        original = source[y1:y2, x1:x2].astype(np.int16)
        old_error = float(np.mean(np.abs(original - before[y1:y2, x1:x2].astype(np.int16))[mask]))
        new_error = float(np.mean(np.abs(original - after[y1:y2, x1:x2].astype(np.int16))[mask]))
        old_pixels = before[y1:y2, x1:x2].astype(np.int16)
        new_pixels = after[y1:y2, x1:x2].astype(np.int16)
        # A pale plate can disappear without exceeding the average color-error gate.
        lost = mask & (np.max(255 - original, axis=2) >= 6) & (np.max(np.abs(original - old_pixels), axis=2) <= 6) & (np.max(255 - old_pixels, axis=2) > 3) & (np.max(255 - new_pixels, axis=2) <= 3)
        lost_pixels = int(np.count_nonzero(lost))
        checked += 1
        if (new_error > old_error + 8 and new_error > 25) or lost_pixels > max(20, np.count_nonzero(mask) * .02):
            worsened.append({"elementId": item.get("id"), "bbox": [x1, y1, x2, y2], "errorBefore": round(old_error, 2), "errorAfter": round(new_error, 2), "lostVisualPixels": lost_pixels})
    report = {"checkedRegions": checked, "ghostingBefore": ghosting_before, "ghostingAfter": ghosting_after, "worsenedRegions": worsened, "safe": ghosting_after <= ghosting_before and not worsened, "improved": ghosting_after < ghosting_before and not worsened}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    cv2.imwrite(str(comparison_path), np.concatenate((before, after), axis=1))
    return report
