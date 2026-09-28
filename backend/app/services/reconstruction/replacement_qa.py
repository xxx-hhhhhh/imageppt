"""Local visual check for replacing source pixels with editable objects."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


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
        x1 = max(0, round(float(item.get("x") or 0)))
        y1 = max(0, round(float(item.get("y") or 0)))
        x2 = min(width, round(x1 + float(item.get("width") or 0)))
        y2 = min(height, round(y1 + float(item.get("height") or 0)))
        if x2 <= x1 or y2 <= y1:
            continue
        mask = np.ones((y2 - y1, x2 - x1), bool)
        for text in layout.get("elements", []):
            if text.get("type") != "text" or any((text.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
                continue
            tx1, ty1 = round(float(text.get("x") or 0)), round(float(text.get("y") or 0))
            tx2, ty2 = tx1 + round(float(text.get("width") or 0)), ty1 + round(float(text.get("height") or 0))
            mask[max(0, ty1 - y1):min(y2 - y1, ty2 - y1), max(0, tx1 - x1):min(x2 - x1, tx2 - x1)] = False
        if np.count_nonzero(mask) < 20:
            continue
        original = source[y1:y2, x1:x2].astype(np.int16)
        old_error = float(np.mean(np.abs(original - before[y1:y2, x1:x2].astype(np.int16))[mask]))
        new_error = float(np.mean(np.abs(original - after[y1:y2, x1:x2].astype(np.int16))[mask]))
        checked += 1
        if new_error > old_error + 8 and new_error > 25:
            worsened.append({"elementId": item.get("id"), "bbox": [x1, y1, x2, y2], "errorBefore": round(old_error, 2), "errorAfter": round(new_error, 2)})
    report = {"checkedRegions": checked, "ghostingBefore": ghosting_before, "ghostingAfter": ghosting_after, "worsenedRegions": worsened, "safe": ghosting_after <= ghosting_before and not worsened, "improved": ghosting_after < ghosting_before and not worsened}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    cv2.imwrite(str(comparison_path), np.concatenate((before, after), axis=1))
    return report
