"""Separate compact colored icons accidentally included in OCR text lines."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


def extract_leading_text_icons(source_path: Path, layout: dict, asset_dir: Path,
                               project_id: str, page_index: int) -> int:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        return 0
    height, width = source.shape[:2]
    created: list[dict] = []
    for text in list(layout.get("elements", [])):
        metadata = text.get("metadata") or {}
        raw = metadata.get("rawOCRBBox")
        if (text.get("type") != "text" or not str(text.get("text") or "").strip()
                or any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy"))
                or not isinstance(raw, list) or len(raw) != 4):
            continue
        try:
            x1, y1, x2, y2 = [round(float(value)) for value in raw]
        except (TypeError, ValueError):
            continue
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        line_width, line_height = x2 - x1, y2 - y1
        if line_height < 12 or line_width < line_height * 3:
            continue
        crop = source[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lead_width = min(round(line_height * 1.7), round(line_width * 0.28))
        body = hsv[:, max(lead_width + 4, round(line_width * 0.3)):]
        body_ink = (body[:, :, 2] <= 210) & (body[:, :, 1] >= 30)
        if np.count_nonzero(body_ink) < max(30, line_height):
            continue
        reference_hue = int(np.median(body[:, :, 0][body_ink]))
        reference_saturation = int(np.median(body[:, :, 1][body_ink]))
        lead = hsv[:, :lead_width]
        hue_delta = np.abs(lead[:, :, 0].astype(np.int16) - reference_hue)
        hue_delta = np.minimum(hue_delta, 180 - hue_delta)
        distinct = (lead[:, :, 1] >= 55) & (lead[:, :, 2] <= 235)
        if reference_saturation >= 50:
            distinct &= hue_delta >= 14
        else:
            distinct &= lead[:, :, 1] >= reference_saturation + 35
        binary = cv2.morphologyEx(np.uint8(distinct) * 255, cv2.MORPH_CLOSE,
                                  np.ones((2, 2), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        candidates = []
        for index in range(1, count):
            cx, cy, cw, ch, pixels = [int(value) for value in stats[index]]
            if (cx > line_height * 0.65 or cw < max(7, line_height * 0.35)
                    or ch < max(7, line_height * 0.5) or ch > line_height * 1.1
                    or pixels < max(28, round(line_height * line_height * 0.12))
                    or pixels / max(1, cw * ch) < 0.30):
                continue
            right = cx + cw
            if right >= lead_width - 1:
                continue
            following = hsv[:, right: min(line_width, right + max(8, line_height))]
            ink_after = (following[:, :, 2] <= 210) & (following[:, :, 1] >= 30)
            ink_after &= np.minimum(np.abs(following[:, :, 0].astype(np.int16) - reference_hue),
                                    180 - np.abs(following[:, :, 0].astype(np.int16) - reference_hue)) <= 12
            columns = np.flatnonzero(np.count_nonzero(ink_after, axis=0) >= 2)
            if len(columns) == 0 or int(columns[0]) < 3:
                continue
            candidates.append((pixels, index, cx, cy, cw, ch, right + int(columns[0])))
        if not candidates:
            continue
        _, index, cx, cy, cw, ch, body_start = max(candidates)
        bx1, by1 = x1 + max(0, cx - 2), y1 + max(0, cy - 2)
        bx2, by2 = x1 + min(line_width, cx + cw + 2), y1 + min(line_height, cy + ch + 2)
        # Do not duplicate a visual that already has a dedicated small owner.
        if any(item.get("type") == "image" and not any((item.get("metadata") or {}).get(key)
               for key in ("suppressed", "suppressRender", "ownedBy"))
               and _covers(item, (bx1, by1, bx2, by2)) for item in layout.get("elements", [])):
            continue
        alpha = np.uint8(labels[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1] == index) * 255
        alpha = cv2.dilate(alpha, np.ones((3, 3), np.uint8), iterations=1)
        if np.count_nonzero(alpha) < 24:
            continue
        asset_dir.mkdir(parents=True, exist_ok=True)
        identifier = f"leading_icon_page_{page_index}_{uuid4().hex[:10]}"
        path = asset_dir / f"{identifier}.png"
        if not cv2.imwrite(str(path), np.dstack((source[by1:by2, bx1:bx2], alpha))):
            continue
        new_start = x1 + body_start
        old_x = float(text.get("x") or x1)
        old_right = old_x + float(text.get("width") or line_width)
        text["x"] = max(old_x, new_start)
        text["width"] = max(4.0, old_right - float(text["x"]))
        metadata["rawOCRBBox"] = [float(new_start), float(y1), float(x2), float(y2)]
        metadata["leadingVisualOwner"] = identifier
        text["metadata"] = metadata
        created.append({"id": identifier, "type": "image", "x": bx1, "y": by1,
                        "width": bx2 - bx1, "height": by2 - by1, "rotation": 0,
                        "zIndex": max(1, int(text.get("zIndex") or 20) - 1),
                        "groupId": text.get("groupId"),
                        "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1},
                        "metadata": {"reconstructionStrategy": "cutout_image",
                                     "reconstructionStrategySource": "leading_text_icon",
                                     "layerRole": "visual", "preserveWholeAsset": True,
                                     "associatedTextId": text.get("id")}})
    layout.setdefault("elements", []).extend(created)
    return len(created)


def _covers(item: dict, box: tuple[int, int, int, int]) -> bool:
    x = float(item.get("x") or 0)
    y = float(item.get("y") or 0)
    w = float(item.get("width") or 0)
    h = float(item.get("height") or 0)
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    intersection = max(0.0, min(x + w, box[2]) - max(x, box[0])) * max(0.0, min(y + h, box[3]) - max(y, box[1]))
    return intersection / area >= 0.85 and w * h <= area * 4
