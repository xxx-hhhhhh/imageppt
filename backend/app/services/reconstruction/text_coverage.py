from __future__ import annotations

from app.services.visual_qa.analyzer import _font


def fit_text_to_ocr_lines(layout: dict) -> None:
    """Constrain each editable OCR line to its detected position and width."""
    for element in layout.get("elements", []):
        metadata = element.get("metadata") or {}
        raw = metadata.get("rawOCRBBox")
        if element.get("type") != "text" or not isinstance(raw, list) or len(raw) != 4:
            continue
        if any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        text = str(element.get("text") or "").replace("\n", " ").strip()
        if not text:
            continue
        x1, y1, x2, y2 = map(float, raw)
        target_width = max(4.0, x2 - x1)
        target_height = max(4.0, y2 - y1)
        style = element.setdefault("style", {})
        size = min(float(style.get("fontSize") or target_height), target_height * 1.12)
        while size > 6:
            font = _font({**style, "fontSize": size})
            if font.getlength(text) <= target_width * 1.08:
                break
            size -= 0.5
        style["fontSize"] = round(max(6.0, size), 2)
        if "refinedFontSize" in style:
            style["refinedFontSize"] = style["fontSize"]
        element.update({"x": x1, "y": y1, "width": min(float(layout.get("slide", {}).get("width") or x2) - x1, max(target_width * 1.08, font.getlength(text) + 2)), "height": max(target_height * 1.4, size * 1.4)})
        metadata["textboxBBox"] = [element["x"], element["y"], element["x"] + element["width"], element["y"] + element["height"]]


def suppress_text_like_assets(layout: dict) -> int:
    """Discard false visual crops whose pixels are mostly an editable title."""
    texts = [item for item in layout.get("elements", []) if item.get("type") == "text" and item.get("role") in {"main_title", "section_title", "subtitle"}]
    suppressed = 0
    for asset in layout.get("elements", []):
        metadata = asset.get("metadata") or {}
        if asset.get("type") != "image" or not metadata.get("wholeBadgeAsset") or metadata.get("suppressed"):
            continue
        ax, ay = float(asset.get("x") or 0), float(asset.get("y") or 0)
        aw, ah = float(asset.get("width") or 0), float(asset.get("height") or 0)
        if aw * ah <= 0:
            continue
        for text in texts:
            tmeta = text.get("metadata") or {}
            raw = tmeta.get("rawOCRBBox")
            if tmeta.get("suppressRender") or not isinstance(raw, list) or len(raw) != 4:
                continue
            x1, y1, x2, y2 = map(float, raw)
            overlap = max(0, min(ax + aw, x2) - max(ax, x1)) * max(0, min(ay + ah, y2) - max(ay, y1))
            if overlap / (aw * ah) >= 0.5:
                metadata["suppressed"] = True
                metadata["fallbackReason"] = "editable_title_owns_region"
                suppressed += 1
                break
    return suppressed


def measure_text_coverage(layout: dict, detected_count: int) -> dict[str, int | float]:
    """Count OCR lines represented by active editable textboxes."""
    editable_ids: set[str] = set()
    for element in layout.get("elements", []):
        metadata = element.get("metadata") or {}
        if element.get("type") != "text" or not str(element.get("text") or "").strip():
            continue
        if any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        ids = metadata.get("sourceOcrIds") or [element.get("id")]
        editable_ids.update(str(value) for value in ids if value)
    detected = max(0, int(detected_count))
    editable = min(detected, len(editable_ids))
    return {
        "detectedTextCount": detected,
        "editableTextCount": editable,
        "nonEditableTextCount": detected - editable,
        "editableTextCoverage": round(editable / detected, 4) if detected else 1.0,
    }
