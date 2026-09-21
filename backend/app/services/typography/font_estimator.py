from __future__ import annotations

import re
from typing import Any

from app.services.typography.font_matcher import match_font


TEXT_ROLES = {"main_title", "section_title", "subtitle", "card_title", "body", "caption", "label", "footer", "number", "badge"}


def _cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def infer_text_role(text: str, bbox: list[float], image_width: int, image_height: int, color: str = "", *, region_hint: str | None = None) -> str:
    if region_hint in TEXT_ROLES:
        return region_hint
    x, y, x2, y2 = bbox
    width, height = max(1.0, x2 - x), max(1.0, y2 - y)
    if y < image_height * 0.16 and width > image_width * 0.16:
        return "main_title"
    if y < image_height * 0.25:
        return "subtitle"
    if len(text.strip()) <= 3 and height > image_height * 0.025:
        return "section_title"
    if width / height > 10 and len(text.strip()) < 12:
        return "badge"
    if y > image_height * 0.86:
        return "footer"
    if len(text.strip()) <= 12 and color.upper() not in {"#111111", "#111827", "#000000"}:
        return "label"
    if height < image_height * 0.025:
        return "caption"
    return "body"


def estimate_font_size(text: str, bbox: list[float], role: str) -> float:
    _, _, x2, y2 = bbox
    height = max(6.0, y2 - bbox[1])
    width = max(6.0, x2 - bbox[0])
    count = max(1, len(text.strip()))
    line_count = max(1, text.count("\n") + 1)
    line_estimate = height * (0.78 if line_count == 1 else 0.68)
    width_estimate = width / count * (1.35 if _cjk(text) else 1.2)
    size = min(line_estimate, width_estimate) if role in {"body", "caption"} else line_estimate
    scale = {"main_title": 1.08, "section_title": 1.0, "subtitle": 0.9, "card_title": 1.0, "label": 0.9, "badge": 0.85, "footer": 0.8}.get(role, 0.86)
    return round(max(7.0, min(120.0, size * scale)), 1)


def estimate_style(text: str, bbox: list[float], image_width: int, image_height: int, color: str, *, region_hint: str | None = None, crop: Any | None = None) -> dict[str, Any]:
    from app.services.typography.font_classifier import classify_font

    role = infer_text_role(text, bbox, image_width, image_height, color, region_hint=region_hint)
    font_class = classify_font(text, role, bbox, crop)
    bold = font_class == "bold-sans" or role in {"main_title", "section_title", "card_title", "label", "badge"}
    category = "sans" if font_class in {"sans", "bold-sans"} else font_class
    legacy_role = {"body": "body_text", "label": "label_text"}.get(role, role)
    return {
        "fontFamily": match_font(category, bold=bold, calligraphic=font_class == "calligraphy", cjk=_cjk(text)),
        "fontClass": font_class,
        "fontSize": estimate_font_size(text, bbox, role),
        "fontWeight": 700 if bold else 400,
        "fontStyle": "normal",
        "color": color or "#111827",
        "align": "left",
        "verticalAlign": "middle" if role in {"main_title", "section_title", "subtitle", "badge", "label"} else "top",
        "lineSpacing": 1.12 if role in {"body", "caption"} else 1.0,
        "letterSpacing": 0,
        "role": legacy_role,
        "textRole": role,
    }
