from __future__ import annotations

import re
from typing import Any

from app.services.typography.font_matcher import match_font, resolve_font_path


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
    colorful = _is_colorful(color)
    if width / height > 10 and len(text.strip()) < 12 and colorful:
        return "badge"
    if y > image_height * 0.86:
        return "footer"
    if len(text.strip()) <= 12 and colorful:
        return "label"
    if height < image_height * 0.018 and len(text.strip()) <= 18:
        return "caption"
    return "body"


def _is_colorful(value: str) -> bool:
    text = str(value or "").lstrip("#")
    if len(text) != 6:
        return False
    try:
        channels = [int(text[index:index + 2], 16) for index in (0, 2, 4)]
    except ValueError:
        return False
    return max(channels) - min(channels) >= 28


def estimate_font_size(text: str, bbox: list[float], role: str, *, textbox_width: float | None = None, font_family: str | None = None) -> float:
    _, _, x2, y2 = bbox
    height = max(6.0, y2 - bbox[1])
    width = max(6.0, x2 - bbox[0])
    line_count = max(1, text.count("\n") + 1)
    line_height = height / line_count
    height_factor = {
        "main_title": 0.78,
        "section_title": 0.82,
        "subtitle": 0.82,
        "card_title": 0.86,
        "label": 0.86,
        "label_text": 0.86,
        "badge": 0.84,
        "slogan": 0.82,
        "caption": 0.82,
        "footer": 0.8,
        "body": 0.9,
        "body_text": 0.9,
    }.get(role, 0.86)
    height_estimate = line_height * height_factor
    target_width = max(width, float(textbox_width or width))
    width_estimate = _font_size_for_width(text, target_width, font_family)
    size = min(height_estimate, width_estimate)
    return round(max(7.0, min(120.0, size)), 1)


def _font_size_for_width(text: str, target_width: float, family: str | None) -> float:
    lines = str(text or "").splitlines() or [""]
    path = resolve_font_path(family)
    if path:
        try:
            from PIL import ImageFont

            probe_size = 100
            font = ImageFont.truetype(str(path), probe_size)
            measured = max(float(font.getlength(line)) for line in lines)
            if measured > 0:
                return target_width * probe_size / measured * 0.97
        except (OSError, AttributeError):
            pass
    longest = max(lines, key=len, default="")
    cjk = len(re.findall(r"[\u3400-\u9fff]", longest))
    latin = max(0, len(longest) - cjk)
    units = max(1.0, cjk + latin * 0.56)
    return target_width / units * 0.97


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
