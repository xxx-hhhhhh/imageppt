from __future__ import annotations

import re
from typing import Any


FONT_CLASSES = {"serif", "sans", "bold-sans", "calligraphy", "display", "monospace", "unknown"}


def classify_font(text: str, role: str, bbox: list[float] | None = None, crop: Any | None = None) -> str:
    value = str(text or "")
    if role == "footer" or (role == "caption" and any(char in value for char in "，。！？")):
        return "calligraphy" if any("\u4e00" <= char <= "\u9fff" for char in value) else "display"
    if value and re.fullmatch(r"[\d\s.,%+\-/:]+", value):
        return "monospace"
    if role in {"main_title", "subtitle", "section_title"}:
        return "serif"
    if role in {"card_title", "label", "badge"}:
        return "bold-sans"
    if bbox and bbox[2] - bbox[0] > max(1, bbox[3] - bbox[1]) * 9:
        return "sans"
    return "sans" if role in {"body", "caption"} else "unknown"
