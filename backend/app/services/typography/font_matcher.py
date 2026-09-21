from __future__ import annotations

import os
from functools import lru_cache


FONT_CANDIDATES = {
    "serif": "SimSun",
    "sans": "Microsoft YaHei",
    "sans-serif": "Microsoft YaHei",
    "bold-sans": "SimHei",
    "bold sans": "SimHei",
    "display": "Georgia",
    "monospace": "Consolas",
    "calligraphy": "KaiTi",
}


@lru_cache(maxsize=1)
def available_fonts() -> set[str]:
    # Keep the logical candidates even when Windows font enumeration is not
    # available in a headless test runner.
    fonts = set(FONT_CANDIDATES.values()) | {"Arial", "Calibri", "Times New Roman", "FangSong"}
    windows_fonts = os.environ.get("WINDIR")
    if windows_fonts:
        fonts_dir = os.path.join(windows_fonts, "Fonts")
        if os.path.isdir(fonts_dir):
            fonts.update(os.path.splitext(name)[0] for name in os.listdir(fonts_dir) if name.lower().endswith((".ttf", ".otf")))
    return fonts


def match_font(category: str, *, bold: bool = False, calligraphic: bool = False, cjk: bool = True) -> str:
    if calligraphic:
        preferred = "calligraphy"
    elif bold and category in {"sans-serif", "sans", "bold-sans", "bold sans"}:
        preferred = "bold-sans"
    elif category in {"sans-serif", "sans"}:
        preferred = "sans"
    elif category in FONT_CANDIDATES:
        preferred = category
    else:
        preferred = "serif" if cjk else "sans-serif"
    candidate = FONT_CANDIDATES[preferred]
    return candidate if candidate in available_fonts() else ("SimSun" if cjk else "Arial")
