from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path


FONT_FALLBACKS = {
    "serif": ["SimSun", "STSong", "Noto Serif CJK SC", "Times New Roman"],
    "sans": ["Microsoft YaHei", "DengXian", "Noto Sans CJK SC", "Arial"],
    "bold-sans": ["SimHei", "Microsoft YaHei", "Arial"],
    "calligraphy": ["KaiTi", "STKaiti", "FangSong"],
    "display": ["FZCuHeiSongS-B-GB", "FZDaBiaoSong-B06S", "SimHei", "Microsoft YaHei"],
    "monospace": ["Consolas", "Courier New"],
}

_KNOWN_FILES = {
    "SimSun": "simsun.ttc",
    "Microsoft YaHei": "msyh.ttc",
    "DengXian": "Deng.ttf",
    "SimHei": "simhei.ttf",
    "KaiTi": "simkai.ttf",
    "FangSong": "simfang.ttf",
    "Consolas": "consola.ttf",
    "Arial": "arial.ttf",
    "Times New Roman": "times.ttf",
    "Courier New": "cour.ttf",
}


@lru_cache(maxsize=1)
def font_records() -> dict[str, tuple[str, Path | None]]:
    records: dict[str, tuple[str, Path | None]] = {}
    fonts_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for family, filename in _KNOWN_FILES.items():
        path = fonts_dir / filename
        if path.exists():
            records[family.casefold()] = (family, path)
    if os.name == "nt":
        try:
            import winreg

            key_path = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                for index in range(winreg.QueryInfoKey(key)[1]):
                    display_name, raw_path, _ = winreg.EnumValue(key, index)
                    family_text = re.sub(r"\s*\([^)]*\)\s*$", "", str(display_name)).strip()
                    path = Path(str(raw_path))
                    if not path.is_absolute():
                        path = fonts_dir / path
                    for family in re.split(r"\s*&\s*", family_text):
                        if family:
                            records.setdefault(family.casefold(), (family, path if path.exists() else None))
        except OSError:
            pass
    return records


def available_fonts() -> set[str]:
    return {record[0] for record in font_records().values()}


def resolve_font_path(family: str | None) -> Path | None:
    if not family:
        return None
    record = font_records().get(str(family).casefold())
    return record[1] if record else None


def match_font(category: str, *, bold: bool = False, calligraphic: bool = False, cjk: bool = True) -> str:
    normalized = str(category or "").strip().lower().replace("_", "-")
    if calligraphic:
        normalized = "calligraphy"
    elif bold and normalized in {"sans", "sans-serif", "bold sans", "bold-sans"}:
        normalized = "bold-sans"
    elif normalized in {"sans-serif", "bold sans"}:
        normalized = "sans" if normalized == "sans-serif" else "bold-sans"
    if normalized not in FONT_FALLBACKS:
        normalized = "serif" if cjk else "sans"
    records = font_records()
    for candidate in FONT_FALLBACKS[normalized]:
        record = records.get(candidate.casefold())
        if record:
            return record[0]
    generic = "SimSun" if cjk else "Arial"
    record = records.get(generic.casefold())
    return record[0] if record else generic


__all__ = ["FONT_FALLBACKS", "available_fonts", "font_records", "match_font", "resolve_font_path"]
