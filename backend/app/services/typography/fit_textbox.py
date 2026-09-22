from __future__ import annotations

import re


def fit_textbox(text: str, width: float, height: float, font_size: float, *, original_line_count: int | None = None, preserve_line_count: bool = False) -> dict[str, float | str | int | bool]:
    size = max(5.0, float(font_size))
    original = size
    lines = str(text or "").splitlines() or [""]
    while size > original * 0.8:
        estimated_width = max((_line_width(line, size) for line in lines), default=0.0)
        estimated_height = len(lines) * size * 1.12
        if estimated_width <= max(1, width) and estimated_height <= max(1, height):
            break
        size *= 0.98
    expected_lines = max(1, int(original_line_count or len(lines)))
    return {
        "fontSize": round(max(size, original * 0.8), 2),
        "text": "\n".join(lines),
        "originalLineCount": expected_lines,
        "preserveLineCount": bool(preserve_line_count),
    }


def _line_width(line: str, size: float) -> float:
    cjk = len(re.findall(r"[\u3400-\u9fff]", line))
    latin = max(0, len(line) - cjk)
    return cjk * size + latin * size * 0.56
