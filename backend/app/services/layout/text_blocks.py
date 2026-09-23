from __future__ import annotations

import copy
import re
from typing import Any

from app.services.typography.font_estimator import estimate_font_size


_PARAGRAPH_ROLES = {"body", "body_text", "caption", "footer"}
_SINGLE_LINE_ROLES = {"main_title", "subtitle", "section_title", "card_title", "label", "label_text", "slogan", "badge"}


def group_text_elements(elements: list[dict[str, Any]], image_width: int, image_height: int, *, merge_paragraphs: bool = True) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Merge OCR fragments into visual text blocks while retaining source geometry."""
    source = [copy.deepcopy(item) for item in elements]
    same_line = _merge_same_line(source)
    blocks = _merge_paragraph_lines(same_line) if merge_paragraphs else same_line
    merged = sum(max(0, len(_source_ids(item)) - 1) for item in blocks)
    single_line = 0
    for item in blocks:
        _finalize_textbox(item, image_width, image_height)
        if int((item.get("metadata") or {}).get("originalLineCount", 1)) == 1:
            single_line += 1
    return blocks, {"textBlocksMerged": merged, "singleLinePreserved": single_line}


def _merge_same_line(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending = sorted(elements, key=lambda item: (float(item.get("y", 0)), float(item.get("x", 0))))
    lines: list[dict[str, Any]] = []
    for item in pending:
        best: dict[str, Any] | None = None
        for candidate in reversed(lines[-8:]):
            if _same_visual_line(candidate, item):
                best = candidate
                break
        if best is None:
            item.setdefault("metadata", {})["sourceOcrIds"] = [item.get("id")]
            lines.append(item)
        else:
            _merge_into(best, item, same_line=True)
    return sorted(lines, key=lambda item: (float(item.get("y", 0)), float(item.get("x", 0))))


def _merge_paragraph_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for line in lines:
        role = _role(line)
        best: dict[str, Any] | None = None
        if role in _PARAGRAPH_ROLES:
            for candidate in reversed(blocks[-8:]):
                if _same_paragraph(candidate, line):
                    best = candidate
                    break
        if best is None:
            blocks.append(line)
        else:
            _merge_into(best, line, same_line=False)
    return blocks


def _same_visual_line(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if _role(left) != _role(right):
        return False
    lh, rh = float(left.get("height", 1)), float(right.get("height", 1))
    if max(lh, rh) / max(1.0, min(lh, rh)) > 1.45:
        return False
    center_delta = abs((float(left.get("y", 0)) + lh / 2) - (float(right.get("y", 0)) + rh / 2))
    if center_delta > max(lh, rh) * 0.38:
        return False
    left_edge = float(left.get("x", 0)) + float(left.get("width", 0))
    gap = float(right.get("x", 0)) - left_edge
    if gap < -max(lh, rh) * 0.4 or gap > max(10.0, min(72.0, max(lh, rh) * 2.2)):
        return False
    if gap <= max(lh, rh) * 0.5:
        left_style, right_style = left.get("style") or {}, right.get("style") or {}
        ls, rs = float(left_style.get("fontSize", 20) or 20), float(right_style.get("fontSize", 20) or 20)
        return max(ls, rs) / max(1.0, min(ls, rs)) <= 1.35
    return _style_compatible(left, right)


def _same_paragraph(block: dict[str, Any], line: dict[str, Any]) -> bool:
    if _role(block) != _role(line) or not _style_compatible(block, line):
        return False
    bx1, by1, bx2, by2 = _bounds(block)
    lx1, ly1, lx2, _ = _bounds(line)
    line_height = float(line.get("height", 1))
    gap = ly1 - by2
    if gap < -line_height * 0.25 or gap > max(10.0, line_height * 0.9):
        return False
    overlap = min(bx2, lx2) - max(bx1, lx1)
    aligned = abs(bx1 - lx1) <= max(18.0, line_height * 1.4)
    return overlap > min(bx2 - bx1, lx2 - lx1) * 0.2 or aligned


def _style_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_style, right_style = left.get("style") or {}, right.get("style") or {}
    ls, rs = float(left_style.get("fontSize", 20) or 20), float(right_style.get("fontSize", 20) or 20)
    if max(ls, rs) / max(1.0, min(ls, rs)) > 1.35:
        return False
    return _color_distance(left_style.get("color"), right_style.get("color")) <= 90


def _merge_into(target: dict[str, Any], item: dict[str, Any], *, same_line: bool) -> None:
    target_lines = target.get("lines") or [{"text": target.get("text", ""), "bbox": list(_bounds(target))}]
    item_lines = item.get("lines") or [{"text": item.get("text", ""), "bbox": list(_bounds(item))}]
    if same_line:
        left_text = str(target_lines[-1].get("text", ""))
        right_text = str(item_lines[0].get("text", ""))
        target_lines[-1]["text"] = left_text + _joiner(left_text, right_text) + right_text
        target_lines[-1]["bbox"] = list(_union(tuple(target_lines[-1].get("bbox") or _bounds(target)), tuple(item_lines[0].get("bbox") or _bounds(item))))
        target_lines.extend(item_lines[1:])
    else:
        target_lines.extend(item_lines)
    bounds = _union(_bounds(target), _bounds(item))
    target.update({"x": bounds[0], "y": bounds[1], "width": bounds[2] - bounds[0], "height": bounds[3] - bounds[1]})
    target["lines"] = target_lines
    target["text"] = "\n".join(str(line.get("text", "")) for line in target_lines)
    metadata = target.setdefault("metadata", {})
    metadata["sourceOcrIds"] = list(dict.fromkeys(_source_ids(target) + _source_ids(item)))


def _finalize_textbox(item: dict[str, Any], image_width: int, image_height: int) -> None:
    visual = _bounds(item)
    role = _role(item)
    line_count = max(1, len(item.get("lines") or []))
    horizontal_ratio = 0.12 if role in _SINGLE_LINE_ROLES else 0.06
    vertical_ratio = 0.18 if line_count == 1 else 0.08
    pad_x = max(3.0, (visual[2] - visual[0]) * horizontal_ratio)
    pad_y = max(1.0, (visual[3] - visual[1]) * vertical_ratio)
    textbox = (
        max(0.0, visual[0] - pad_x),
        max(0.0, visual[1] - pad_y),
        min(float(image_width), visual[2] + pad_x),
        min(float(image_height), visual[3] + pad_y),
    )
    item.update({"x": textbox[0], "y": textbox[1], "width": textbox[2] - textbox[0], "height": textbox[3] - textbox[1]})
    metadata = item.setdefault("metadata", {})
    metadata.update({
        "rawOCRBBox": list(visual),
        "visualTextBBox": list(visual),
        "textboxBBox": list(textbox),
        "originalLineCount": line_count,
        "preserveOriginalLineCount": role in _SINGLE_LINE_ROLES,
        "willReconstruct": True,
        "sourceContentPreserved": False,
    })
    style = item.setdefault("style", {})
    style["fontSize"] = estimate_font_size(
        str(item.get("text") or ""),
        list(visual),
        role,
        textbox_width=textbox[2] - textbox[0],
        font_family=style.get("fontFamily"),
    )


def _source_ids(item: dict[str, Any]) -> list[str]:
    values = (item.get("metadata") or {}).get("sourceOcrIds") or [item.get("id")]
    return [str(value) for value in values if value]


def _bounds(item: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y = float(item.get("x", 0)), float(item.get("y", 0))
    return x, y, x + float(item.get("width", 0)), y + float(item.get("height", 0))


def _union(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return min(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), max(left[3], right[3])


def _role(item: dict[str, Any]) -> str:
    style = item.get("style") or {}
    return str(item.get("role") or style.get("textRole") or style.get("role") or "body_text")


def _joiner(left: str, right: str) -> str:
    if not left or not right:
        return ""
    if re.search(r"[A-Za-z0-9]$", left) and re.match(r"^[A-Za-z0-9]", right):
        return " "
    return ""


def _color_distance(left: Any, right: Any) -> float:
    def rgb(value: Any) -> tuple[int, int, int]:
        text = str(value or "#111827").lstrip("#")
        if len(text) != 6:
            return 17, 24, 39
        try:
            return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
        except ValueError:
            return 17, 24, 39

    a, b = rgb(left), rgb(right)
    return sum((a[index] - b[index]) ** 2 for index in range(3)) ** 0.5


__all__ = ["group_text_elements"]
