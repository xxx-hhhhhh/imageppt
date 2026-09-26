from __future__ import annotations

import copy
import statistics
from collections import defaultdict
from typing import Any

from app.services.typography.font_matcher import font_records, match_font, resolve_font_path


_ROLE_ALIASES = {
    "body": "body_text",
    "label_text": "label",
    "badge": "label",
    "caption": "annotation",
    "footer": "slogan",
    "title": "main_title",
}

_ROLE_POLICY = {
    "main_title": {"fontClass": "serif", "weight": 700, "priority": ["SimSun", "STSong", "KaiTi", "SimHei"]},
    "section_title": {"fontClass": "serif", "weight": 700, "priority": ["SimSun", "STSong", "SimHei"]},
    "subtitle": {"fontClass": "sans", "weight": 400, "priority": ["Microsoft YaHei", "DengXian", "SimSun"]},
    "card_title": {"fontClass": "bold-sans", "weight": 700, "priority": ["SimHei", "Microsoft YaHei", "DengXian"]},
    "body_text": {"fontClass": "sans", "weight": 400, "priority": ["Microsoft YaHei", "DengXian", "Noto Sans CJK SC", "SimSun"]},
    "label": {"fontClass": "bold-sans", "weight": 700, "priority": ["SimHei", "Microsoft YaHei"]},
    "slogan": {"fontClass": "calligraphy", "weight": 400, "priority": ["KaiTi", "STKaiti", "SimSun"]},
    "annotation": {"fontClass": "sans", "weight": 400, "priority": ["Microsoft YaHei", "DengXian", "SimSun"]},
}

_SINGLE_LINE_ROLES = {"main_title", "section_title", "subtitle", "card_title", "label", "slogan"}


class TypographyLayoutRefiner:
    """Apply bounded typography and alignment polish after scene fusion.

    OCR/CV remain the source of truth for geometry.  This pass expands useful
    text boxes, measures the selected installed font, and makes only small
    alignment corrections so the page keeps its original composition.
    """

    def refine(self, layout: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int | bool]]:
        refined = copy.deepcopy(layout)
        width = float(refined.get("slide", {}).get("width") or 1)
        height = float(refined.get("slide", {}).get("height") or 1)
        stats: dict[str, int | bool] = {
            "typographyRefined": False,
            "fontRoleAssignments": 0,
            "fontFamilyAdjustments": 0,
            "fontSizeAdjustments": 0,
            "textPositionAdjustments": 0,
            "textboxResizeAdjustments": 0,
            "singleLinePreserved": 0,
            "pageAlignmentAdjustments": 0,
        }
        texts = [item for item in refined.get("elements", []) if item.get("type") == "text" and not _suppressed(item)]
        if not texts:
            return refined, stats

        for item in texts:
            self._refine_text(item, refined.get("elements", []), width, height, stats)
        page_changes = self._polish_page_alignment(texts, width, height)
        stats["pageAlignmentAdjustments"] = page_changes
        stats["textPositionAdjustments"] = int(stats["textPositionAdjustments"]) + page_changes
        for item in texts:
            if _font_role(item) != "body_text":
                continue
            raw = (item.get("metadata") or {}).get("rawOCRBBox")
            if isinstance(raw, list) and len(raw) == 4:
                anchored_x = max(0.0, min(max(0.0, width - float(item["width"])), float(raw[0])))
                if abs(float(item.get("x") or 0) - anchored_x) > 0.5:
                    item["x"] = anchored_x
                    item.setdefault("metadata", {})["refinedTextboxBBox"] = [anchored_x, float(item["y"]), anchored_x + float(item["width"]), float(item["y"]) + float(item["height"])]
                    item["metadata"]["textboxBBox"] = list(item["metadata"]["refinedTextboxBBox"])
                    stats["textPositionAdjustments"] = int(stats["textPositionAdjustments"]) + 1
        stats["typographyRefined"] = True
        refined.setdefault("metadata", {})["typographyLayoutRefinement"] = copy.deepcopy(stats)
        return refined, stats

    def _refine_text(
        self,
        item: dict[str, Any],
        elements: list[dict[str, Any]],
        page_width: float,
        page_height: float,
        stats: dict[str, int | bool],
    ) -> None:
        style = item.setdefault("style", {})
        metadata = item.setdefault("metadata", {})
        role = _font_role(item)
        policy = _ROLE_POLICY.get(role, _ROLE_POLICY["body_text"])
        old_family = str(style.get("fontFamily") or "")
        family = _select_family(list(policy["priority"]), str(policy["fontClass"]), role)
        style.update({
            "fontRole": role,
            "fontClass": policy["fontClass"],
            "fontFamily": family,
            "fontWeight": int(policy["weight"]),
            "fontPriority": list(policy["priority"]),
        })
        metadata["fontRole"] = role
        item["role"] = role
        stats["fontRoleAssignments"] = int(stats["fontRoleAssignments"]) + 1
        if old_family.casefold() != family.casefold():
            stats["fontFamilyAdjustments"] = int(stats["fontFamilyAdjustments"]) + 1

        original_box = _box(item)
        visual = _metadata_box(metadata.get("visualTextBBox") or metadata.get("rawOCRBBox"), original_box)
        initial_size = max(5.0, float(style.get("fontSize") or 20.0))
        textbox = _refined_textbox(item, visual, role, initial_size, family, page_width, page_height)
        item.update({"x": textbox[0], "y": textbox[1], "width": textbox[2] - textbox[0], "height": textbox[3] - textbox[1]})
        metadata["rawOCRBBox"] = list(_metadata_box(metadata.get("rawOCRBBox"), visual))
        metadata["visualTextBBox"] = list(visual)
        metadata["refinedTextboxBBox"] = list(textbox)
        metadata["textboxBBox"] = list(textbox)
        if abs((original_box[2] - original_box[0]) - item["width"]) > 0.5 or abs((original_box[3] - original_box[1]) - item["height"]) > 0.5:
            stats["textboxResizeAdjustments"] = int(stats["textboxResizeAdjustments"]) + 1
        if abs(original_box[0] - item["x"]) > 0.5 or abs(original_box[1] - item["y"]) > 0.5:
            stats["textPositionAdjustments"] = int(stats["textPositionAdjustments"]) + 1

        refined_size = _refine_font_size(item, visual, initial_size, family, role)
        style["estimatedFontSize"] = round(initial_size, 2)
        style["refinedFontSize"] = round(refined_size, 2)
        style["fontSizeAdjustment"] = round(refined_size - initial_size, 2)
        style["fontSize"] = round(refined_size, 2)
        if abs(refined_size - initial_size) >= 0.15:
            stats["fontSizeAdjustments"] = int(stats["fontSizeAdjustments"]) + 1

        line_count = max(1, int(metadata.get("originalLineCount") or len(item.get("lines") or []) or 1))
        preserve_single = line_count == 1 and role in _SINGLE_LINE_ROLES
        metadata["preserveOriginalLineCount"] = preserve_single or bool(metadata.get("preserveOriginalLineCount"))
        if preserve_single:
            stats["singleLinePreserved"] = int(stats["singleLinePreserved"]) + 1
        line_spacing = _line_spacing(role, line_count)
        style.update({
            "lineSpacing": line_spacing,
            "lineHeight": round(refined_size * line_spacing, 2),
            "paragraphSpacing": round(refined_size * (0.12 if role == "body_text" else 0.0), 2),
            "textInset": 0,
            "alignment": style.get("align", "left"),
        })
        self._refine_group_relation(item, elements, page_width, page_height, stats)

    def _refine_group_relation(
        self,
        item: dict[str, Any],
        elements: list[dict[str, Any]],
        page_width: float,
        page_height: float,
        stats: dict[str, int | bool],
    ) -> None:
        metadata = item.setdefault("metadata", {})
        group_id = item.get("groupId") or metadata.get("groupId")
        metadata["anchorRegion"] = group_id or "page"
        metadata["visualCenter"] = [round(item["x"] + item["width"] / 2, 2), round(item["y"] + item["height"] / 2, 2)]
        metadata["baselineReference"] = round(item["y"] + item["height"] * 0.82, 2)
        metadata["alignmentReference"] = "ocr-visual-bbox"
        metadata["neighborRelation"] = "none"
        if not group_id:
            return
        members = [candidate for candidate in elements if candidate is not item and (candidate.get("groupId") or (candidate.get("metadata") or {}).get("groupId")) == group_id]
        icon = next((candidate for candidate in members if (candidate.get("metadata") or {}).get("wholeBadgeAsset")), None)
        if icon is None:
            icon = next((candidate for candidate in members if (candidate.get("componentType") or (candidate.get("metadata") or {}).get("componentType")) == "iconCircle"), None)
        if icon is None:
            icon = _nearest_badge(item, elements, _font_role(item))
            if icon is not None:
                metadata["anchorRegion"] = icon.get("id")
        if not icon:
            return
        old_x, old_y = float(item["x"]), float(item["y"])
        icon_right = float(icon["x"]) + float(icon["width"])
        icon_bottom = float(icon["y"]) + float(icon["height"])
        visual = _metadata_box(metadata.get("visualTextBBox") or metadata.get("rawOCRBBox"), _box(item))
        same_row = abs((visual[1] + visual[3]) / 2 - (float(icon["y"]) + float(icon["height"]) / 2)) <= max(float(icon["height"]), visual[3] - visual[1]) * 0.72
        icon_is_left = float(icon["x"]) + float(icon["width"]) / 2 < (visual[0] + visual[2]) / 2
        below_badge = visual[1] >= icon_bottom - float(icon["height"]) * 0.18
        if below_badge:
            target_x = float(icon["x"]) + float(icon["width"]) / 2 - float(item["width"]) / 2
            item["x"] = _bounded(old_x, target_x, page_width * 0.025, 0.0, page_width - float(item["width"]))
            metadata["alignmentReference"] = "badge-center-x"
            metadata["neighborRelation"] = "below-badge"
        elif same_row and icon_is_left:
            gap = max(4.0, float(icon["width"]) * 0.08)
            target_x = max(visual[0], icon_right + gap)
            item["x"] = _bounded(old_x, target_x, page_width * 0.05, 0.0, page_width - float(item["width"]))
            target_y = float(icon["y"]) + float(icon["height"]) / 2 - float(item["height"]) / 2
            item["y"] = _bounded(old_y, target_y, page_height * 0.02, 0.0, page_height - float(item["height"]))
            metadata["alignmentReference"] = "badge-right-edge"
            metadata["neighborRelation"] = "right-of-badge"
        if abs(old_x - float(item["x"])) > 0.5 or abs(old_y - float(item["y"])) > 0.5:
            stats["textPositionAdjustments"] = int(stats["textPositionAdjustments"]) + 1

    def _polish_page_alignment(self, texts: list[dict[str, Any]], page_width: float, page_height: float) -> int:
        changes = 0
        by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in texts:
            by_role[_font_role(item)].append(item)
        for role, members in by_role.items():
            rows = _row_clusters(members, max(8.0, page_height * 0.025))
            for row in rows:
                if len(row) < 3:
                    continue
                target_y = statistics.median(float(item["y"]) for item in row)
                for item in row:
                    old_y = float(item["y"])
                    item["y"] = _bounded(old_y, target_y, page_height * 0.012, 0.0, page_height - float(item["height"]))
                    if abs(old_y - float(item["y"])) > 0.5:
                        changes += 1
                if role in {"card_title", "label"} and len(row) >= 3:
                    ordered = sorted(row, key=lambda item: float(item["x"]) + float(item["width"]) / 2)
                    first = float(ordered[0]["x"]) + float(ordered[0]["width"]) / 2
                    last = float(ordered[-1]["x"]) + float(ordered[-1]["width"]) / 2
                    step = (last - first) / max(1, len(ordered) - 1)
                    for index, item in enumerate(ordered[1:-1], start=1):
                        old_x = float(item["x"])
                        target_x = first + step * index - float(item["width"]) / 2
                        item["x"] = _bounded(old_x, target_x, page_width * 0.015, 0.0, page_width - float(item["width"]))
                        if abs(old_x - float(item["x"])) > 0.5:
                            changes += 1
        return changes


def _font_role(item: dict[str, Any]) -> str:
    style = item.get("style") or {}
    metadata = item.get("metadata") or {}
    value = str(style.get("fontRole") or item.get("role") or style.get("textRole") or style.get("role") or metadata.get("role") or "body_text")
    return _ROLE_ALIASES.get(value, value if value in _ROLE_POLICY else "body_text")


def _select_family(priority: list[str], font_class: str, role: str) -> str:
    records = font_records()
    for candidate in priority:
        record = records.get(candidate.casefold())
        if record:
            return record[0]
    return match_font(font_class, bold=role in {"main_title", "section_title", "card_title", "label"}, calligraphic=role == "slogan", cjk=True)


def _refined_textbox(
    item: dict[str, Any],
    visual: tuple[float, float, float, float],
    role: str,
    font_size: float,
    family: str,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    current = _box(item)
    visual_width, visual_height = visual[2] - visual[0], visual[3] - visual[1]
    pad_x_ratio = 0.1 if role in _SINGLE_LINE_ROLES else 0.045
    pad_y_ratio = 0.12 if role in _SINGLE_LINE_ROLES else 0.06
    text_width, _ = _measure_text(str(item.get("text") or ""), font_size, family, float((item.get("style") or {}).get("lineSpacing") or 1.0))
    desired_width = max(visual_width * (1 + pad_x_ratio * 2), text_width * 1.04)
    desired_height = visual_height * (1 + pad_y_ratio * 2)
    current_width, current_height = current[2] - current[0], current[3] - current[1]
    max_width = max(current_width, current_width * 1.15)
    max_height = max(current_height, current_height * 1.15)
    width = min(page_width, min(max_width, max(current_width, desired_width)))
    height = min(page_height, min(max_height, max(current_height, desired_height)))
    if role in _SINGLE_LINE_ROLES or str((item.get("style") or {}).get("align")) == "center":
        center_x = (visual[0] + visual[2]) / 2
        x = center_x - width / 2
    else:
        x = min(current[0], visual[0] - visual_width * pad_x_ratio)
    center_y = (visual[1] + visual[3]) / 2
    y = center_y - height / 2
    x = _bounded(current[0], x, page_width * 0.03, 0.0, max(0.0, page_width - width))
    y = _bounded(current[1], y, page_height * 0.02, 0.0, max(0.0, page_height - height))
    return x, y, x + width, y + height


def _refine_font_size(item: dict[str, Any], visual: tuple[float, float, float, float], initial: float, family: str, role: str) -> float:
    line_spacing = _line_spacing(role, max(1, str(item.get("text") or "").count("\n") + 1))
    rendered_width, rendered_height = _measure_text(str(item.get("text") or ""), initial, family, line_spacing)
    target_width = max(1.0, visual[2] - visual[0])
    target_height = max(1.0, visual[3] - visual[1])
    width_scale = target_width / max(1.0, rendered_width)
    height_scale = target_height / max(1.0, rendered_height)
    scale = min(width_scale, height_scale)
    scale = max(0.85, min(1.15, scale))
    refined = initial * (0.62 + 0.38 * scale)
    return max(initial * 0.85, min(initial * 1.15, refined))


def _measure_text(text: str, size: float, family: str, line_spacing: float) -> tuple[float, float]:
    lines = str(text or "").splitlines() or [""]
    path = resolve_font_path(family)
    if path:
        try:
            from PIL import ImageFont

            font = ImageFont.truetype(str(path), max(1, int(round(size))))
            widths = [float(font.getlength(line)) for line in lines]
            boxes = [font.getbbox(line or " ") for line in lines]
            glyph_height = max(float(box[3] - box[1]) for box in boxes)
            return max(widths, default=0.0), max(glyph_height, size * 0.72) + max(0, len(lines) - 1) * size * line_spacing
        except (OSError, AttributeError):
            pass
    widths = []
    for line in lines:
        cjk = sum(1 for char in line if "\u3400" <= char <= "\u9fff")
        widths.append((cjk + (len(line) - cjk) * 0.56) * size)
    return max(widths, default=0.0), max(size * 0.82, len(lines) * size * line_spacing)


def _line_spacing(role: str, line_count: int) -> float:
    if role == "body_text":
        return 1.12 if line_count <= 3 else 1.08
    if role == "annotation":
        return 1.04
    return 1.0


def _row_clusters(items: list[dict[str, Any]], tolerance: float) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for item in sorted(items, key=lambda value: float(value["y"])):
        center = float(item["y"]) + float(item["height"]) / 2
        row = next((candidate for candidate in rows if abs(statistics.median(float(member["y"]) + float(member["height"]) / 2 for member in candidate) - center) <= tolerance), None)
        if row is None:
            rows.append([item])
        else:
            row.append(item)
    return rows


def _nearest_badge(item: dict[str, Any], elements: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    if role not in {"main_title", "section_title", "card_title", "label"}:
        return None
    metadata = item.get("metadata") or {}
    visual = _metadata_box(metadata.get("visualTextBBox") or metadata.get("rawOCRBBox"), _box(item))
    tx = (visual[0] + visual[2]) / 2
    ty = (visual[1] + visual[3]) / 2
    candidates: list[tuple[float, dict[str, Any]]] = []
    for candidate in elements:
        candidate_metadata = candidate.get("metadata") or {}
        if not candidate_metadata.get("wholeBadgeAsset"):
            continue
        cx = float(candidate["x"]) + float(candidate["width"]) / 2
        cy = float(candidate["y"]) + float(candidate["height"]) / 2
        same_row = abs(ty - cy) <= max(float(candidate["height"]), visual[3] - visual[1]) * 0.75
        below = visual[1] >= float(candidate["y"]) + float(candidate["height"]) * 0.72 and abs(tx - cx) <= float(candidate["width"]) * 1.2
        left_neighbor = same_row and cx < tx and visual[0] - (float(candidate["x"]) + float(candidate["width"])) <= max(float(candidate["width"]) * 0.8, 80.0)
        if below or left_neighbor:
            candidates.append((((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5, candidate))
    return min(candidates, key=lambda value: value[0])[1] if candidates else None


def _metadata_box(value: Any, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            box = tuple(float(number) for number in value)
            if box[2] > box[0] and box[3] > box[1]:
                return box
        except (TypeError, ValueError):
            pass
    return fallback


def _box(item: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y = float(item.get("x", 0)), float(item.get("y", 0))
    return x, y, x + float(item.get("width", 1)), y + float(item.get("height", 1))


def _bounded(current: float, target: float, maximum_delta: float, lower: float, upper: float) -> float:
    value = max(current - maximum_delta, min(current + maximum_delta, target))
    return max(lower, min(max(lower, upper), value))


def _suppressed(item: dict[str, Any]) -> bool:
    metadata = item.get("metadata") or {}
    return bool(metadata.get("suppressed") or metadata.get("suppressRender") or metadata.get("ownedBy"))


__all__ = ["TypographyLayoutRefiner"]
