from __future__ import annotations

from app.services.refinement import TypographyLayoutRefiner


def _text(
    element_id: str,
    text: str,
    role: str,
    x: float,
    y: float,
    width: float,
    height: float,
    font_size: float,
) -> dict:
    return {
        "id": element_id,
        "type": "text",
        "role": role,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "rotation": 0,
        "zIndex": 20,
        "text": text,
        "lines": [{"text": text, "bbox": [x, y, x + width, y + height]}],
        "style": {"fontSize": font_size, "fontFamily": "Microsoft YaHei", "align": "left"},
        "metadata": {
            "rawOCRBBox": [x, y, x + width, y + height],
            "visualTextBBox": [x, y, x + width, y + height],
            "originalLineCount": 1,
        },
    }


def test_single_line_title_expands_textbox_before_wrapping() -> None:
    title = _text("title", "一、总体思路", "main_title", 100, 30, 220, 52, 46)
    layout = {"slide": {"width": 1000, "height": 600}, "elements": [title]}
    refined, stats = TypographyLayoutRefiner().refine(layout)
    result = refined["elements"][0]
    assert "\n" not in result["text"]
    assert result["metadata"]["preserveOriginalLineCount"] is True
    assert result["width"] >= title["width"]
    assert result["metadata"]["refinedTextboxBBox"] == [result["x"], result["y"], result["x"] + result["width"], result["y"] + result["height"]]
    assert stats["singleLinePreserved"] == 1


def test_font_roles_map_to_distinct_font_strategies() -> None:
    elements = [
        _text("title", "总体思路", "main_title", 80, 30, 260, 54, 48),
        _text("body", "正文内容说明", "body_text", 80, 180, 240, 36, 28),
        _text("label", "融合发展", "label_text", 80, 320, 180, 40, 32),
    ]
    refined, stats = TypographyLayoutRefiner().refine({"slide": {"width": 1000, "height": 600}, "elements": elements})
    styles = {item["style"]["fontRole"]: item["style"] for item in refined["elements"]}
    assert set(styles) == {"main_title", "body_text", "label"}
    assert len({style["fontFamily"] for style in styles.values()}) >= 2
    assert styles["main_title"]["fontPriority"] != styles["body_text"]["fontPriority"]
    assert styles["label"]["fontWeight"] >= 600
    assert stats["fontRoleAssignments"] == 3


def test_font_size_refinement_is_bounded_and_traceable() -> None:
    body = _text("body", "较长的正文内容用于测试", "body_text", 100, 120, 210, 28, 30)
    refined, stats = TypographyLayoutRefiner().refine({"slide": {"width": 1000, "height": 600}, "elements": [body]})
    style = refined["elements"][0]["style"]
    assert style["estimatedFontSize"] == 30
    assert style["refinedFontSize"] != style["estimatedFontSize"]
    assert style["fontSizeAdjustment"] == round(style["refinedFontSize"] - style["estimatedFontSize"], 2)
    assert 30 * 0.85 <= style["refinedFontSize"] <= 30 * 1.15
    assert stats["fontSizeAdjustments"] == 1


def test_page_alignment_polish_is_small_and_improves_row() -> None:
    elements = [
        _text("card_1", "数智", "card_title", 100, 100, 120, 42, 34),
        _text("card_2", "筑基", "card_title", 400, 103, 120, 42, 34),
        _text("card_3", "红芯", "card_title", 700, 106, 120, 42, 34),
    ]
    refined, stats = TypographyLayoutRefiner().refine({"slide": {"width": 1000, "height": 600}, "elements": elements})
    ys = [item["y"] for item in refined["elements"]]
    assert max(ys) - min(ys) < 1
    assert all(abs(item["y"] - original) <= 600 * 0.012 for item, original in zip(refined["elements"], [100, 103, 106]))
    assert stats["pageAlignmentAdjustments"] >= 2


def test_label_text_stays_to_the_right_of_nearby_badge() -> None:
    badge = {
        "id": "badge",
        "type": "image",
        "x": 90,
        "y": 300,
        "width": 100,
        "height": 100,
        "groupId": "local-badge-group",
        "metadata": {"wholeBadgeAsset": True},
    }
    label = _text("label", "助推专业融合发展", "label_text", 205, 330, 260, 42, 27)
    label["groupId"] = "vision-workflow-group"
    refined, _ = TypographyLayoutRefiner().refine({"slide": {"width": 1000, "height": 600}, "elements": [badge, label]})
    result = next(item for item in refined["elements"] if item["id"] == "label")
    assert result["x"] >= badge["x"] + badge["width"]
    assert result["metadata"]["neighborRelation"] == "right-of-badge"


def test_card_title_stays_centered_below_badge() -> None:
    badge = {
        "id": "badge",
        "type": "image",
        "x": 170,
        "y": 160,
        "width": 140,
        "height": 140,
        "groupId": "local-badge-group",
        "metadata": {"wholeBadgeAsset": True},
    }
    title = _text("title", "数智", "card_title", 180, 305, 130, 62, 44)
    title["groupId"] = "vision-card-group"
    refined, _ = TypographyLayoutRefiner().refine({"slide": {"width": 1000, "height": 600}, "elements": [badge, title]})
    result = next(item for item in refined["elements"] if item["id"] == "title")
    badge_center = badge["x"] + badge["width"] / 2
    title_center = result["x"] + result["width"] / 2
    assert abs(badge_center - title_center) < 1
    assert result["metadata"]["neighborRelation"] == "below-badge"
