from __future__ import annotations

from app.services.pptx.renderer import font_px_to_pt
from app.services.typography.fit_textbox import fit_textbox
from app.services.typography.font_estimator import estimate_style
from app.services.typography.analyzer import foreground_color
from app.services.refinement.typography_layout import TypographyLayoutRefiner
import numpy as np


def test_typography_uses_role_and_explicit_pixel_conversion():
    title = estimate_style("标题", [0, 0, 220, 42], 1000, 600, "#111111", region_hint="main_title")
    body = estimate_style("正文内容", [0, 0, 220, 26], 1000, 600, "#333333", region_hint="body")
    assert title["fontClass"] == "serif"
    assert body["verticalAlign"] == "top"
    assert font_px_to_pt(10, 0.01) == 7.2


def test_fit_textbox_only_shrinks_within_twenty_percent():
    result = fit_textbox("一段很长的正文内容需要适应文本框", 40, 20, 20)
    assert 16 <= result["fontSize"] <= 20


def test_foreground_color_uses_opaque_strokes():
    crop = np.full((20, 100, 3), 250, dtype=np.uint8)
    crop[4:16, 5:65] = 90
    crop[7:13, 15:55] = 12
    color = foreground_color(crop)
    assert int(color[1:3], 16) < 40


def test_typography_keeps_ocr_left_edge():
    layout = {"slide": {"width": 500, "height": 300}, "elements": [{"id": "text", "type": "text", "text": "正文内容", "x": 84, "y": 95, "width": 130, "height": 30, "role": "body_text", "style": {"fontSize": 24, "fontFamily": "Microsoft YaHei"}, "metadata": {"rawOCRBBox": [100, 100, 220, 130]}}]}
    refined, _ = TypographyLayoutRefiner().refine(layout)
    assert refined["elements"][0]["x"] == 100


def test_label_cannot_drift_far_from_its_ocr_position():
    layout = {"slide": {"width": 500, "height": 300}, "elements": [{"id": "label", "type": "text", "text": "EMIT", "x": 165, "y": 100, "width": 80, "height": 25, "role": "label", "style": {"fontSize": 20}, "metadata": {"rawOCRBBox": [100, 100, 155, 125]}}]}
    refined, _ = TypographyLayoutRefiner().refine(layout)
    assert refined["elements"][0]["x"] == 100
