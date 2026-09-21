from __future__ import annotations

from app.services.pptx.renderer import font_px_to_pt
from app.services.typography.fit_textbox import fit_textbox
from app.services.typography.font_estimator import estimate_style


def test_typography_uses_role_and_explicit_pixel_conversion():
    title = estimate_style("标题", [0, 0, 220, 42], 1000, 600, "#111111", region_hint="main_title")
    body = estimate_style("正文内容", [0, 0, 220, 26], 1000, 600, "#333333", region_hint="body")
    assert title["fontClass"] == "serif"
    assert body["verticalAlign"] == "top"
    assert font_px_to_pt(10, 0.01) == 7.2


def test_fit_textbox_only_shrinks_within_twenty_percent():
    result = fit_textbox("一段很长的正文内容需要适应文本框", 40, 20, 20)
    assert 16 <= result["fontSize"] <= 20
