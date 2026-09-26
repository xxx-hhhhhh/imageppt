from app.services.reconstruction.text_coverage import fit_text_to_ocr_lines, measure_text_coverage, suppress_text_like_assets


def test_coverage_counts_source_ocr_lines_and_suppressed_text() -> None:
    layout = {"elements": [
        {"id": "line-1", "type": "text", "text": "Title", "metadata": {"sourceOcrIds": ["text_001", "text_002"]}},
        {"id": "line-2", "type": "text", "text": "Label", "metadata": {"sourceOcrIds": ["text_003"], "suppressRender": True}},
    ]}
    assert measure_text_coverage(layout, 3) == {
        "detectedTextCount": 3, "editableTextCount": 2,
        "nonEditableTextCount": 1, "editableTextCoverage": 0.6667,
    }


def test_ocr_fit_caps_oversized_title_and_suppresses_false_badge() -> None:
    layout = {"slide": {"width": 300, "height": 100}, "elements": [
        {"id": "title", "type": "text", "text": "Long title", "role": "main_title", "x": 60, "y": 0, "width": 200, "height": 60, "style": {"fontSize": 60}, "metadata": {"rawOCRBBox": [20, 10, 120, 30]}},
        {"id": "false-badge", "type": "image", "x": 40, "y": 0, "width": 50, "height": 40, "metadata": {"wholeBadgeAsset": True}},
    ]}
    fit_text_to_ocr_lines(layout)
    assert layout["elements"][0]["x"] == 20
    assert layout["elements"][0]["y"] == 10
    assert layout["elements"][0]["style"]["fontSize"] < 60
    assert suppress_text_like_assets(layout) == 1
    assert layout["elements"][1]["metadata"]["suppressed"] is True
