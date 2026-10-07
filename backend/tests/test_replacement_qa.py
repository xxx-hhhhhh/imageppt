import json

import cv2
import numpy as np
import pytest

from app.services.reconstruction.replacement_qa import check_replacement_regions
from app.services.reconstruction.text_erasure import count_text_ghosting, erase_editable_text_sources
from app.services.reconstruction.white_objectization import objectize_on_white
from app.services.visual_qa.analyzer import render_preview


def test_local_replacement_report_records_ghost_reduction_and_comparison(tmp_path):
    source = np.full((120, 240, 3), 255, np.uint8)
    cv2.rectangle(source, (20, 20), (175, 95), (175, 100, 45), -1)
    before = source.copy()
    after = source.copy()
    cv2.rectangle(after, (65, 40), (140, 70), (175, 100, 45), -1)
    for name, image in (("source", source), ("before", before), ("after", after)):
        cv2.imwrite(str(tmp_path / f"{name}.png"), image)
    layout = {"elements": [
        {"id": "card", "type": "rectangle", "x": 20, "y": 20, "width": 156, "height": 76, "metadata": {"layerRole": "container"}},
        {"id": "label", "type": "text", "x": 65, "y": 40, "width": 75, "height": 30, "text": "Label", "metadata": {}},
    ]}

    report = check_replacement_regions(tmp_path / "source.png", tmp_path / "before.png", tmp_path / "after.png", layout, tmp_path / "qa.json", tmp_path / "compare.png", 1, 0)

    assert report["ghostingBefore"] == 1
    assert report["ghostingAfter"] == 0
    assert report["worsenedRegions"] == []
    assert json.loads((tmp_path / "qa.json").read_text(encoding="utf-8")) == report
    assert cv2.imread(str(tmp_path / "compare.png")).shape == (120, 480, 3)


def test_local_replacement_report_flags_worsened_module(tmp_path):
    source = np.full((90, 150, 3), 255, np.uint8)
    before = source.copy()
    after = np.full_like(source, 120)
    for name, image in (("source", source), ("before", before), ("after", after)):
        cv2.imwrite(str(tmp_path / f"{name}.png"), image)
    layout = {"elements": [{"id": "bad_card", "type": "image", "x": 20, "y": 20, "width": 90, "height": 45, "metadata": {"layerRole": "container"}}]}

    report = check_replacement_regions(tmp_path / "source.png", tmp_path / "before.png", tmp_path / "after.png", layout, tmp_path / "qa.json", tmp_path / "compare.png", 0, 0)

    assert report["worsenedRegions"][0]["elementId"] == "bad_card"


def test_expanded_editor_textbox_does_not_hide_erased_pale_plate(tmp_path):
    source = np.full((100, 160, 3), 255, np.uint8)
    source[20:80, 20:140] = 245
    after = np.full_like(source, 255)
    for name, image in (("source", source), ("before", source), ("after", after)):
        cv2.imwrite(str(tmp_path / f"{name}.png"), image)
    layout = {"elements": [
        {"id": "plate", "type": "image", "x": 20, "y": 20, "width": 120, "height": 60},
        {"id": "text", "type": "text", "x": 20, "y": 20, "width": 120, "height": 60,
         "metadata": {"rawOCRBBox": [55, 40, 105, 60]}},
    ]}
    report = check_replacement_regions(tmp_path / "source.png", tmp_path / "before.png", tmp_path / "after.png", layout, tmp_path / "qa.json", tmp_path / "compare.png", 0, 0)
    assert report["checkedRegions"] == 1
    assert report["safe"] is False
    assert report["worsenedRegions"][0]["lostVisualPixels"] == 5904
    assert report["worsenedRegions"][0]["errorAfter"] == 10


def test_nonoverlapping_text_does_not_mask_module_with_negative_slice(tmp_path):
    source = np.full((100, 160, 3), 255, np.uint8)
    source[50:90, 50:140] = 230
    after = source.copy()
    after[50:70, 50:110] = 255
    for name, image in (("source", source), ("before", source), ("after", after)):
        cv2.imwrite(str(tmp_path / f"{name}.png"), image)
    layout = {"elements": [
        {"id": "plate", "type": "image", "x": 50, "y": 50, "width": 90, "height": 40},
        {"id": "outside", "type": "text", "x": 0, "y": 0, "width": 20, "height": 30},
    ]}
    report = check_replacement_regions(tmp_path / "source.png", tmp_path / "before.png", tmp_path / "after.png", layout, tmp_path / "qa.json", tmp_path / "compare.png", 0, 0)
    assert report["safe"] is False
    assert report["worsenedRegions"][0]["lostVisualPixels"] == 1200


def test_unchanged_existing_pale_gap_does_not_reject_revision(tmp_path):
    source = np.full((80, 120, 3), 249, np.uint8)
    previous = np.full_like(source, 255)
    for name, image in (("source", source), ("before", previous), ("after", previous)):
        cv2.imwrite(str(tmp_path / f"{name}.png"), image)
    layout = {"elements": [{"id": "plate", "type": "image", "x": 0, "y": 0, "width": 120, "height": 80}]}
    report = check_replacement_regions(tmp_path / "source.png", tmp_path / "before.png", tmp_path / "after.png", layout, tmp_path / "qa.json", tmp_path / "compare.png", 0, 0)
    assert report["safe"] is True
    assert report["worsenedRegions"] == []


@pytest.mark.parametrize("erase_decoration", [False, True])
def test_qa_allows_glyph_margin_but_protects_neighboring_decoration(tmp_path, erase_decoration):
    source = np.full((100, 140, 3), 255, np.uint8)
    source[40:80, 40:90] = 245
    source[50:65, 50:70] = 15
    source[50:65, 44:47] = (40, 70, 150)
    after = source.copy()
    after[50:65, 50:70] = 245
    if erase_decoration:
        after[50:65, 44:47] = 255
    for name, pixels in (("source", source), ("before", source), ("after", after)):
        cv2.imwrite(str(tmp_path / f"{name}.png"), pixels)
    layout = {"elements": [
        {"id": "module", "type": "image", "x": 40, "y": 40, "width": 50, "height": 40},
        {"id": "label", "type": "text", "metadata": {"rawOCRBBox": [52, 52, 68, 63]}},
    ]}
    report = check_replacement_regions(tmp_path / "source.png", tmp_path / "before.png", tmp_path / "after.png", layout, tmp_path / "qa.json", tmp_path / "compare.png", 1, 0)
    assert report["safe"] is (not erase_decoration)
    if erase_decoration:
        assert report["worsenedRegions"][0]["lostVisualPixels"] == 45


def test_source_glyph_changes_are_excluded_without_hiding_support(tmp_path):
    source = np.full((100, 160, 3), 255, np.uint8)
    source[20:80, 20:140] = 245
    source[40:60, 55:105] = 20
    after = source.copy()
    after[40:60, 55:105] = 245
    for name, image in (("source", source), ("before", source), ("after", after)):
        cv2.imwrite(str(tmp_path / f"{name}.png"), image)
    layout = {"elements": [
        {"id": "plate", "type": "image", "x": 20, "y": 20, "width": 120, "height": 60},
        {"id": "text", "type": "text", "x": 20, "y": 20, "width": 120, "height": 60,
         "metadata": {"rawOCRBBox": [55, 40, 105, 60]}},
    ]}
    report = check_replacement_regions(tmp_path / "source.png", tmp_path / "before.png", tmp_path / "after.png", layout, tmp_path / "qa.json", tmp_path / "compare.png", 1, 0)
    assert report["safe"] is True
    assert report["improved"] is True


def test_white_replacement_removes_old_glyph_from_image_below_text(tmp_path):
    project = tmp_path / "project"
    (project / "assets").mkdir(parents=True)
    source = np.full((140, 300, 3), 255, np.uint8)
    cv2.rectangle(source, (30, 25), (265, 115), (215, 175, 125), -1)
    cv2.putText(source, "TITLE", (72, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 15, 15), 2)
    source_path = project / "source.png"
    asset_path = project / "assets" / "card.png"
    background_path = project / "backgrounds" / "page_1.png"
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(asset_path), source[25:116, 30:266])
    layout = {"slide": {"width": 300, "height": 140}, "elements": [
        {"id": "card", "type": "image", "x": 30, "y": 25, "width": 236, "height": 91, "zIndex": 10, "src": str(asset_path), "metadata": {"reconstructionStrategy": "cutout_image"}},
        {"id": "title", "type": "text", "x": 68, "y": 55, "width": 115, "height": 34, "zIndex": 20, "text": "TITLE", "style": {"fontSize": 28}, "metadata": {}},
    ]}

    objectize_on_white(source_path, background_path, layout, project / "assets", "project", 1)
    before_path = project / "before.png"
    after_path = project / "after.png"
    render_preview(background_path, layout, before_path)
    before_ghosts = count_text_ghosting(source_path, background_path, layout)
    erase_editable_text_sources(background_path, layout, clean_background=False)
    render_preview(background_path, layout, after_path)
    after_ghosts = count_text_ghosting(source_path, background_path, layout)
    report = check_replacement_regions(source_path, before_path, after_path, layout, project / "qa.json", project / "compare.png", before_ghosts, after_ghosts)

    assert before_ghosts == 1
    assert after_ghosts == 0
    assert report["safe"] is True
    assert np.all(cv2.imread(str(background_path)) == 255)
