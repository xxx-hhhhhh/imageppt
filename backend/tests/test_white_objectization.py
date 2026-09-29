from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pptx import Presentation

from app.services.reconstruction.white_objectization import objectize_on_white
from app.services.reconstruction.objectization_qa import repair_objectized_modules
from app.services.pptx.renderer import PPTXRenderer
from app.services.visual_qa.analyzer import render_preview


def test_white_surface_extracts_unowned_visual_and_keeps_text_editable(tmp_path):
    source = np.full((240, 400, 3), 255, np.uint8)
    cv2.rectangle(source, (30, 40), (105, 115), (30, 70, 180), -1)
    cv2.putText(source, "Title", (180, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    source_path = tmp_path / "source.png"
    background_path = tmp_path / "backgrounds" / "page_1.png"
    cv2.imwrite(str(source_path), source)
    layout = {"slide": {"width": 400, "height": 240}, "elements": [{"id": "title", "type": "text", "x": 175, "y": 45, "width": 100, "height": 40, "text": "Title", "metadata": {"rawOCRBBox": [175, 45, 275, 85]}}]}

    stats = objectize_on_white(source_path, background_path, layout, tmp_path / "assets", "demo", 1)

    assert stats["whiteObjectAssets"] + stats["whiteContainerShapes"] >= 1
    assert np.all(cv2.imread(str(background_path)) == 255)
    assert layout["metadata"]["reconstructionSurfaceMode"] == "white_objectized"
    assert layout["elements"][0]["type"] == "text"
    visuals = [item for item in layout["elements"] if item["type"] != "text"]
    assert any(item["x"] <= 30 and item["x"] + item["width"] >= 105 for item in visuals)
    assert all(not (item["x"] >= 170 and item["x"] < 280) for item in visuals)


def test_text_backplates_and_cards_export_as_independent_shapes(tmp_path):
    source = np.full((300, 500, 3), 255, np.uint8)
    cv2.rectangle(source, (25, 30), (190, 80), (225, 155, 90), -1)
    cv2.putText(source, "LABEL", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 25), 2)
    cv2.rectangle(source, (230, 35), (460, 245), (245, 238, 225), -1)
    cv2.putText(source, "CARD", (265, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2)
    source_path = tmp_path / "source.png"
    background_path = tmp_path / "backgrounds" / "page_1.png"
    cv2.imwrite(str(source_path), source)
    layout = {"slide": {"width": 500, "height": 300}, "elements": [
        {"id": "label_text", "type": "text", "x": 40, "y": 40, "width": 115, "height": 35, "zIndex": 20, "text": "LABEL", "metadata": {"rawOCRBBox": [40, 40, 155, 75]}},
        {"id": "card_text", "type": "text", "x": 260, "y": 75, "width": 105, "height": 42, "zIndex": 20, "text": "CARD", "metadata": {"rawOCRBBox": [260, 75, 365, 117]}},
    ]}

    stats = objectize_on_white(source_path, background_path, layout, tmp_path / "assets", "demo", 1)

    shapes = [item for item in layout["elements"] if item.get("metadata", {}).get("layerRole") == "container"]
    assert stats["whiteContainerShapes"] >= 2
    assert any(item["x"] <= 25 and item["width"] >= 160 for item in shapes)
    assert any(item["x"] <= 230 and item["width"] >= 225 for item in shapes)
    assert all(item["zIndex"] < 20 for item in shapes)
    assert np.all(cv2.imread(str(background_path)) == 255)
    preview = tmp_path / "preview.png"
    render_preview(background_path, layout, preview)
    pixels = cv2.imread(str(preview))
    assert np.max(np.abs(pixels[38, 32].astype(int) - source[38, 32].astype(int))) < 8
    assert np.all(pixels[10, 10] == 255)
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    for item in shapes:
        PPTXRenderer()._add_element(slide, item, 0.02, 0.02)
    assert len(slide.shapes) == len(shapes)
    assert all(shape.shape_type == 1 for shape in slide.shapes)  # native PowerPoint shapes


def test_pale_module_backplate_survives_with_foreground_and_group(tmp_path):
    source = np.full((240, 400, 3), 250, np.uint8)
    cv2.rectangle(source, (48, 42), (226, 112), (242, 244, 246), -1)
    cv2.circle(source, (76, 77), 13, (180, 80, 40), -1)
    cv2.putText(source, "ITEM", (101, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    source_path = tmp_path / "pale.png"
    background_path = tmp_path / "backgrounds" / "page_1.png"
    cv2.imwrite(str(source_path), source)
    layout = {"slide": {"width": 400, "height": 240}, "elements": [
        {"id": "module_icon", "type": "ellipse", "x": 63, "y": 64, "width": 26, "height": 26, "zIndex": 10, "groupId": "module_a", "style": {"fill": "#2850B4"}},
        {"id": "module_title", "type": "text", "x": 99, "y": 62, "width": 91, "height": 29, "zIndex": 20, "groupId": "module_a", "text": "ITEM", "metadata": {"rawOCRBBox": [99, 62, 190, 91]}},
    ]}
    stats = objectize_on_white(source_path, background_path, layout, tmp_path / "assets", "pale-module", 1)
    plates = [item for item in layout["elements"] if item.get("metadata", {}).get("layerRole") == "container"]
    assert stats["whiteContainerShapes"] >= 1
    plate = next(item for item in plates if item["x"] <= 48 and item["x"] + item["width"] >= 226)
    assert plate["groupId"] == "module_a"
    assert set(plate["metadata"]["moduleMemberIds"]) == {"module_icon", "module_title"}
    assert plate["zIndex"] < 10
    assert plate["type"] in {"rectangle", "roundedRectangle", "image"}
    assert np.all(cv2.imread(str(background_path)) == 255)
    preview = tmp_path / "preview.png"
    render_preview(background_path, layout, preview)
    pixels = cv2.imread(str(preview))
    assert np.max(np.abs(pixels[48, 51].astype(int) - source[48, 51].astype(int))) < 8
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    PPTXRenderer()._add_element(slide, plate, 0.02, 0.02)
    assert len(slide.shapes) == 1


def test_page_wide_pale_background_is_not_a_local_backplate(tmp_path):
    source = np.full((240, 400, 3), 245, np.uint8)
    source_path = tmp_path / "page.png"
    background_path = tmp_path / "backgrounds" / "page_1.png"
    cv2.imwrite(str(source_path), source)
    layout = {"slide": {"width": 400, "height": 240}, "elements": [
        {"id": "title", "type": "text", "x": 80, "y": 60, "width": 120, "height": 30, "text": "Title", "metadata": {"rawOCRBBox": [80, 60, 200, 90]}}
    ]}
    objectize_on_white(source_path, background_path, layout, tmp_path / "assets", "page-bg", 1)
    assert not [item for item in layout["elements"] if item.get("metadata", {}).get("layerRole") == "container"]


def test_colored_title_surface_keeps_white_inner_frame_as_movable_detail(tmp_path):
    source = np.full((240, 420, 3), 255, np.uint8)
    cv2.rectangle(source, (42, 48), (352, 142), (165, 86, 38), -1)
    cv2.rectangle(source, (54, 60), (340, 130), (255, 255, 255), 4)
    cv2.putText(source, "TITLE", (130, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    source_path = tmp_path / "source.png"
    background_path = tmp_path / "backgrounds" / "page_1.png"
    cv2.imwrite(str(source_path), source)
    layout = {"slide": {"width": 420, "height": 240}, "elements": [
        {"id": "title", "type": "text", "x": 125, "y": 78, "width": 110, "height": 35, "zIndex": 20, "groupId": "header", "text": "TITLE", "style": {"color": "#FFFFFF", "fontSize": 24}, "metadata": {"rawOCRBBox": [125, 78, 235, 113]}}
    ]}
    stats = objectize_on_white(source_path, background_path, layout, tmp_path / "assets", "inner-frame", 1)
    parents = [item for item in layout["elements"] if (item.get("metadata") or {}).get("layerRole") == "container" and item.get("type") in {"rectangle", "roundedRectangle"}]
    details = [item for item in layout["elements"] if (item.get("metadata") or {}).get("layerRole") == "container_detail"]
    assert parents and details and stats["whiteInternalDetails"] >= 1
    parent = next(item for item in parents if item["x"] <= 42 and item["x"] + item["width"] >= 352)
    detail = next(item for item in details if item["x"] <= 54 and item["x"] + item["width"] >= 340)
    assert parent["groupId"] == detail["groupId"] == "header"
    assert detail["id"] in parent["metadata"]["moduleMemberIds"]
    assert parent["zIndex"] < detail["zIndex"] < layout["elements"][0]["zIndex"]
    assert np.all(cv2.imread(str(background_path)) == 255)
    if detail["type"] == "image":
        asset = cv2.imread(str(tmp_path / "assets" / Path(detail["src"]).name), cv2.IMREAD_UNCHANGED)
        assert asset.shape[2] == 4 and np.count_nonzero(asset[:, :, 3]) > 0
        assert asset[asset.shape[0] // 2, asset.shape[1] // 2, 3] == 0
        detail["src"] = str(tmp_path / "assets" / Path(detail["src"]).name)
    preview = tmp_path / "preview.png"
    render_preview(background_path, layout, preview)
    pixels = cv2.imread(str(preview))
    assert np.max(np.abs(pixels[60, 100].astype(int) - source[60, 100].astype(int))) < 12
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    for item in (parent, detail, layout["elements"][0]):
        PPTXRenderer()._add_element(slide, item, 0.02, 0.02)
    assert len(slide.shapes) == 3
    assert any(shape.has_text_frame and "TITLE" in shape.text for shape in slide.shapes)


def test_outline_card_becomes_movable_container(tmp_path):
    source = np.full((240, 400, 3), 255, np.uint8)
    cv2.rectangle(source, (45, 35), (330, 190), (120, 65, 35), 3)
    cv2.putText(source, "NOTE", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)
    layout = {"slide": {"width": 400, "height": 240}, "elements": [{"id": "note", "type": "text", "x": 78, "y": 72, "width": 105, "height": 38, "zIndex": 20, "text": "NOTE", "metadata": {"rawOCRBBox": [78, 72, 183, 110]}}]}

    stats = objectize_on_white(source_path, tmp_path / "backgrounds" / "page_1.png", layout, tmp_path / "assets", "demo", 1)

    containers = [item for item in layout["elements"] if item.get("metadata", {}).get("layerRole") == "container"]
    assert stats["whiteContainerShapes"] >= 1
    assert any(item["x"] <= 46 and item["width"] >= 280 and item["zIndex"] < 20 for item in containers)


def test_textured_bordered_panel_uses_independent_text_clean_image(tmp_path):
    source = np.full((220, 380, 3), 255, np.uint8)
    for x in range(42, 338):
        source[32:184, x] = (85 + (x % 90), 145, 215)
    cv2.rectangle(source, (40, 30), (339, 185), (28, 52, 100), 3)
    cv2.putText(source, "INFO", (95, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)
    layout = {"slide": {"width": 380, "height": 220}, "elements": [{"id": "info", "type": "text", "x": 90, "y": 76, "width": 95, "height": 40, "zIndex": 20, "text": "INFO", "metadata": {"rawOCRBBox": [90, 76, 185, 116]}}]}

    stats = objectize_on_white(source_path, tmp_path / "backgrounds" / "page_1.png", layout, tmp_path / "assets", "demo", 1)

    assets = [item for item in layout["elements"] if item.get("metadata", {}).get("layerRole") == "container" and item["type"] == "image"]
    assert stats["whiteContainerShapes"] >= 1
    assert assets and assets[0]["zIndex"] < 20
    assert assets[0]["metadata"]["textCleaned"] is True
    assert (tmp_path / "assets" / f"{assets[0]['id']}.png").exists()


def test_residual_icon_inside_flat_card_is_not_claimed_by_card(tmp_path):
    source = np.full((220, 360, 3), 255, np.uint8)
    cv2.rectangle(source, (30, 30), (320, 180), (230, 220, 205), -1)
    cv2.circle(source, (105, 105), 20, (180, 55, 25), -1)
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)
    layout = {"slide": {"width": 360, "height": 220}, "elements": []}

    stats = objectize_on_white(source_path, tmp_path / "backgrounds" / "page_1.png", layout, tmp_path / "assets", "demo", 1)

    assert stats["whiteContainerShapes"] >= 1
    assert stats["whiteObjectAssets"] >= 1
    residual = next(item for item in layout["elements"] if item.get("metadata", {}).get("layerRole") == "residual" and 80 <= item["x"] <= 110)
    container = next(item for item in layout["elements"] if item.get("metadata", {}).get("layerRole") == "container")
    assert residual["zIndex"] > container["zIndex"]
    residual["src"] = str(tmp_path / "assets" / f"{residual['id']}.png")
    preview = tmp_path / "preview.png"
    render_preview(tmp_path / "backgrounds" / "page_1.png", layout, preview)
    assert np.max(np.abs(cv2.imread(str(preview))[105, 105].astype(int) - source[105, 105].astype(int))) < 8


def test_ocr_box_does_not_erase_its_pale_supporting_strip(tmp_path):
    source = np.full((150, 320, 3), 255, np.uint8)
    cv2.rectangle(source, (45, 52), (265, 92), (249, 250, 249), -1)
    cv2.putText(source, "LABEL", (85, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2)
    source_path = tmp_path / "source.png"
    background_path = tmp_path / "backgrounds" / "page.png"
    cv2.imwrite(str(source_path), source)
    layout = {"slide": {"width": 320, "height": 150}, "elements": [
        {"id": "label", "type": "text", "x": 40, "y": 47, "width": 230, "height": 50,
         "text": "LABEL", "zIndex": 20, "metadata": {"rawOCRBBox": [40, 47, 270, 97]}}
    ]}
    stats = objectize_on_white(source_path, background_path, layout, tmp_path / "assets", "pale", 1)
    visuals = [item for item in layout["elements"] if item["type"] != "text"]
    assert visuals and stats["whiteObjectAssets"] + stats["whiteObjectShapes"] > 0
    assert any(item["x"] <= 50 and item["x"] + item["width"] >= 260 for item in visuals)
    assert np.all(cv2.imread(str(background_path)) == 255)
    for item in visuals:
        if item["type"] == "image":
            item["src"] = str(tmp_path / "assets" / Path(item["src"]).name)
    preview = tmp_path / "preview.png"
    render_preview(background_path, layout, preview)
    assert np.max(np.abs(cv2.imread(str(preview))[55, 55].astype(int) - source[55, 55].astype(int))) < 8


def test_unplanned_complex_local_region_is_movable_fallback(tmp_path):
    source = np.full((170, 300, 3), 255, np.uint8)
    for row in range(40, 120):
        source[row, 35:140] = (235 + row % 3, 240, 244)
    cv2.circle(source, (88, 80), 19, (80, 100, 190), -1)
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)
    layout = {"slide": {"width": 300, "height": 170}, "elements": []}
    stats = objectize_on_white(source_path, tmp_path / "backgrounds" / "page.png", layout, tmp_path / "assets", "fallback", 1)
    assert stats["whiteObjectAssets"] + stats["whiteObjectShapes"] >= 2
    assert any(item["x"] <= 88 <= item["x"] + item["width"] for item in layout["elements"])


def test_round_badge_and_pale_support_remain_separate_movable_objects(tmp_path):
    source = np.full((220, 320, 3), 255, np.uint8)
    cv2.rectangle(source, (38, 95), (210, 177), (246, 248, 250), -1)
    cv2.circle(source, (84, 92), 30, (35, 92, 185), -1, cv2.LINE_AA)
    cv2.circle(source, (84, 92), 29, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(source, (84, 92), 9, (230, 220, 55), -1, cv2.LINE_AA)
    source_path = tmp_path / "source.png"
    background = tmp_path / "background.png"
    cv2.imwrite(str(source_path), source)
    layout = {"slide": {"width": 320, "height": 220}, "elements": []}
    objectize_on_white(source_path, background, layout, tmp_path / "assets", "round", 1)
    badges = [item for item in layout["elements"] if item["type"] == "image" and item["x"] <= 84 <= item["x"] + item["width"] and item["y"] <= 70]
    assert badges, [(item["type"], item["x"], item["y"], item["width"], item["height"]) for item in layout["elements"]]
    badge = badges[0]
    image = cv2.imread(str(tmp_path / "assets" / Path(badge["src"]).name), cv2.IMREAD_UNCHANGED)
    assert image.shape[2] == 4
    assert image[0, 0, 3] < 32 and image[image.shape[0] // 2, image.shape[1] // 2, 3] > 220, [(item["type"], item["x"], item["y"], item["width"], item["height"], item.get("metadata", {}).get("layerRole")) for item in layout["elements"]]
    assert any(item is not badge and item["x"] <= 45 and item["x"] + item["width"] >= 200 and item["y"] <= 110 <= item["y"] + item["height"] for item in layout["elements"]), [(item["type"], item["x"], item["y"], item["width"], item["height"], item.get("metadata", {}).get("layerRole")) for item in layout["elements"]]
    assert np.all(cv2.imread(str(background)) == 255)
    report = repair_objectized_modules(source_path, layout, tmp_path / "assets", "round")
    assert report["roundCutoutsChecked"] >= 1 and report["roundCutoutIssues"] == 0
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    for item in layout["elements"]:
        if item["type"] == "image":
            item["src"] = str(tmp_path / "assets" / Path(item["src"]).name)
        PPTXRenderer()._add_element(slide, item, 0.02, 0.02)
    assert len(slide.shapes) == len(layout["elements"])
    preview = tmp_path / "preview.png"
    render_preview(background, layout, preview)
    pixels = cv2.imread(str(preview))
    assert np.max(np.abs(pixels[92, 84].astype(int) - source[92, 84].astype(int))) < 12
    assert np.max(np.abs(pixels[130, 170].astype(int) - source[130, 170].astype(int))) < 12


def test_large_map_silk_and_pale_card_survive_white_objectization(tmp_path):
    source = np.full((400, 700, 3), 255, np.uint8)
    map_contour = np.array([[45, 55], [300, 18], [650, 48], [675, 170], [620, 270], [490, 300], [270, 320], [65, 275]], np.int32)
    cv2.fillPoly(source, [map_contour], (105, 145, 175))
    cv2.rectangle(source, (450, 75), (590, 145), (245, 248, 250), -1)
    silk = np.array([[80, 350], [270, 340], [460, 358], [625, 335], [590, 383], [310, 375]], np.int32)
    cv2.fillPoly(source, [silk], (35, 50, 195))
    path = tmp_path / "source.png"
    background = tmp_path / "background.png"
    cv2.imwrite(str(path), source)
    layout = {"slide": {"width": 700, "height": 400}, "elements": []}
    stats = objectize_on_white(path, background, layout, tmp_path / "assets", "complex", 1)
    assert stats["whiteObjectAssets"] >= 2
    assert any(item["type"] == "image" and item["width"] > 500 for item in layout["elements"])
    assert any(item["y"] >= 330 and item["type"] == "image" for item in layout["elements"])
    assert np.all(cv2.imread(str(background)) == 255)
    for item in layout["elements"]:
        if item["type"] == "image":
            item["src"] = str(tmp_path / "assets" / Path(item["src"]).name)
    preview = tmp_path / "preview.png"
    render_preview(background, layout, preview)
    pixels = cv2.imread(str(preview))
    for y, x in ((150, 170), (370, 420), (100, 500)):
        assert np.max(np.abs(pixels[y, x].astype(int) - source[y, x].astype(int))) < 15
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    for item in layout["elements"]:
        PPTXRenderer()._add_element(slide, item, 0.01, 0.01)
    assert len(slide.shapes) == len(layout["elements"])
    assert sum(shape.shape_type == 13 for shape in slide.shapes) >= 2
