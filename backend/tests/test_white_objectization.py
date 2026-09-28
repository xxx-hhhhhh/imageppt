from __future__ import annotations

import cv2
import numpy as np
from pptx import Presentation

from app.services.reconstruction.white_objectization import objectize_on_white
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
