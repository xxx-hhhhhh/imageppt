from __future__ import annotations

from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

from app.config import OUTPUTS_DIR
from app.services.validation.service import save_validation, validate_pptx
from app.services.typography.fit_textbox import fit_textbox


def _rgb(value: str | None, fallback: str = "#111827") -> RGBColor:
    raw = (value or fallback).lstrip("#")
    if len(raw) != 6:
        raw = fallback.lstrip("#")
    try:
        return RGBColor(int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))
    except ValueError:
        return RGBColor(17, 24, 39)


def _path_from_src(src: str | None) -> Path | None:
    if not src:
        return None
    clean = src.split("?", 1)[0]
    parts = clean.replace("\\", "/").split("/")
    try:
        if "backgrounds" in parts:
            index = parts.index("backgrounds")
            return OUTPUTS_DIR / parts[index + 1] / "backgrounds" / parts[index + 2]
        if "assets" in parts:
            index = parts.index("assets")
            return OUTPUTS_DIR / parts[index + 1] / "assets" / parts[index + 2]
    except IndexError:
        return None
    path = Path(src)
    return path if path.is_absolute() else None


def _set_font_family(font: Any, family: str) -> None:
    font.name = family
    # python-pptx's public API sets the latin face; Chinese text uses the East
    # Asian face in PowerPoint, so set both on the underlying run properties.
    try:
        rpr = font._element.get_or_add_rPr()
        for tag in ("a:latin", "a:ea"):
            node = rpr.find(f"{{http://schemas.openxmlformats.org/drawingml/2006/main}}{tag.split(':')[1]}")
            if node is None:
                from pptx.oxml.xmlchemy import OxmlElement
                node = OxmlElement(tag)
                rpr.append(node)
            node.set("typeface", family)
    except Exception:
        pass


def font_px_to_pt(font_size_px: float, pixels_to_inches: float) -> float:
    """Convert the one stored pixel-space estimate exactly once at export."""
    return max(5.0, float(font_size_px) * float(pixels_to_inches) * 72.0)


class PPTXRenderer:
    def render_project(self, project_id: str, layouts: list[dict[str, Any]]) -> tuple[Path, dict[str, Any]]:
        if not layouts:
            raise ValueError("No analyzed slides to export")
        first = layouts[0]["slide"]
        slide_width = 13.333
        slide_height = slide_width * first["height"] / first["width"]
        presentation = Presentation()
        presentation.slide_width = Inches(slide_width)
        presentation.slide_height = Inches(slide_height)
        blank_layout = presentation.slide_layouts[6]
        for layout in layouts:
            slide = presentation.slides.add_slide(blank_layout)
            px_width = float(layout["slide"]["width"])
            px_height = float(layout["slide"]["height"])
            sx = slide_width / px_width
            sy = slide_height / px_height
            for element in sorted(layout.get("elements", []), key=lambda item: item.get("zIndex", 0)):
                self._add_element(slide, element, sx, sy)
        output_path = OUTPUTS_DIR / project_id / "editable.pptx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        presentation.save(output_path)
        report = validate_pptx(output_path, layouts)
        save_validation(report, OUTPUTS_DIR / project_id / "validation.json")
        return output_path, report

    def _add_element(self, slide: Any, element: dict[str, Any], sx: float, sy: float) -> None:
        kind = element.get("type")
        metadata = element.get("metadata") or {}
        if metadata.get("suppressed") or metadata.get("suppressRender") or metadata.get("ownedBy"):
            return
        strategy = metadata.get("reconstructionStrategy") or _default_strategy(kind)
        if (metadata.get("preserveWholeAsset") or metadata.get("wholeBadgeAsset")) and strategy not in {"cutout_image", "transparent_image"}:
            strategy = "local_image"
        if strategy == "group":
            return
        x = Inches(element.get("x", 0) * sx)
        y = Inches(element.get("y", 0) * sy)
        width = Inches(max(0.01, element.get("width", 1) * sx))
        height = Inches(max(0.01, element.get("height", 1) * sy))
        style = element.get("style") or {}
        if strategy in {"transparent_image", "local_image", "cutout_image", "background_image"}:
            image_path = _path_from_src(element.get("src"))
            if image_path and image_path.exists():
                picture = slide.shapes.add_picture(str(image_path), x, y, width=width, height=height)
                crop = element.get("crop") or {}
                for key in ("left", "right", "top", "bottom"):
                    if key in crop and hasattr(picture, f"crop_{key}"):
                        setattr(picture, f"crop_{key}", float(crop[key]))
                picture.rotation = float(element.get("rotation", 0))
            return
        if strategy == "editable_text":
            shape = slide.shapes.add_textbox(x, y, width, height)
            shape.rotation = float(element.get("rotation", 0))
            frame = shape.text_frame
            frame.clear()
            frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
            original_line_count = int(metadata.get("originalLineCount") or max(1, len(element.get("lines") or [])) or 1)
            preserve_line_count = bool(metadata.get("preserveOriginalLineCount"))
            frame.word_wrap = True
            vertical = style.get("verticalAlign", "top")
            frame.vertical_anchor = {"top": MSO_ANCHOR.TOP, "bottom": MSO_ANCHOR.BOTTOM, "middle": MSO_ANCHOR.MIDDLE}.get(vertical, MSO_ANCHOR.TOP)
            paragraph = frame.paragraphs[0]
            paragraph.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}.get(style.get("align", "left"), PP_ALIGN.LEFT)
            paragraph.line_spacing = float(style.get("lineSpacing", 1.12))
            run = paragraph.add_run()
            lines = element.get("lines") or []
            text = "\n".join(str(line.get("text", "")) for line in lines) if lines else (element.get("text") or "")
            fit = fit_textbox(
                text,
                float(element.get("width", 1)),
                float(element.get("height", 1)),
                float(style.get("fontSize", 24)),
                original_line_count=original_line_count,
                preserve_line_count=preserve_line_count,
            )
            run.text = str(fit["text"])
            font = run.font
            _set_font_family(font, style.get("fontFamily", "Microsoft YaHei"))
            font.size = Pt(font_px_to_pt(float(fit["fontSize"]), sx))
            font.bold = int(style.get("fontWeight", 400)) >= 600
            font.italic = style.get("fontStyle", "normal") == "italic"
            font.color.rgb = _rgb(style.get("color"))
            return
        if strategy != "native_shape":
            return
        if kind in ("line", "arrow"):
            connector_type = MSO_CONNECTOR.STRAIGHT
            shape = slide.shapes.add_connector(connector_type, x, y, x + width, y + height)
            shape.rotation = float(element.get("rotation", 0))
            shape.line.color.rgb = _rgb(style.get("stroke"), "#17365D")
            shape.line.width = Pt(max(0.5, float(style.get("strokeWidth", 1))))
            if kind == "arrow":
                try:
                    shape.line.end_arrowhead = True
                except Exception:
                    pass
            return
        shape_type = {"rectangle": MSO_SHAPE.RECTANGLE, "roundedRectangle": MSO_SHAPE.ROUNDED_RECTANGLE, "ellipse": MSO_SHAPE.OVAL}.get(kind)
        if shape_type is None:
            return
        shape = slide.shapes.add_shape(shape_type, x, y, width, height)
        shape.rotation = float(element.get("rotation", 0))
        shape.fill.solid()
        shape.fill.fore_color.rgb = _rgb(style.get("fill"), "#DCE6F1")
        try:
            shape.fill.transparency = max(0, min(100, int((1 - float(style.get("opacity", 1))) * 100)))
        except Exception:
            pass
        shape.line.color.rgb = _rgb(style.get("stroke"), "#17365D")
        shape.line.width = Pt(max(0.5, float(style.get("strokeWidth", 1))))


def _default_strategy(kind: str | None) -> str:
    if kind == "text":
        return "editable_text"
    if kind == "group":
        return "group"
    if kind == "background":
        return "background_image"
    if kind == "image":
        return "local_image"
    if kind in {"rectangle", "roundedRectangle", "ellipse", "line", "arrow"}:
        return "native_shape"
    return "local_image"
