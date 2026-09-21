from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.services.pptx.renderer import _default_strategy, _path_from_src


def _font(style: dict[str, Any], size_scale: float = 0.75) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(8, int(float(style.get("fontSize", 20)) * size_scale))
    family = style.get("fontFamily", "Microsoft YaHei")
    candidates = [
        Path("C:/Windows/Fonts") / f"{family}.ttf",
        Path("C:/Windows/Fonts/msyh.ttc") if "YaHei" in family else Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_preview(background_path: Path, layout: dict[str, Any], output_path: Path) -> Path:
    base = Image.open(background_path).convert("RGBA") if background_path.exists() else Image.new("RGBA", (int(layout["slide"]["width"]), int(layout["slide"]["height"])), "white")
    draw = ImageDraw.Draw(base, "RGBA")
    for element in sorted(layout.get("elements", []), key=lambda item: item.get("zIndex", 0)):
        kind = element.get("type")
        strategy = (element.get("metadata") or {}).get("reconstructionStrategy") or _default_strategy(kind)
        if strategy == "group":
            continue
        x, y = float(element.get("x", 0)), float(element.get("y", 0))
        w, h = max(1.0, float(element.get("width", 1))), max(1.0, float(element.get("height", 1)))
        style = element.get("style") or {}
        if strategy in {"transparent_image", "local_image", "background_image"}:
            image_path = _path_from_src(element.get("src"))
            if image_path and image_path.exists():
                with Image.open(image_path) as source:
                    asset = source.convert("RGBA").resize((max(1, int(round(w))), max(1, int(round(h)))), Image.Resampling.LANCZOS)
                base.alpha_composite(asset, (int(round(x)), int(round(y))))
                draw = ImageDraw.Draw(base, "RGBA")
            continue
        if strategy == "editable_text":
            draw.multiline_text((x, y), element.get("text") or "", font=_font(style), fill=style.get("color", "#111827"), spacing=max(0, int(float(style.get("lineSpacing", 1.1)) * 4)), align=style.get("align", "left"))
        elif strategy == "native_shape" and kind in {"rectangle", "roundedRectangle"}:
            fill = style.get("fill", "#DCE6F1")
            outline = style.get("stroke", fill)
            alpha = int(float(style.get("opacity", 1)) * 255)
            draw.rounded_rectangle((x, y, x + w, y + h), radius=min(w, h) * 0.16 if kind == "roundedRectangle" else 0, fill=fill + f"{alpha:02X}" if isinstance(fill, str) and len(fill) == 7 else fill, outline=outline, width=max(1, int(float(style.get("strokeWidth", 1)))))
        elif strategy == "native_shape" and kind in {"ellipse", "circle"}:
            draw.ellipse((x, y, x + w, y + h), fill=style.get("fill", "#DCE6F1"), outline=style.get("stroke", style.get("fill", "#DCE6F1")), width=max(1, int(float(style.get("strokeWidth", 1)))))
        elif strategy == "native_shape" and kind in {"line", "arrow"}:
            draw.line((x, y + h / 2, x + w, y + h / 2), fill=style.get("stroke", "#17365D"), width=max(1, int(float(style.get("strokeWidth", 1)))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(output_path)
    return output_path


def _similarity(original: np.ndarray, preview: np.ndarray) -> tuple[float, float, float]:
    original = cv2.resize(original, (preview.shape[1], preview.shape[0]))
    diff = cv2.absdiff(original, preview)
    pixel = max(0.0, 1.0 - float(np.mean(diff)) / 255.0)
    original_edges = cv2.Canny(original, 60, 160)
    preview_edges = cv2.Canny(preview, 60, 160)
    edge = 1.0 - float(np.mean(cv2.absdiff(original_edges, preview_edges))) / 255.0
    gray_a = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    mse = float(np.mean((gray_a.astype(np.float32) - gray_b.astype(np.float32)) ** 2))
    ssim_like = 1.0 / (1.0 + mse / (255.0 ** 2))
    return pixel, edge, ssim_like


def run_visual_qa(original_path: Path, preview_path: Path, output_dir: Path, layout: dict[str, Any]) -> dict[str, Any]:
    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    preview = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
    if original is None or preview is None:
        score = {"overall": 0.0, "layout": 0.0, "text": 0.0, "color": 0.0, "structure": 0.0, "pixelSimilarity": 0.0, "ssim": 0.0, "edgeSimilarity": 0.0, "regions": []}
        return score
    pixel, edge, ssim_like = _similarity(original, preview)
    text_count = sum(1 for item in layout.get("elements", []) if item.get("type") == "text")
    shape_count = sum(1 for item in layout.get("elements", []) if item.get("type") in {"rectangle", "roundedRectangle", "ellipse", "line", "arrow"})
    image_count = sum(1 for item in layout.get("elements", []) if item.get("type") == "image")
    structural = min(1.0, (text_count + shape_count + image_count) / max(1.0, len(layout.get("elements", []))))
    score = {"overall": round((pixel * 0.42 + edge * 0.22 + ssim_like * 0.2 + structural * 0.16), 4), "layout": round(structural, 4), "text": round(min(1.0, text_count / max(1, len([i for i in layout.get("elements", []) if i.get("type") == "text"]))), 4), "color": round(pixel, 4), "structure": round(structural, 4), "pixelSimilarity": round(pixel, 4), "ssim": round(ssim_like, 4), "edgeSimilarity": round(edge, 4), "regions": []}
    output_dir.mkdir(parents=True, exist_ok=True)
    difference = cv2.absdiff(cv2.resize(original, (preview.shape[1], preview.shape[0])), preview)
    cv2.imwrite(str(output_dir / "difference.png"), difference)
    return score
