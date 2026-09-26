from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

from app.services.settings.runtime_settings import load_vision_settings


_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


def repair_complex_text(source_path: Path, background_path: Path, strategies: list[dict], *, max_regions: int = 4) -> int:
    """Use image editing on text over textured backgrounds, keeping the edit inside OCR bounds."""
    settings = load_vision_settings()
    provider = settings.providers["qwen"]
    if not settings.enabled or not provider.enabled or not provider.api_key:
        return 0
    # Image Edit and the configured vision key must belong to the same region.
    if "dashscope.aliyuncs.com" not in provider.base_url.lower():
        return 0
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if source is None or background is None:
        return 0
    repaired = 0
    for strategy in strategies:
        if repaired >= max_regions:
            break
        if not strategy.get("willReconstruct") or strategy.get("category") not in {"complex", "texture"}:
            continue
        box = strategy.get("cleanBBox") or strategy.get("bbox")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = (int(value) for value in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(source.shape[1], x2), min(source.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        try:
            candidate = _edit_crop(source, (x1, y1, x2, y2), provider.api_key, settings.timeout)
        except (requests.RequestException, KeyError, ValueError, OSError):
            strategy["aiRepair"] = "failed"
            continue
        original_crop = source[y1:y2, x1:x2]
        candidate_crop = candidate[y1:y2, x1:x2]
        if not _text_reduced(original_crop, candidate_crop):
            strategy["aiRepair"] = "rejected"
            continue
        background[y1:y2, x1:x2] = candidate_crop
        strategy["aiRepair"] = "accepted"
        strategy["reconstructionStrategy"] = "ai_image_edit"
        repaired += 1
    if repaired:
        cv2.imwrite(str(background_path), background)
    return repaired


def _edit_crop(source: np.ndarray, box: tuple[int, int, int, int], key: str, timeout: int) -> np.ndarray:
    height, width = source.shape[:2]
    x1, y1, x2, y2 = box
    side = max(512, x2 - x1 + 160, y2 - y1 + 160)
    side = min(2048, side)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    left, top = cx - side // 2, cy - side // 2
    patch = np.full((side, side, 3), 255, dtype=np.uint8)
    sx1, sy1 = max(0, left), max(0, top)
    sx2, sy2 = min(width, left + side), min(height, top + side)
    patch[sy1 - top:sy2 - top, sx1 - left:sx2 - left] = source[sy1:sy2, sx1:sx2]
    ok, encoded = cv2.imencode(".png", patch)
    if not ok:
        raise ValueError("Unable to encode image edit crop")
    image_data = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    prompt = (
        "Remove the printed text in the center of this image and reconstruct the background behind it. "
        "Keep nearby diagrams, icons, borders, colors and layout in place. Do not add new text or symbols."
    )
    response = requests.post(
        _ENDPOINT,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": "qwen-image-edit-plus", "input": {"messages": [{"role": "user", "content": [{"image": image_data}, {"text": prompt}]}]}, "parameters": {"n": 1, "watermark": False, "prompt_extend": False, "size": f"{side}*{side}"}},
        timeout=max(30, min(300, timeout)),
    )
    response.raise_for_status()
    url = response.json()["output"]["choices"][0]["message"]["content"][0]["image"]
    edited_response = requests.get(url, timeout=120)
    edited_response.raise_for_status()
    with Image.open(io.BytesIO(edited_response.content)) as image:
        edited = np.asarray(image.convert("RGB").resize((side, side)))[:, :, ::-1].copy()
    result = source.copy()
    result[sy1:sy2, sx1:sx2] = edited[sy1 - top:sy2 - top, sx1 - left:sx2 - left]
    return result


def _text_reduced(original: np.ndarray, edited: np.ndarray) -> bool:
    if original.size == 0 or edited.shape != original.shape:
        return False
    before = cv2.Canny(original, 50, 150)
    after = cv2.Canny(edited, 50, 150)
    before_edges = int(np.count_nonzero(before))
    return before_edges > 5 and np.count_nonzero(after) <= before_edges * 0.85
