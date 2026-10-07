from __future__ import annotations

import copy
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.services.layout.detectors import detect_text_elements
from app.services.scene.ownership import OwnershipLedger, canonicalize, resolve_asset


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unreadable source image")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", image)[1].tofile(path)


def page_color(image: np.ndarray) -> np.ndarray:
    """Dominant border colour; pale module colour is not classified as white."""
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    colors, counts = np.unique((border // 4).astype(np.uint8), axis=0, return_counts=True)
    dominant = colors[np.argmax(counts)]
    selected = border[np.all(border // 4 == dominant, axis=1)]
    return np.median(selected, axis=0).astype(np.uint8)


def important_mask(image: np.ndarray) -> np.ndarray:
    distance = np.max(np.abs(image.astype(np.int16) - page_color(image).astype(np.int16)), axis=2)
    # Four intensity levels retains even very pale cards. No saturation gate.
    return (distance >= 4).astype(np.uint8) * 255


def bounds(item: dict, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x, y = float(item.get("x", 0)), float(item.get("y", 0))
    return max(0, int(np.floor(x))), max(0, int(np.floor(y))), min(w, int(np.ceil(x + item.get("width", 0)))), min(h, int(np.ceil(y + item.get("height", 0))))


def tight_text_mask(image: np.ndarray, item: dict) -> np.ndarray:
    """Ink evidence AND OCR rectangle. Never a filled bbox/card mask."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = bounds(item, image.shape[:2])
    if x2 <= x1 or y2 <= y1:
        return mask
    crop = image[y1:y2, x1:x2]
    buckets, counts = np.unique(crop.reshape(-1, 3) // 8, axis=0, return_counts=True)
    dominant = buckets[np.argmax(counts)]
    pixels = crop.reshape(-1, 3)
    background = np.median(pixels[np.all(pixels // 8 == dominant, axis=1)], axis=0)
    distance = np.max(np.abs(crop.astype(np.float32) - background), axis=2)
    # OCR text colour can be a blend or poor estimate. It must never restrict
    # mask coverage to only dark centres and leave antialiased ghost lettering.
    ink = distance > 10
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    selected = np.zeros(crop.shape[:2], dtype=np.uint8)
    for index in range(1, count):
        _x, _y, width, height, area = stats[index]
        # Exclude continuous borders, solid strips and large illustrations.
        if area < 2 or area > crop.shape[0] * crop.shape[1] * 0.35:
            continue
        if width >= crop.shape[1] * 0.85 and height < max(3, crop.shape[0] * 0.15):
            continue
        selected[labels == index] = 255
    selected = cv2.dilate(selected, np.ones((3, 3), np.uint8), iterations=1)
    mask[y1:y2, x1:x2] = selected
    return mask


def contour_proposals(image: np.ndarray, excluded: np.ndarray) -> list[np.ndarray]:
    """Multi-colour components find nested pale plates ignored by edge detectors."""
    work = image.copy()
    quantized = (work // 8).astype(np.uint8)
    # Enumerate frequent colours without producing thousands of tiny PPT shapes.
    packed = quantized[:, :, 0].astype(np.int32) * 1024 + quantized[:, :, 1].astype(np.int32) * 32 + quantized[:, :, 2]
    colors, counts = np.unique(packed, return_counts=True)
    candidates: list[np.ndarray] = []
    area_limit = max(100, image.shape[0] * image.shape[1] * 0.0007)
    for color in colors[np.argsort(counts)[-32:]]:
        binary = ((packed == color) & (excluded == 0)).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < area_limit or area > binary.size * 0.85:
                continue
            _x, _y, w, h = cv2.boundingRect(contour)
            if min(w, h) < 4:
                continue
            mask = np.zeros(binary.shape, np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            candidates.append(mask)
    return candidates


def native_geometry(image: np.ndarray, mask: np.ndarray, text_mask: np.ndarray) -> tuple[str, dict] | None:
    """Require a solid component AND faithful geometry, not just a rectangular bbox."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return None
    contour = contours[0]
    x, y, w, h = cv2.boundingRect(contour)
    selected = (mask > 0) & (text_mask == 0)
    pixels = image[selected]
    if len(pixels) < 80:
        return None
    median = np.median(pixels, axis=0)
    if np.mean(np.max(np.abs(pixels.astype(np.float32) - median), axis=1) <= 10) < 0.985:
        return None
    if np.max(np.abs(median - page_color(image))) < 4:
        return None
    approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.012, True)
    expected = np.zeros(mask.shape, np.uint8)
    if len(approx) == 4 and cv2.contourArea(contour) / (w * h) > 0.97:
        kind = "rectangle"
        cv2.rectangle(expected, (x, y), (x+w-1, y+h-1), 255, -1)
    elif len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(expected, ellipse, 255, -1)
        kind = "ellipse"
    else:
        return None
    iou = np.count_nonzero((expected > 0) & (mask > 0)) / max(1, np.count_nonzero((expected > 0) | (mask > 0)))
    if iou < 0.975:
        # Rounded rectangles: preserve as PNG until radius is faithfully known.
        return None
    fill = "#" + "".join(f"{int(channel):02X}" for channel in median[::-1])
    return kind, {"fill": fill, "stroke": fill, "strokeWidth": 0, "opacity": 1}


def consolidate_segment_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Keep a round badge's internal artwork inside its silhouette, not as layers.

    Only compact, strongly elliptical parents absorb contained proposals.
    Large module/card parents must not swallow independent child objects.
    """
    kept: list[np.ndarray] = []
    badge_parents: list[np.ndarray] = []
    for mask in sorted(masks, key=np.count_nonzero, reverse=True):
        area = np.count_nonzero(mask)
        if any(np.count_nonzero((mask > 0) & (parent > 0)) / max(1, area) > 0.97 for parent in badge_parents):
            continue
        if any(np.count_nonzero((mask > 0) & (prior > 0)) / max(1, np.count_nonzero((mask > 0) | (prior > 0))) > 0.9 for prior in kept):
            continue
        kept.append(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1 or len(contours[0]) < 5 or area > mask.size * 0.12:
            continue
        ellipse = cv2.fitEllipse(contours[0])
        expected = np.zeros(mask.shape, np.uint8)
        cv2.ellipse(expected, ellipse, 255, -1)
        iou = np.count_nonzero((mask > 0) & (expected > 0)) / max(1, np.count_nonzero((mask > 0) | (expected > 0)))
        if iou > 0.92:
            badge_parents.append(mask)
    return kept


class ObjectFirstBuilder:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def build(self, image_path: Path, layout: dict, scene: dict, regions: list, segmentation: list,
              project_id: str, page: int, background_path: Path, inpainting) -> dict:
        image = read_image(image_path)
        height, width = image.shape[:2]
        asset_dir = self.output_root / project_id / "assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"p{page}_{uuid.uuid4().hex[:12]}"
        ledger = OwnershipLedger((height, width), self.output_root)
        elements: list[dict] = []
        text_mask = np.zeros((height, width), np.uint8)
        semantics = {item["id"]: item for item in scene.get("elements", [])}
        # Never allow VLM/planner to rewrite OCR text or its source geometry.
        texts = detect_text_elements(regions)
        for item in texts:
            semantic = semantics.get(item["id"]) or {}
            metadata = semantic.get("metadata") or {}
            if item.get("confidence", 0) < 0.5 or metadata.get("visionSemanticType") == "logo":
                continue  # untouched pixels will get movable_image ownership
            item["role"] = semantic.get("role") or item.get("role") or "body_text"
            item["groupId"] = semantic.get("groupId")
            item["owner"] = "editable_text"
            item["source"] = "ocr"
            item.setdefault("metadata", {}).update({"rawOCRBBox": [item["x"], item["y"], item["x"]+item["width"], item["y"]+item["height"]], "reconstructionStrategy": "editable_text", "originalLineCount": 1, "preserveOriginalLineCount": True})
            mask = tight_text_mask(image, item)
            if not np.any(mask):
                continue  # no reliable ink evidence, preserve as a visual
            elements.append(item)
            ledger.claim(item, mask)
            text_mask |= mask
        protected = cv2.bitwise_not(text_mask)
        # Only confirmed ink gets patched; provider output outside the mask is discarded.
        clean_path = asset_dir / f"{prefix}_text_clean.png"
        if hasattr(inpainting, "restore_owned_text"):
            inpainting.restore_owned_text(image_path, text_mask, protected, ledger, clean_path)
            clean = read_image(clean_path)
        else:
            clean = image.copy()
        ghosting_count = 0
        for item in elements:
            x1, y1, x2, y2 = bounds(item, image.shape[:2])
            original_crop = image[y1:y2, x1:x2]
            patched_crop = clean[y1:y2, x1:x2]
            background_hint = np.median(original_crop.reshape(-1, 3), axis=0)
            ink = (np.max(np.abs(original_crop.astype(np.float32)-background_hint), axis=2) > 25) & (ledger.claims[item["id"]][y1:y2, x1:x2] > 0)
            same_ink = np.max(np.abs(patched_crop.astype(np.int16)-original_crop.astype(np.int16)), axis=2) < 12
            ghosting_count += int(ink.any() and np.mean(same_ink[ink]) > 0.15)
        masks = contour_proposals(clean, np.zeros_like(text_mask))
        # The segmentation alpha is a proposal, never deletion permission by itself.
        segment_masks = []
        for segment in segmentation:
            path = resolve_asset(segment.get("alphaCrop"), self.output_root)
            if not path or not path.is_file():
                continue
            box = segment.get("bbox") or {}
            x, y = int(box.get("left", 0)), int(box.get("top", 0))
            with Image.open(path) as asset:
                alpha = np.asarray(asset.convert("RGBA"))[:, :, 3]
            if x < 0 or y < 0 or x + alpha.shape[1] > width or y + alpha.shape[0] > height:
                continue
            mask = np.zeros((height, width), np.uint8)
            mask[y:y+alpha.shape[0], x:x+alpha.shape[1]] = alpha
            segment_masks.append(mask)
        shape_mask = np.zeros_like(text_mask)
        for mask in sorted(masks, key=np.count_nonzero, reverse=True):
            if np.count_nonzero((mask > 0) & (shape_mask > 0)) > np.count_nonzero(mask) * 0.9:
                continue
            geometry = native_geometry(clean, mask, text_mask)
            if geometry is None:
                continue
            kind, style = geometry
            x, y, w, h = cv2.boundingRect(mask)
            item = {"id": f"{prefix}_shape_{len(elements)}", "type": kind, "role": "module_plate", "x": x, "y": y, "width": w, "height": h, "rotation": 0, "zIndex": 1+len(elements), "style": style, "owner": "native_shape", "source": "cv-verified", "confidence": 0.98, "metadata": {"geometryVerified": True, "reconstructionStrategy": "native_shape"}}
            elements.append(item)
            ledger.claim(item, mask)
            shape_mask |= mask
        visual_mask = important_mask(image)
        # Unknown, thin and low-confidence regions remain data, never discarded noise.
        unowned = cv2.bitwise_and(visual_mask, cv2.bitwise_not(ledger.covered))
        visual_owned = np.zeros_like(text_mask)

        def add_asset(mask: np.ndarray, source: str, role: str = "complex_visual") -> None:
            nonlocal visual_owned
            if not np.any(mask):
                return
            x, y, w, h = cv2.boundingRect(mask)
            crop = cv2.cvtColor(clean[y:y+h, x:x+w], cv2.COLOR_BGR2BGRA)
            crop[:, :, 3] = mask[y:y+h, x:x+w]
            asset_id = f"{prefix}_visual_{len(elements)}"
            path = asset_dir / f"{asset_id}.png"
            write_image(path, crop)
            contours, _ = cv2.findContours(mask[y:y+h, x:x+w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            item = {"id": asset_id, "type": "image", "role": role, "x": x, "y": y, "width": w, "height": h, "rotation": 0, "zIndex": 100+len(elements), "src": f"/media/assets/{project_id}/{path.name}", "style": {"opacity": 1}, "owner": "movable_image", "source": source, "confidence": 0.75 if source == "segmentation" else 0.5, "contour": [cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).tolist() for c in contours], "metadata": {"transparent": True, "reconstructionStrategy": "transparent_image", "fallbackReason": "visual fidelity before vectorization"}}
            elements.append(item)
            ledger.claim(item, mask)
            visual_owned |= mask

        for mask in sorted(consolidate_segment_masks(segment_masks), key=np.count_nonzero):
            actual = cv2.bitwise_and(mask, cv2.bitwise_not(shape_mask | visual_owned))
            if np.count_nonzero((actual > 0) & (unowned > 0)) < 64:
                continue
            # An external contour gives badges a real silhouette and retains white artwork inside.
            add_asset(actual, "segmentation")
            unowned &= cv2.bitwise_not(actual)
        # Close nearby fragments for selectable objects, but retain original alpha support.
        grouped = cv2.morphologyEx(unowned, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) < 64:
                continue  # retained in the exact spill asset, not hundreds of tiny objects
            mask = np.zeros_like(text_mask)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mask &= cv2.bitwise_not(shape_mask | visual_owned)
            add_asset(mask, "residual-contour")
        # Morphology can miss tiny features. A transparent spill asset guarantees no loss.
        remainder = visual_mask & cv2.bitwise_not(ledger.covered)
        add_asset(remainder, "residual-exact", "retained_visual")
        # Only now is background generated, and only leased pixels may change.
        removable = ledger.authorize(visual_mask | text_mask | shape_mask | visual_owned)
        background = image.copy()
        background[removable > 0] = page_color(image)
        write_image(background_path, background)
        background_url = f"/media/backgrounds/{project_id}/{background_path.name}"
        elements.insert(0, {"id": "background_001", "type": "background", "role": "page_background", "x": 0, "y": 0, "width": width, "height": height, "rotation": 0, "zIndex": 0, "src": background_url, "owner": "background", "source": "background-last", "style": {"opacity": 1}})
        for item in elements:
            if item["type"] == "text":
                item["zIndex"] = 10000 + elements.index(item)
            elif item["type"] == "image":
                # Do not leave whole-page fallback welded to an unselectable background.
                item["editable"] = True
            claim = ledger.claims.get(item["id"])
            if claim is not None:
                mask_path = asset_dir / f"{prefix}_{item['id']}_mask.png"
                write_image(mask_path, claim)
                item["mask"] = f"/media/assets/{project_id}/{mask_path.name}"
                item.setdefault("metadata", {})["sourcePixelCount"] = int(np.count_nonzero(claim))
            item.setdefault("metadata", {})["sourceBBox"] = {"left": item["x"], "top": item["y"], "width": item["width"], "height": item["height"]}
        # Parent modules come from existing semantic groups; no separate renderer graph.
        for item in elements:
            if item.get("groupId"):
                continue
            cx, cy = item["x"] + item["width"] / 2, item["y"] + item["height"] / 2
            parents = [p for p in elements if p.get("role") == "module_plate" and p["id"] != item["id"] and p["x"] <= cx <= p["x"]+p["width"] and p["y"] <= cy <= p["y"]+p["height"] and p["width"]*p["height"] > item["width"]*item["height"]]
            if parents:
                item["groupId"] = min(parents, key=lambda p: p["width"]*p["height"])["id"]
        result = {"version": "1.2", "sceneVersion": "3.0", "coordinateSystem": "source-pixels-left-top", "slide": {"width": width, "height": height}, "source": image_path.name, "backgroundUrl": background_url, "elements": elements, "metadata": copy.deepcopy(layout.get("metadata") or {})}
        result["metadata"].update({"pipeline": "object-first", "ownershipPolicy": "no-reliable-owner-no-deletion", "ocrRegionCount": len(regions), "editableOCRCount": sum(i["type"] == "text" for i in elements), "removedPixelCount": int(np.count_nonzero(removable)), "unownedVisualPixels": int(np.count_nonzero(visual_mask & cv2.bitwise_not(ledger.covered))), "ghostingCount": ghosting_count})
        return canonicalize(result)
