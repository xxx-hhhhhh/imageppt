"""Source-pixel evidence gate shared by objectization and revision validation.

Layout JSON remains the canonical scene. A bbox or a declared owner is not
proof: image alpha/color, shape geometry/color, or tight OCR ink must match.
Uncertain pixels are retained as independent transparent assets before a
background can be cleared. No learned segmentation result authorizes deletion.
"""
from __future__ import annotations
from app.utils import image_io

import math
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageDraw


def tight_text_mask(image: np.ndarray, region: np.ndarray, color: str | None = None) -> np.ndarray:
    """Select small contrasting glyph components, never the entire OCR box."""
    result = np.zeros(region.shape, np.uint8)
    if not np.any(region):
        return result
    x, y, w, h = cv2.boundingRect(region)
    pixels = image[y:y+h, x:x+w, :3].astype(np.int16)
    base = np.median(pixels.reshape(-1, 3), axis=0)
    contrast = np.max(np.abs(pixels - base), axis=2) >= 24
    known_ink = False
    try:
        ink = np.array(list(bytes.fromhex(str(color).lstrip("#")))[::-1], np.int16)
        if ink.shape == (3,) and np.max(np.abs(base - ink)) >= 24:
            contrast &= np.max(np.abs(pixels - ink), axis=2) <= 75
            known_ink = True
    except (ValueError, TypeError):
        pass
    candidate = np.uint8(contrast & (region[y:y+h, x:x+w] > 0))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    local = result[y:y+h, x:x+w]
    for index in range(1, count):
        _, _, cw, ch, area = stats[index]
        if 1 <= area <= w*h*(.55 if known_ink else .25) and cw*ch <= w*h*(.95 if known_ink else .40) and cw <= max(h*2, 8):
            local[labels == index] = 255
    # Dilation only selects nearby contrasting antialias pixels, not a plate.
    expanded = cv2.dilate(result, np.ones((3, 3), np.uint8))
    delta = np.zeros(region.shape, bool)
    delta[y:y+h, x:x+w] = np.max(np.abs(pixels - base), axis=2) >= 8
    return np.uint8((expanded > 0) & delta & (region > 0)) * 255


def _box(item: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    try:
        x, y, w, h, rotation = (float(item.get(key, 0)) for key in ("x", "y", "width", "height", "rotation"))
        if not all(math.isfinite(v) for v in (x, y, w, h, rotation)) or w <= 0 or h <= 0 or abs(rotation) > .01:
            return None  # Rotated objects cannot claim an axis-aligned bbox.
        if x < 0 or y < 0 or x+w > width+.5 or y+h > height+.5:
            return None
        edge = 1 if item.get("type") in {"rectangle", "roundedRectangle", "ellipse", "line", "arrow"} else 0
        return round(x), round(y), min(width, round(x+w)+edge), min(height, round(y+h)+edge)
    except (ValueError, TypeError):
        return None


def _visible(item: dict) -> bool:
    return not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))


def page_surface_evidence(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recognize only a border-anchored flat/one-axis smooth page field.

    No brightness criterion is used. A pale inset plate must differ from this
    field and cannot become background merely because it is nearly white.
    """
    height, width = source.shape[:2]
    white = np.full_like(source, 255)
    for axis in (0, 1):
        first, last = (source[:, 0], source[:, -1]) if axis == 0 else (source[0], source[-1])
        if not np.array_equal(first, last):
            continue
        if len(first) > 1 and np.max(np.abs(np.diff(first.astype(np.int16), axis=0))) > 2:
            continue
        field = np.repeat(first[:, None, :], width, axis=1) if axis == 0 else np.repeat(first[None, :, :], height, axis=0)
        match = np.uint8(np.all(source == field, axis=2))
        count, labels, _, _ = cv2.connectedComponentsWithStats(match, 8)
        border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        owned = np.isin(labels, border_labels[border_labels > 0])
        if count > 1 and np.mean(owned) >= .35:
            return field, owned | np.all(source == 255, axis=2)
    return white, np.all(source == 255, axis=2)


def ownership_evidence(source: np.ndarray, layout: dict, asset_dir: Path, *, include_background: bool = True) -> tuple[np.ndarray, dict]:
    height, width = source.shape[:2]
    _, occupied = page_surface_evidence(source)
    if not include_background:
        occupied = np.zeros((height, width), bool)
    missing, invalid = [], []
    for item in layout.get("elements", []):
        if not _visible(item) or item.get("type") in {"background", "group"}:
            continue
        box = _box(item, width, height)
        if box is None:
            invalid.append(str(item.get("id")))
            continue
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            invalid.append(str(item.get("id")))
            continue
        patch = source[y1:y2, x1:x2]
        claim = np.zeros(patch.shape[:2], bool)
        kind = item.get("type")
        owner = None
        if kind == "text" and str(item.get("text") or "").strip() and float(item.get("confidence", 1) or 0) >= .5:
            raw = (item.get("metadata") or {}).get("rawOCRBBox")
            region = np.zeros((height, width), np.uint8)
            # OCR is the source of location, but a displaced replacement must
            # not authorize erasing text it no longer covers.
            if isinstance(raw, list) and len(raw) == 4:
                try:
                    a, b, c, d = [round(float(v)) for v in raw]
                except (ValueError, TypeError, OverflowError):
                    invalid.append(str(item.get("id")))
                    continue
                region[max(y1,b):min(y2,d), max(x1,a):min(x2,c)] = 255
            else:
                region[y1:y2, x1:x2] = 255
            mask = tight_text_mask(source, region, (item.get("style") or {}).get("color"))
            occupied |= mask > 0
            owner = "editable_text"
        elif kind == "image":
            src = str(item.get("src") or "").split("?", 1)[0]
            path = Path(src) if Path(src).is_absolute() and not src.startswith("/media/") else asset_dir / Path(src).name
            image = image_io.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
            if image is None or image.ndim != 3:
                missing.append(str(item.get("id")))
                continue
            if item.get("crop"):
                continue  # Cropped assets need transform evidence, not bbox trust.
            image = cv2.resize(image, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
            alpha = image[:, :, 3] if image.shape[2] == 4 else np.full(claim.shape, 255, np.uint8)
            # Claim only near-opaque source-matching pixels. Translucent support
            # is conservatively recovered, never treated as proof of replacement.
            claim = (alpha >= 250) & (np.max(np.abs(image[:, :, :3].astype(np.int16)-patch.astype(np.int16)), axis=2) <= 2)
            owner = "movable_image"
        elif kind in {"rectangle", "roundedRectangle", "ellipse", "line", "arrow"}:
            style = item.get("style") or {}
            if float(style.get("opacity", 1)) < .99:
                continue
            color = style.get("stroke") if kind in {"line", "arrow"} else style.get("fill")
            try:
                bgr = np.array(list(bytes.fromhex(str(color).lstrip("#")))[::-1], np.int16)
                if bgr.shape != (3,):
                    continue
            except (ValueError, TypeError):
                continue
            h, w = claim.shape
            canvas = Image.new("L", (w, h))
            draw = ImageDraw.Draw(canvas)
            if kind in {"rectangle", "roundedRectangle"}:
                draw.rounded_rectangle((0, 0, w, h), radius=min(w,h)*.16 if kind == "roundedRectangle" else 0, fill=255)
            elif kind == "ellipse":
                draw.ellipse((0, 0, w, h), fill=255)
            else:
                draw.line((0, h/2, w, h/2), fill=255, width=max(1, round(float(style.get("strokeWidth", 1)))))
            geometry = cv2.dilate(np.asarray(canvas), np.ones((3, 3), np.uint8))
            claim = (geometry > 0) & (np.max(np.abs(patch.astype(np.int16)-bgr), axis=2) <= 2)
            # Native shapes also replace their border. Geometry must agree,
            # rather than claiming every similarly-colored pixel in the bbox.
            if kind not in {"line", "arrow"} and style.get("stroke"):
                try:
                    stroke = np.array(list(bytes.fromhex(str(style["stroke"]).lstrip("#")))[::-1], np.int16)
                    boundary = cv2.morphologyEx(geometry, cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8))
                    # Include canvas edges in the narrow border envelope.
                    boundary[:3, :] = boundary[-3:, :] = 255
                    boundary[:, :3] = boundary[:, -3:] = 255
                    claim |= (boundary > 0) & (np.max(np.abs(patch.astype(np.int16)-stroke), axis=2) <= 2)
                except (ValueError, TypeError):
                    pass
            owner = "native_shape"
        occupied[y1:y2, x1:x2] |= claim
        if owner:
            item["owner"] = owner
            item["bbox"] = {"left": item["x"], "top": item["y"], "width": item["width"], "height": item["height"]}
            item["parent"] = item.get("groupId") or (item.get("metadata") or {}).get("moduleId")
            item["assetPath"] = item.get("src")
            item["editable"] = True
            item["reconstructionStrategy"] = owner
            item.setdefault("metadata", {})["ownerEvidence"] = "source_pixels" if kind != "text" else "tight_ocr_ink"
    unknown = int(np.count_nonzero(~occupied))
    return occupied, {"unownedPixelCount": unknown, "objectExtractionCoverage": round(float(np.mean(occupied)), 6),
                      "missingAssetCount": len(missing), "missingAssetIds": missing, "invalidOwnerIds": invalid}


def ensure_visual_owners(source: np.ndarray, layout: dict, asset_dir: Path, project_id: str, page: int) -> dict:
    """Persist and reopen replacement assets BEFORE permitting a white surface."""
    occupied, report = ownership_evidence(source, layout, asset_dir)
    unknown = np.uint8(~occupied)
    connected = cv2.dilate(unknown, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, 8)
    created = []
    asset_dir.mkdir(parents=True, exist_ok=True)
    z = _retained_layer(layout)
    regions = []
    height, width = source.shape[:2]
    for index in range(1, count):
        x, y, w, h, _ = [int(v) for v in stats[index]]
        # Keep fallback bounded even if a gradient or compression fringe
        # connects unrelated modules. Never introduce an opaque page screenshot.
        tw, th = max(1, width//3), max(1, height//3)
        if w*h <= width*height*.45:
            regions.append((index, x, y, w, h))
        else:
            regions.extend((index, a, b, min(tw,x+w-a), min(th,y+h-b))
                           for b in range(y,y+h,th) for a in range(x,x+w,tw))
    for index, x, y, w, h in regions:
        alpha = np.uint8((labels[y:y+h, x:x+w] == index) & (unknown[y:y+h, x:x+w] > 0)) * 255
        if not np.any(alpha):
            continue
        rgba = np.dstack((source[y:y+h, x:x+w], alpha))
        identifier = f"retained_{page}_{uuid4().hex[:12]}"
        path = asset_dir / f"{identifier}.png"
        if not image_io.imwrite(str(path), rgba):
            raise OSError("Replacement asset write failed; background clearing is forbidden")
        reread = image_io.imread(str(path), cv2.IMREAD_UNCHANGED)
        if reread is None or not np.array_equal(reread, rgba):
            raise OSError("Replacement asset verification failed; background clearing is forbidden")
        created.append({"id": identifier, "type": "image", "x": x, "y": y, "width": w, "height": h,
                        "rotation": 0, "zIndex": z, "src": f"/media/assets/{project_id}/{path.name}",
                        "confidence": 1, "source": "original_pixels", "owner": "movable_image", "editable": True,
                        "metadata": {"retainedVisual": True, "reconstructionStrategy": "local_image",
                                     "reconstructionStrategySource": "owner_gate", "sourcePixelArea": int(np.count_nonzero(alpha))}})
    # All replacements must succeed before changing the canonical scene.
    layout.setdefault("elements", []).extend(created)
    compacted = compact_retained_fragments(source, layout, asset_dir, project_id, page)
    _, final = ownership_evidence(source, layout, asset_dir)
    if final["unownedPixelCount"]:
        raise ValueError("Unowned source pixels remain; background clearing is forbidden")
    final.update({"recoveredPixelCount": report["unownedPixelCount"], "retainedAssetCount": len(created), "deletionAllowed": True})
    final["compactedFragmentCount"] = compacted
    layout.setdefault("metadata", {})["ownerGate"] = final
    layout["sceneVersion"] = "2.1-owner-evidence"
    layout["pageOwner"] = {"owner": "background", "source": "border_anchored_surface", "editable": False}
    return final


def compact_retained_fragments(source: np.ndarray, layout: dict, asset_dir: Path, project_id: str, page: int) -> int:
    """Coalesce sparse source fringes only after pixel-exact replacement proof.

    Larger/meaningful objects stay independent. Old immutable asset files and
    retired AST nodes remain available; no file or source pixel is deleted.
    """
    height, width = source.shape[:2]
    small = []
    union = np.zeros((height, width), np.uint8)
    for item in layout.get("elements", []):
        if not _visible(item) or not (item.get("metadata") or {}).get("retainedVisual") or (item.get("metadata") or {}).get("reconstructionStrategySource") != "owner_gate":
            continue
        box = _box(item, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        path = asset_dir / Path(str(item.get("src") or "")).name
        asset = image_io.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
        if asset is None or asset.shape != (y2-y1, x2-x1, 4):
            continue
        alpha = asset[:, :, 3]
        if np.count_nonzero(alpha) > 128 or not np.all((alpha == 0) | (alpha == 255)):
            continue
        if not np.array_equal(asset[:, :, :3][alpha > 0], source[y1:y2, x1:x2][alpha > 0]):
            continue
        small.append(item)
        union[y1:y2, x1:x2] |= alpha
    if len(small) < 32:
        return 0
    replacements = []
    evidence = np.zeros_like(union)
    z = _retained_layer(layout)
    tw, th = max(1, width//6), max(1, height//4)
    for y in range(0, height, th):
        for x in range(0, width, tw):
            region = union[y:y+th, x:x+tw]
            if not np.any(region):
                continue
            a, b, w, h = cv2.boundingRect(region)
            left, top = x+a, y+b
            alpha = region[b:b+h, a:a+w]
            rgba = np.dstack((source[top:top+h, left:left+w], alpha))
            identifier = f"retained_cluster_{page}_{uuid4().hex[:12]}"
            path = asset_dir / f"{identifier}.png"
            if not image_io.imwrite(str(path), rgba) or not np.array_equal(image_io.imread(str(path), -1), rgba):
                raise OSError("Fragment replacement verification failed; old owners remain active")
            evidence[top:top+h, left:left+w] |= alpha
            replacements.append({"id": identifier, "type": "image", "x": left, "y": top, "width": w, "height": h,
                                 "rotation": 0, "zIndex": z, "confidence": 1, "source": "original_pixels",
                                 "src": f"/media/assets/{project_id}/{path.name}", "owner": "movable_image", "editable": True,
                                 "metadata": {"retainedVisual": True, "reconstructionStrategy": "local_image",
                                              "reconstructionStrategySource": "owner_gate", "sourcePixelArea": int(np.count_nonzero(alpha)),
                                              "fragmentCluster": True}})
    if not np.array_equal(evidence, union):
        raise ValueError("Fragment replacement coverage is incomplete; old owners remain active")
    layout["elements"].extend(replacements)
    replacement_ids = [item["id"] for item in replacements]
    for item in small:
        item.setdefault("metadata", {}).update({"suppressed": True, "suppressRender": True,
                                                "replacementOwners": replacement_ids,
                                                "ownerEvidence": "pixel_exact_fragment_union"})
    return len(small)


def _retained_layer(layout: dict) -> int:
    active = [item for item in layout.get("elements", []) if _visible(item)]
    visual_z = [int(item.get("zIndex", 0)) for item in active
                if item.get("type") != "text" and (item.get("metadata") or {}).get("reconstructionStrategySource") != "owner_gate"]
    z = max(visual_z, default=0) + 1
    # Editable ordinary text is in front of support/fringe images. Preserve the
    # existing separation when there is a clear gap in the resolved layer stack.
    text_z = [int(item.get("zIndex", 0)) for item in active if item.get("type") == "text"]
    if text_z and max(visual_z, default=0) < min(text_z):
        z = min(z, min(text_z)-1)
    return z
