"""Source-pixel ownership, with verified assets BEFORE background construction.

No whitening, residual recovery, suppression, LaMa or revision is used here.
Uncertain text stays in its movable visual owner; it is NEVER also rendered
as editable text. Semantic proposals are hints, not permission to erase.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.utils import image_io


def active(element: dict) -> bool:
    return not any((element.get("metadata") or {}).get(k) for k in ("suppressed", "suppressRender", "ownedBy"))


def box_of(item: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    box = item.get("bboxPixels") or item.get("bbox")
    if isinstance(box, dict):
        x, y = box.get("left", 0), box.get("top", 0)
        vals = [x, y, x + box.get("width", 0), y + box.get("height", 0)]
        if max(abs(float(v)) for v in vals) <= 1.01:
            vals = [vals[0]*width, vals[1]*height, vals[2]*width, vals[3]*height]
    elif isinstance(box, (list, tuple)) and len(box) == 4:
        vals = box
    else:
        x, y = item.get("x", 0), item.get("y", 0)
        vals = [x, y, x + item.get("width", 0), y + item.get("height", 0)]
    if not all(np.isfinite(float(v)) for v in vals):
        return None
    x1, y1, x2, y2 = (round(float(v)) for v in vals)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _save_exact(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not image_io.imwrite(path, rgba):
        raise OSError("Replacement asset could not be written; source deletion is forbidden")
    reopened = image_io.imread(path, cv2.IMREAD_UNCHANGED)
    if reopened is None or not np.array_equal(rgba, reopened):
        raise OSError("Replacement asset failed round-trip verification")


def _text_candidate(source: np.ndarray, item: dict) -> tuple[dict, np.ndarray, np.ndarray] | None:
    """Only plain, uniform-paper text with a measured font match may be replaced.

    The full contrast mask (including antialiasing) is used, not a bbox fill.
    Border texture, badges, grid lines, gradients or incompatible fonts cause
    raster ownership instead. This is intentionally a conservative gate.
    """
    h, w = source.shape[:2]
    box = box_of(item, w, h)
    text = str(item.get("text") or "")
    if box is None or not text.strip() or "\n" in text or float(item.get("confidence") or 0) < .95:
        return None
    x1, y1, x2, y2 = box
    pad = 4
    x1, y1, x2, y2 = max(0, x1-pad), max(0, y1-pad), min(w, x2+pad), min(h, y2+pad)
    patch = source[y1:y2, x1:x2]
    if min(patch.shape[:2]) < 8:
        return None
    ring = np.concatenate((patch[:2].reshape(-1, 3), patch[-2:].reshape(-1, 3), patch[:, :2].reshape(-1, 3), patch[:, -2:].reshape(-1, 3)))
    colors, counts = np.unique(ring, axis=0, return_counts=True)
    bg = colors[counts.argmax()]
    # Tolerate mild compression, NOT a pale or patterned carrier gradient.
    if np.mean(np.max(np.abs(ring.astype(float)-bg), axis=1) <= 3) < .985:
        return None
    delta = np.max(np.abs(patch.astype(float)-bg), axis=2)
    ink = delta > 0
    if not ink.any() or np.mean(ink) > .45 or ink[:2].any() or ink[-2:].any() or ink[:, :2].any() or ink[:, -2:].any():
        return None
    ys, xs = np.where(ink)
    iy, ix, ih, iw = ys.min(), xs.min(), ys.max()-ys.min()+1, xs.max()-xs.min()+1
    dark = delta >= max(30, np.quantile(delta[ink], .65))
    if not dark.any():
        return None
    fg = np.median(patch[dark], axis=0).astype(np.uint8)
    # Multicolour illustrations inside an OCR bbox must remain images.
    if np.mean(np.max(np.abs(patch[dark].astype(float)-fg), axis=1) <= 28) < .95:
        return None
    target = ink[iy:iy+ih, ix:ix+iw].astype(np.uint8)
    best = None
    for family, filename in (("Microsoft YaHei", "msyh.ttc"), ("SimSun", "simsun.ttc"), ("Arial", "arial.ttf")):
        path = Path("C:/Windows/Fonts") / filename
        if not path.is_file():
            continue
        for size in range(max(8, int(ih*.8)), min(180, int(ih*1.55)+2)):
            font = ImageFont.truetype(str(path), size=size)
            bounds = font.getbbox(text)
            tw, th = bounds[2]-bounds[0], bounds[3]-bounds[1]
            if abs(tw-iw) > 3 or abs(th-ih) > 3 or min(tw, th) <= 0:
                continue
            raster = Image.new("L", (tw+6, th+6))
            ImageDraw.Draw(raster).text((3-bounds[0], 3-bounds[1]), text, font=font, fill=255)
            raw = np.array(raster) > 0
            py, px = np.where(raw)
            actual_w, actual_h = px.max()-px.min()+1, py.max()-py.min()+1
            if actual_w != iw or actual_h != ih:
                continue
            predicted = raw[py.min():py.max()+1, px.min():px.max()+1]
            union = np.count_nonzero(predicted | (target > 0))
            iou = np.count_nonzero(predicted & (target > 0))/max(1, union)
            if best is None or iou > best[0]:
                best = (iou, family, size, (bounds[0]+px.min()-3, bounds[1]+py.min()-3))
    if best is None or best[0] < .82:
        return None
    _, family, size, bounds = best
    element = copy.deepcopy(item)
    element.update(x=float(x1+ix-bounds[0]), y=float(y1+iy-bounds[1]), width=float(max(iw+8, size*len(text))), height=float(size*1.6),
                   type="text", owner="editable_text", editable=True, source="ocr", role=item.get("role", "body_text"))
    element["width"] = min(element["width"], w-element["x"])
    element["height"] = min(element["height"], h-element["y"])
    element["style"] = {**element.get("style", {}), "fontFamily": family, "fontSize": size, "fontWeight": 400,
                        "color": "#" + "".join(f"{v:02X}" for v in fg[::-1]), "align": "left", "verticalAlign": "top"}
    element["metadata"] = {"reconstructionStrategy": "editable_text", "exclusiveFontVerified": True,
                           "fontMaskIoU": round(best[0], 4), "rawOCRBBox": list(box), "originalLineCount": 1}
    mask = np.zeros((h, w), bool)
    # Every non-background pixel of the proven plain text is removed once.
    mask[y1:y2, x1:x2] = ink
    fill = np.broadcast_to(bg, source.shape).copy()
    return element, mask, fill


def build_exclusive_scene(source_path: Path, ocr_layout: dict, scene: dict, segments: list[dict],
                          output: Path, project_id: str, page: int) -> dict:
    source = image_io.imread(source_path)
    if source is None:
        raise FileNotFoundError("Source image cannot be decoded")
    h, w = source.shape[:2]
    labels = np.zeros((h, w), np.int32)
    ink = np.zeros((h, w), bool)
    cleaned = source.copy()
    elements: list[dict] = []
    nodes: list[dict] = []
    text_decisions = []
    prefix = f"exclusive_p{page}_{uuid4().hex[:8]}"
    assets = output / "assets"
    mask_dir = output / "ownership_masks" / prefix
    semantic = {i.get("id"): i for i in scene.get("elements", [])}

    def register(element: dict, mask: np.ndarray, owner: str, source_name: str) -> int:
        if np.any(labels[mask]):
            raise ValueError("Source pixels already have an owner; refusing a duplicate replacement")
        number = len(nodes)+1
        ys, xs = np.where(mask)
        if not len(xs):
            raise ValueError("An empty mask is not a replacement owner")
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
        identifier = str(element["id"])
        mask_file = mask_dir / f"{identifier}.png"
        rgba_mask = np.zeros((y2-y1, x2-x1, 4), np.uint8)
        rgba_mask[:, :, 3] = mask[y1:y2, x1:x2]*255
        _save_exact(mask_file, rgba_mask)
        bbox = {"left": element["x"], "top": element["y"], "width": element["width"], "height": element["height"]}
        element.update(owner=owner, source=source_name, bbox=bbox, editable=owner != "intentional_background",
                       parent=element.get("groupId"), assetPath=element.get("src"))
        element.setdefault("metadata", {})["exclusiveSourceOwner"] = number
        element["metadata"]["sourceMaskPath"] = str(mask_file.relative_to(output)).replace("\\", "/")
        element["metadata"]["sourceMaskBBox"] = [int(x1), int(y1), int(x2), int(y2)]
        nodes.append({"id": identifier, "owner": owner, "type": element["type"], "role": element.get("role"), "bbox": bbox,
                      "sourceBBox": [int(x1), int(y1), int(x2), int(y2)], "mask": element["metadata"]["sourceMaskPath"],
                      "parent": element.get("parent"), "zIndex": element.get("zIndex", 0), "source": source_name,
                      "confidence": element.get("confidence", 1), "reconstructionStrategy": element["metadata"].get("reconstructionStrategy"),
                      "assetPath": element.get("src"), "editable": element["editable"], "sourcePixelCount": int(mask.sum()), "verified": True})
        labels[mask] = number  # ONLY after the owner and mask have been persisted.
        elements.append(element)
        return number

    # OCR coordinates and body are authoritative. VLM assigns semantics only.
    for item in ocr_layout.get("elements", []):
        if item.get("type") != "text":
            continue
        candidate = _text_candidate(source, item)
        if candidate is not None and not np.any(ink & candidate[1]):
            element, mask, fill = candidate
            hints = semantic.get(item["id"]) or {}
            element["role"] = hints.get("role") or element.get("role")
            element["groupId"] = hints.get("groupId")
            register(element, mask, "editable_text", "ocr_verified_font")
            ink |= mask
            cleaned[mask] = fill[mask]  # No whole-bbox erase, no generative inpaint.
            decision = "editable_text"
        else:
            decision = "movable_image"
        text_decisions.append({"id": item["id"], "text": item.get("text"), "bbox": list(box_of(item, w, h) or ()),
                               "decision": decision, "reason": "verified_plain_text" if decision == "editable_text" else "preserve_source_until_replacement_is_reliable"})

    # Native geometry needs pixel evidence, not a VLM label or guessed colour.
    # Only fully uniform, axis-aligned rectangles pass this first safety gate.
    # Rounded/elliptic/irregular or textured carriers remain independent images.
    for item in ocr_layout.get("elements", []):
        if item.get("type") != "rectangle" or abs(float(item.get("rotation", 0))) > .01:
            continue
        box = box_of(item, w, h)
        if not box:
            continue
        x1,y1,x2,y2 = box
        patch = source[y1:y2,x1:x2]
        color = patch[0,0]
        if not np.all(patch == color) or np.any(labels[y1:y2,x1:x2]):
            continue
        mask = np.zeros((h,w), bool)
        mask[y1:y2,x1:x2] = True
        shape = {"id": item["id"], "type": "rectangle", "x": x1, "y": y1, "width": x2-x1, "height": y2-y1,
                 "rotation": 0, "zIndex": 1, "role": "verified_flat_carrier", "confidence": 1,
                 "style": {"fill": "#"+"".join(f"{v:02X}" for v in color[::-1]),
                           "stroke": "#"+"".join(f"{v:02X}" for v in color[::-1]), "strokeWidth": 0, "opacity": 1},
                 "metadata": {"reconstructionStrategy": "native_shape", "exclusiveGeometryVerified": True}}
        register(shape, mask, "native_shape", "cv_exact_uniform_rectangle")

    def image_owner(mask: np.ndarray, role: str, hint: str, parent: str | None = None) -> None:
        mask = mask & (labels == 0)
        if not mask.any():
            return
        ys, xs = np.where(mask)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
        identifier = f"{prefix}_visual_{len(nodes)+1:03d}"
        rgba = np.dstack((cleaned[y1:y2, x1:x2], (mask[y1:y2, x1:x2]*255).astype(np.uint8)))
        # Clean paper beneath native glyphs is a substrate, not an old glyph.
        substrate = ink[y1:y2, x1:x2]
        rgba[:, :, 3][substrate] = 255
        # PowerPoint resamples each PNG independently. Complementary hard masks
        # can otherwise expose white hairline cracks between two exact owners.
        # Extend ONLY the lower carrier's existing boundary colour by two pixels
        # beneath already-verified replacement owners. This is a synthetic
        # substrate, NOT another claim on their original source content.
        floor = rgba[:, :, 3] > 0
        eligible = labels[y1:y2, x1:x2] > 0
        synthetic_floor = np.zeros(floor.shape, bool)
        for _ in range(2):
            previous = floor.copy()
            previous_rgb = rgba[:, :, :3].copy()
            for axis, shift in ((0,1),(0,-1),(1,1),(1,-1)):
                neighbor = np.roll(previous, shift, axis=axis)
                if axis == 0:
                    neighbor[0 if shift == 1 else -1, :] = False
                else:
                    neighbor[:, 0 if shift == 1 else -1] = False
                extend = neighbor & ~floor & eligible & ~substrate
                shifted_rgb = np.roll(previous_rgb, shift, axis=axis)
                rgba[:, :, :3][extend] = shifted_rgb[extend]
                rgba[:, :, 3][extend] = 255
                floor |= extend
                synthetic_floor |= extend
        path = assets / f"{identifier}.png"
        _save_exact(path, rgba)  # A missing/corrupt asset aborts without a background write.
        element = {"id": identifier, "type": "image", "x": int(x1), "y": int(y1), "width": int(x2-x1), "height": int(y2-y1),
                   "src": f"/media/assets/{project_id}/{path.name}", "rotation": 0, "zIndex": 2,
                   "role": role, "groupId": parent, "style": {"opacity": 1}, "confidence": 1,
                   "metadata": {"reconstructionStrategy": "transparent_image", "proposal": hint,
                                "cleanSubstratePixels": int(substrate.sum()), "edgeSubstratePixels": int(synthetic_floor.sum()),
                                "edgeSubstratePolicy": "two_pixel_owned_boundary_color_not_source_copy", "preserveWholeAsset": True}}
        register(element, mask, "movable_image", hint)

    # Small segmented children precede carrier modules. Large parent masks then
    # retain ONLY the unclaimed carrier pixels, without duplicating the child.
    def segment_area(segment):
        box = box_of(segment, w, h)
        return (box[2]-box[0])*(box[3]-box[1]) if box else float("inf")

    for segment in sorted(segments, key=segment_area):
        box = box_of(segment, w, h)
        src = segment.get("alphaCrop")
        if not box or not src:
            continue
        asset = image_io.imread(assets / src.replace("\\", "/").split("/")[-1], cv2.IMREAD_UNCHANGED)
        x1, y1, x2, y2 = box
        if asset is None or asset.ndim != 3 or asset.shape[2] != 4 or asset.shape[:2] != (y2-y1, x2-x1):
            continue
        mask = np.zeros((h, w), bool)
        mask[y1:y2, x1:x2] = asset[:, :, 3] > 0
        image_owner(mask, "segmented_visual", segment.get("source", "opencv_segment"))

    modules = ((scene.get("vision") or {}).get("reconstructionPlan") or {}).get("modules") or []
    proposals = []
    for module in modules:
        box = box_of(module, w, h)
        if box and (box[2]-box[0])*(box[3]-box[1]) < .8*w*h:
            proposals.append((module, box))
    # Whole carriers first: nested labels cannot cut a header into fragments.
    proposals.sort(key=lambda pair: (pair[1][2]-pair[1][0])*(pair[1][3]-pair[1][1]), reverse=True)
    for module, (x1, y1, x2, y2) in proposals:
        mask = np.zeros((h, w), bool)
        mask[y1:y2, x1:x2] = True
        image_owner(mask, module.get("role") or "module_visual", "vlm_module_crop", module.get("moduleId"))

    # Fallback OCR is an image owner, not a hidden duplicate native text object.
    for decision in text_decisions:
        if decision["decision"] != "movable_image" or not decision["bbox"]:
            continue
        x1, y1, x2, y2 = decision["bbox"]
        mask = np.zeros((h, w), bool)
        mask[max(0,y1-2):min(h,y2+2), max(0,x1-2):min(w,x2+2)] = True
        image_owner(mask, "raster_text_fallback", "ocr_unreliable_font")

    # Background LAST: only an edge-connected, exact flat colour is intentional
    # background. Pale plates and disconnected white cards are NOT background.
    edge = np.concatenate((source[0], source[-1], source[:, 0], source[:, -1]))
    colors, counts = np.unique(edge, axis=0, return_counts=True)
    bg_color = colors[counts.argmax()]
    flat = np.all(source == bg_color, axis=2).astype(np.uint8)
    _, components = cv2.connectedComponents(flat, connectivity=8)
    edge_components = np.unique(np.concatenate((components[0], components[-1], components[:, 0], components[:, -1])))
    bg_mask = np.isin(components, edge_components[edge_components != 0]) & (labels == 0)
    # All other source content gets independent transparent assets, including
    # tiny decorations and antialiasing. Never discard a small component.
    remaining = (labels == 0) & ~bg_mask
    joined = cv2.dilate(remaining.astype(np.uint8), np.ones((5, 5), np.uint8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    for number in range(1, count):
        x, y, cw, ch, _ = stats[number]
        mask = np.zeros((h, w), bool)
        mask[y:y+ch, x:x+cw] = (components[y:y+ch, x:x+cw] == number) & remaining[y:y+ch, x:x+cw]
        image_owner(mask, "retained_decoration", "source_component_fallback")
    background_path = output / "backgrounds" / f"page_{page}.png"
    background = np.broadcast_to(bg_color, source.shape).copy()
    background_element = {"id": f"{prefix}_background", "type": "background", "x": 0, "y": 0, "width": w, "height": h,
                          "zIndex": 0, "src": f"/media/backgrounds/{project_id}/{background_path.name}", "role": "page_bottom",
                          "metadata": {"reconstructionStrategy": "background_image", "backgroundEvidence": "edge_connected_exact_flat_pixels"}}
    if bg_mask.any():
        register(background_element, bg_mask, "intentional_background", "edge_connected_flat_field")
    else:
        background_element.update(owner="intentional_background", editable=False)
        elements.insert(0, background_element)  # Empty bottom support has no source-content claim.
    if np.any(labels == 0):
        raise ValueError("Unowned source pixels: background construction is forbidden")
    # Only now is it legal to emit the page-bottom canvas.
    _save_exact(background_path, cv2.cvtColor(background, cv2.COLOR_BGR2BGRA))
    label_bytes = np.dstack((labels & 255, (labels >> 8) & 255, (labels >> 16) & 255)).astype(np.uint8)
    image_io.imwrite(output / f"ownership_labels_{page}.png", label_bytes)
    # Carriers draw below their smaller children; otherwise an extrapolated
    # carrier edge could paint over the object it is intended to support.
    visuals = sorted((e for e in elements if e.get("owner") == "movable_image"),
                     key=lambda e: e["width"]*e["height"], reverse=True)
    for rank, element in enumerate(visuals, 2):
        element["zIndex"] = rank
    for element in elements:
        if element.get("owner") == "editable_text":
            element["zIndex"] = len(visuals)+3
    owner_layers = np.zeros(len(nodes)+1, np.int32)
    for element in elements:
        number = (element.get("metadata") or {}).get("exclusiveSourceOwner")
        if number:
            owner_layers[number] = element.get("zIndex", 0)
    # Never let a synthetic floor paint on top of the verified original owner.
    # This layer check is independent of segmentation / VLM insertion order.
    for element in visuals:
        x,y = element["x"],element["y"]
        cw,ch = element["width"],element["height"]
        path = assets / element["src"].split("/")[-1]
        rgba = image_io.imread(path, cv2.IMREAD_UNCHANGED)
        owned = labels[y:y+ch,x:x+cw] == element["metadata"]["exclusiveSourceOwner"]
        glyph_substrate = ink[y:y+ch,x:x+cw]
        synthetic = (rgba[:,:,3] > 0) & ~owned & ~glyph_substrate
        keep_floor = owner_layers[labels[y:y+ch,x:x+cw]] > element["zIndex"]
        rgba[:,:,3][synthetic & ~keep_floor] = 0
        element["metadata"]["edgeSubstratePixels"] = int((synthetic & keep_floor).sum())
        _save_exact(path,rgba)
    by_id = {e["id"]: e for e in elements}
    for node in nodes:
        node["zIndex"] = by_id[node["id"]].get("zIndex", 0)
    for decision in text_decisions:
        box = decision["bbox"]
        if box:
            x1, y1, x2, y2 = box
            decision["replacementOwners"] = [nodes[i-1]["id"] for i in np.unique(labels[y1:y2, x1:x2]) if i > 0]
    graph = {"version": "3.0-exclusive", "policy": "exactly_one_source_owner", "nodes": nodes,
             "textDecisions": text_decisions, "unownedPixelCount": int((labels == 0).sum()),
             "sourcePixelCount": w*h, "automaticRevisionsPaused": True, "inpaintCalls": 0}
    (output / f"ownership_audit_{page}.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"version": "1.1", "sceneVersion": "3.0-exclusive-owner-evidence", "slide": {"width": w, "height": h},
              "coordinateSystem": "source-pixels-left-top", "source": source_path.name,
              "backgroundUrl": background_element["src"], "elements": sorted(elements, key=lambda e: e.get("zIndex", 0)),
              "metadata": {"reconstructionSurfaceMode": "exclusive_object_first", "ownershipAudit": graph,
                           "automaticRevisionsPaused": True, "reconstructionPlan": {"modules": modules}}}
    return result


def audit_raster_text(source_path: Path, layout: dict, output: Path) -> dict:
    """Audit actual old glyph pixels present in rendered raster layers.

    Geometric bbox overlap is NOT evidence of duplicate visual content. Only
    opaque raster pixels matching source ink are counted against native text.
    Results are conservative evidence, not a claim to catch every ghost.
    """
    source = image_io.imread(source_path)
    if source is None:
        raise FileNotFoundError("Audit source is unreadable")
    h, w = source.shape[:2]
    raster = np.full_like(source, 255)
    claims = np.zeros((h, w), np.uint16)
    missing = []
    image_count = 0
    for item in sorted(layout.get("elements", []), key=lambda e: e.get("zIndex", 0)):
        if not active(item) or item.get("type") not in {"image", "background"}:
            continue
        src = str(item.get("src") or "")
        directory = "backgrounds" if item.get("type") == "background" else "assets"
        path = Path(src) if Path(src).is_file() else output / directory / src.replace("\\", "/").split("/")[-1]
        asset = image_io.imread(path, cv2.IMREAD_UNCHANGED)
        box = box_of(item, w, h)
        if asset is None or box is None:
            missing.append(item.get("id"))
            continue
        x1, y1, x2, y2 = box
        asset = cv2.resize(asset, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        if asset.ndim == 2:
            asset = cv2.cvtColor(asset, cv2.COLOR_GRAY2BGRA)
        alpha = asset[:, :, 3:4].astype(float)/255 if asset.shape[2] == 4 else np.ones((*asset.shape[:2], 1))
        exact_source = np.max(np.abs(asset[:, :, :3].astype(float)-source[y1:y2, x1:x2]), axis=2) <= 3
        evidence = (item.get("metadata") or {}).get("sourceMaskPath")
        if evidence:
            owner_mask = image_io.imread(output / evidence, cv2.IMREAD_UNCHANGED)
            if owner_mask is not None:
                # A measured carrier substrate is not a second original-ink claim.
                mask_box = (item.get("metadata") or {}).get("sourceMaskBBox") or box
                mx1,my1,mx2,my2 = mask_box
                source_claim = np.zeros((h,w), bool)
                source_claim[my1:my2,mx1:mx2] = owner_mask[:, :, 3] > 0
                exact_source &= source_claim[y1:y2,x1:x2]
        claims[y1:y2, x1:x2] += exact_source & (alpha[:, :, 0] >= .99)
        raster[y1:y2, x1:x2] = np.rint(asset[:, :, :3]*alpha + raster[y1:y2, x1:x2]*(1-alpha)).astype(np.uint8)
        image_count += item.get("type") == "image"
    duplicates = []
    # Existing OCR source footprints, even when a font replacement is unreliable.
    from app.services.reconstruction.owner_gate import tight_text_mask
    for item in layout.get("elements", []):
        if item.get("type") != "text" or not active(item):
            continue
        raw = (item.get("metadata") or {}).get("rawOCRBBox")
        box = tuple(map(int, raw)) if raw and len(raw) == 4 else box_of(item, w, h)
        if not box:
            continue
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
        patch = source[y1:y2, x1:x2]
        mask = tight_text_mask(patch, np.ones(patch.shape[:2], np.uint8)*255, (item.get("style") or {}).get("color")) > 0
        match = np.max(np.abs(source[y1:y2, x1:x2].astype(float)-raster[y1:y2, x1:x2]), axis=2) <= 3
        count = int((mask & match).sum())
        ratio = count/max(1, int(mask.sum()))
        if count >= 8 and ratio >= .08:
            duplicates.append({"id": item["id"], "text": item.get("text"), "sourceBBox": [x1,y1,x2,y2], "retainedOriginalInkPixels": count, "inkRatio": round(ratio,4)})
    important = np.max(np.abs(source.astype(float)-255), axis=2) > 6
    multiply_owned = (claims > 1) & important
    coverage_image = np.zeros((h,w,3), np.uint8)
    coverage_image[(claims == 1) & important] = (190,130,30)
    coverage_image[multiply_owned] = (40,40,220)
    image_io.imwrite(output / "raster_ownership_debug.png", coverage_image)
    return {"activeImageCount": image_count, "missingAssets": missing, "missingAssetCount": len(missing),
            "multiplyClaimedRasterVisualPixels": int(multiply_owned.sum()),
            "rasterNativeDuplicateTextCount": len(duplicates), "rasterNativeDuplicateTexts": duplicates,
            "method": "visible_raster_source_ink_match_not_bbox_overlap"}
