"""Source-pixel ownership, with verified assets BEFORE background construction.

No whitening, residual recovery, suppression, LaMa or revision is used here.
Uncertain text stays in its movable visual owner; it is NEVER also rendered
as editable text. Semantic proposals are hints, not permission to erase.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.utils import image_io


def active(element: dict) -> bool:
    return not any((element.get("metadata") or {}).get(k) for k in ("suppressed", "suppressRender", "ownedBy"))


def box_of(item: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    box = item.get("bboxPixels") or item.get("bbox")
    if isinstance(box, dict):
        x, y = box.get("left", 0), box.get("top", 0)
        vals = [x, y, x + box.get("width", 0), y + box.get("height", 0)]
        if max(abs(float(v)) for v in (x,y,box.get('width',0),box.get('height',0))) <= 1.01:
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


def _text_candidate(source: np.ndarray, item: dict, hints: dict | None = None,
                    diagnostics: dict | None = None) -> tuple[dict, np.ndarray, np.ndarray] | None:
    from app.services.reconstruction.editable_text import restore_text_candidate
    return restore_text_candidate(source,item,hints,diagnostics)


def build_exclusive_scene(source_path: Path, ocr_layout: dict, scene: dict, segments: list[dict],
                          output: Path, project_id: str, page: int) -> dict:
    source = image_io.imread(source_path)
    if source is None:
        raise FileNotFoundError("Source image cannot be decoded")
    h, w = source.shape[:2]
    labels = np.zeros((h, w), np.int32)
    ink = np.zeros((h, w), bool)
    substrate_claimed = np.zeros((h,w),bool)
    cleaned = source.copy()
    elements: list[dict] = []
    nodes: list[dict] = []
    photographic_owners: list[int] = []
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

    from app.services.reconstruction.text_cohorts import prepare_text_lines
    prepared_lines = prepare_text_lines(source,
        [i for i in ocr_layout.get('elements', []) if i.get('type') == 'text'], semantic)
    # OCR coordinates and body are authoritative. VLM assigns semantics only.
    for item in ocr_layout.get("elements", []):
        if item.get("type") != "text":
            continue
        hints = semantic.get(item["id"]) or {}
        candidates, diagnostics = prepared_lines[item['id']]
        # Validate the WHOLE OCR line before registering any span. Adjacent
        # glyph repair halos may overlap only when their measured paper agrees.
        pending = ink.copy()
        pending_fill = cleaned.copy()
        prepared = []
        for element, mask, fill in candidates:
            overlap = pending & mask
            if overlap.any() and np.max(np.abs(pending_fill[overlap].astype(float)-fill[overlap]))>7:
                diagnostics['reason'] = 'conflicting_text_owner_retained'
                prepared = []
                break
            mask = mask & ~pending
            if not mask.any():
                diagnostics['reason'] = 'duplicate_ocr_owner_retained'
                prepared = []
                break
            prepared.append((element,mask,fill))
            pending |= mask
            pending_fill[mask] = fill[mask]
        candidates = prepared
        restored_ids=[]
        editable_characters=0
        for candidate in candidates:
            element, mask, fill = candidate
            element["role"] = hints.get("role") or element.get("role")
            element["groupId"] = hints.get("groupId")
            register(element, mask, "editable_text", "ocr_verified_font")
            ink |= mask
            # Repair ONLY after replacement + source mask are persisted. A
            # measured smooth local surface avoids Telea's wave/blotch artifacts
            # in dense text. The card itself is never an authorized erase mask.
            cleaned[mask] = fill[mask]
            element["metadata"]["textRepairStrategy"] = "owner_gated_measured_local_paper"
            restored_ids.append(element['id'])
            editable_characters += len(str(element.get('text') or ''))
        decision = "editable_text" if restored_ids else "movable_image"
        text_decisions.append({"id": item["id"], "text": item.get("text"), "bbox": list(box_of(item, w, h) or ()),
                               "decision": decision,"editableSpanIds":restored_ids,
                               "editableCharacterCount":editable_characters,"sourceCharacterCount":len(str(item.get('text') or '')),
                               "reason": diagnostics.get("reason"), "evidence": diagnostics})

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
        patch = cleaned[y1:y2,x1:x2]
        color = patch[0,0]
        if not np.all(patch == color) or np.any((labels>0)[y1:y2,x1:x2] & ~ink[y1:y2,x1:x2]):
            continue
        mask = np.zeros((h,w), bool)
        mask[y1:y2,x1:x2] = labels[y1:y2,x1:x2]==0
        if not mask.any():
            continue
        shape = {"id": item["id"], "type": "rectangle", "x": x1, "y": y1, "width": x2-x1, "height": y2-y1,
                 "rotation": 0, "zIndex": 1, "role": "verified_flat_carrier", "confidence": 1,
                 "style": {"fill": "#"+"".join(f"{v:02X}" for v in color[::-1]),
                           "stroke": "#"+"".join(f"{v:02X}" for v in color[::-1]), "strokeWidth": 0, "opacity": 1},
                 "metadata": {"reconstructionStrategy": "native_shape", "exclusiveGeometryVerified": True}}
        substrate = ink[y1:y2,x1:x2] & ~substrate_claimed[y1:y2,x1:x2]
        paper_path = mask_dir / f"{item['id']}_paper.png"
        paper_mask = np.zeros((y2-y1,x2-x1,4),np.uint8)
        paper_mask[:,:,3] = substrate.astype(np.uint8)*255
        _save_exact(paper_path,paper_mask)
        shape['metadata'].update(paperSubstrateMask=str(paper_path.relative_to(output)).replace('\\','/'),
                                 paperSubstrateBBox=[x1,y1,x2,y2],cleanSubstratePixels=int(substrate.sum()))
        register(shape, mask, "native_shape", "cv_exact_uniform_rectangle")
        substrate_claimed[y1:y2,x1:x2] |= substrate

    def image_owner(mask: np.ndarray, role: str, hint: str, parent: str | None = None,
                    seed: tuple[tuple[int,int,int,int],np.ndarray,dict] | None = None) -> None:
        support = mask.copy()
        mask = mask & (labels == 0)
        if not mask.any():
            return
        ys, xs = np.where(mask)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
        identifier = f"{prefix}_visual_{len(nodes)+1:03d}"
        rgba = np.dstack((cleaned[y1:y2, x1:x2], (mask[y1:y2, x1:x2]*255).astype(np.uint8)))
        seed_evidence = {}
        if seed is not None:
            (sx,sy,_,_),seed_rgba,seed_evidence = seed
            source_rgba = seed_rgba[y1-sy:y2-sy,x1-sx:x2-sx]
            rgba[:,:,:3] = source_rgba[:,:,:3]
            rgba[:,:,3] = np.where(mask[y1:y2,x1:x2],source_rgba[:,:,3],0)
        # Clean paper beneath native glyphs is a substrate, not an old glyph.
        # Exactly one raster carrier supplies repaired paper beneath each
        # native glyph. Copying the same jagged substrate into every overlapping
        # image creates interpolation seams in PowerPoint even when PIL looks OK.
        substrate = ink[y1:y2, x1:x2] & support[y1:y2,x1:x2] & ~substrate_claimed[y1:y2,x1:x2]
        if not seed_evidence.get('carrierSurfaceRebuilt'):
            rgba[:,:,:3][substrate] = cleaned[y1:y2,x1:x2][substrate]
        rgba[:, :, 3][substrate] = 255
        from app.services.reconstruction.contour_objects import complete_carrier_holes
        if seed_evidence.get('carrierSurfaceRebuilt'):
            # Use the plate's measured field under every retained child. A
            # second child-local fit can reintroduce white icon rectangles or
            # old card colours into this already-clean background.
            floor_mask = support[y1:y2,x1:x2]
            filled = floor_mask & (rgba[:,:,3] == 0)
            rgba[:,:,3][floor_mask] = 255
            completion = {'completedChildSurfacePixels': int(filled.sum()),
                          'unresolvedChildSurfacePixels': 0,
                          'childSurfacePolicy': 'same_rebuilt_carrier_field_below_child_owners'}
        elif seed_evidence.get('visualAssetPolicy') == 'closed_carrier_outer_contour':
            _, completion = complete_carrier_holes(cleaned,rgba,(x1,y1,x2,y2),
                support[y1:y2,x1:x2],labels,ink)
        else:
            completion = {'completedChildSurfacePixels': 0, 'unresolvedChildSurfacePixels': 0,
                          'childSurfacePolicy': 'foreground_keeps_real_alpha_without_invented_floor'}
        # PowerPoint resamples each PNG independently. Complementary hard masks
        # can otherwise expose white hairline cracks between two exact owners.
        # Extend ONLY the lower carrier's existing boundary colour by two pixels
        # beneath already-verified replacement owners. This is a synthetic
        # substrate, NOT another claim on their original source content.
        floor = rgba[:, :, 3] > 0
        eligible = (labels[y1:y2, x1:x2] > 0) & ~ink[y1:y2,x1:x2]
        if not seed_evidence.get('carrierSupportMask') and seed_evidence.get('visualAssetPolicy') != 'closed_carrier_outer_contour':
            # Foreground underpainting must not leave blue arrow fragments or
            # white frames when an independent photograph is moved away.
            eligible &= ~np.isin(labels[y1:y2,x1:x2], photographic_owners)
        if seed_evidence.get('carrierSurfaceRebuilt'):
            eligible &= support[y1:y2,x1:x2]
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
        if seed_evidence.get('carrierSurfaceRebuilt'):
            error = np.max(np.abs(rgba[:,:,:3].astype(float)-source[y1:y2,x1:x2]),axis=2)
            source_owned = mask[y1:y2,x1:x2]
            if np.any(error[source_owned]>6):
                raise ValueError('Carrier foreground must have a persisted owner before surface reconstruction')
            approximation = np.zeros_like(rgba)
            approximation[:,:,3] = (source_owned & (error>1)).astype(np.uint8)*255
            approximation_path = mask_dir/f'{identifier}_tone.png'
            _save_exact(approximation_path,approximation)
            seed_evidence.update(surfaceApproximationMask=str(approximation_path.relative_to(output)).replace('\\','/'),
                                 surfaceApproximationMaxError=6,
                                 surfaceApproximationPixels=int((approximation[:,:,3]>0).sum()))
        if seed_evidence.get('visualAssetPolicy') == 'closed_carrier_outer_contour':
            support_path = mask_dir / f'{identifier}_support.png'
            support_rgba = np.zeros(rgba.shape,np.uint8)
            support_rgba[:,:,3] = support[y1:y2,x1:x2].astype(np.uint8)*255
            _save_exact(support_path,support_rgba)
            seed_evidence['carrierSupportMask'] = str(support_path.relative_to(output)).replace('\\','/')
        substrate_path = mask_dir / f"{identifier}_paper.png"
        paper_mask = np.zeros(rgba.shape,np.uint8)
        paper_mask[:,:,3] = substrate.astype(np.uint8)*255
        _save_exact(substrate_path,paper_mask)
        substrate_claimed[y1:y2,x1:x2] |= substrate
        element = {"id": identifier, "type": "image", "x": int(x1), "y": int(y1), "width": int(x2-x1), "height": int(y2-y1),
                   "src": f"/media/assets/{project_id}/{path.name}", "rotation": 0, "zIndex": 2,
                   "role": role, "groupId": parent, "style": {"opacity": 1}, "confidence": 1,
                   "metadata": {"reconstructionStrategy": "transparent_image", "proposal": hint,
                                **seed_evidence,
                                **completion,
                                "paperSubstrateMask":str(substrate_path.relative_to(output)).replace("\\","/"),
                                "paperSubstrateBBox":[int(x1),int(y1),int(x2),int(y2)],
                                "cleanSubstratePixels": int(substrate.sum()), "edgeSubstratePixels": int(synthetic_floor.sum()),
                                "edgeSubstratePolicy": "two_pixel_owned_boundary_color_not_source_copy", "preserveWholeAsset": True}}
        number = register(element, mask, "movable_image", hint)
        if seed_evidence.get('visualAssetPolicy') == 'source_verified_photographic_rectangle':
            photographic_owners.append(number)

    # Resolve nested contour proposals before ownership. Smallest-first used to
    # cut one photo stack/title into a center and multiple disconnected frames.
    from app.services.reconstruction.contour_objects import plan_contour_objects
    planned, contour_audit = plan_contour_objects(cleaned,segments,assets,text_decisions,box_of)
    for proposal in planned:
        if proposal['panel'] and proposal['evidence'].get('carrierBoundaryVerified'):
            from app.services.reconstruction.carrier_repair import (
                separate_carrier_surface,
            )
            x1,y1,x2,y2 = proposal['box']
            support = proposal['rgba'][:,:,3]>0
            separated = separate_carrier_surface(
                cleaned[y1:y2,x1:x2], support, labels[y1:y2,x1:x2]==0)
            if separated is not None:
                surface, details, _ = separated
                detail_ids = []
                for detail in details:
                    mask = np.zeros((h,w),bool)
                    mask[y1:y2,x1:x2] = detail
                    before = len(elements)
                    image_owner(mask,'carrier_detail','measured_carrier_foreground',seed=(
                        proposal['box'],np.dstack((cleaned[y1:y2,x1:x2],detail.astype(np.uint8)*255)),
                        {'visualAssetPolicy':'independent_carrier_detail',
                         'carrierDetailOf':proposal['evidence']['contourProposalId']}))
                    if len(elements)>before:
                        detail_ids.append(elements[-1]['id'])
                proposal['rgba'][:,:,:3] = surface
                proposal['evidence'].update(carrierSurfaceRebuilt=True,carrierDetailIds=detail_ids,
                    carrierSurfacePolicy='rim_measured_paper_after_independent_foreground_owners')
        refined = (proposal['box'],proposal['rgba'],proposal['evidence'])
        (x1,y1,x2,y2),rgba_seed,_ = refined
        mask = np.zeros((h, w), bool)
        mask[y1:y2, x1:x2] = rgba_seed[:, :, 3] > 0
        if proposal['evidence'].get('carrierRejectedReason'):
            # After complete photo owners are installed, the residual strip
            # contains independent arrows, not one draggable photo-frame net.
            residual = mask[y1:y2,x1:x2] & (labels[y1:y2,x1:x2] == 0)
            count, parts, stats, _ = cv2.connectedComponentsWithStats(residual.astype(np.uint8), 8)
            fragments = np.zeros((h, w), bool)
            for number in range(1, count):
                piece = np.zeros((h, w), bool)
                piece[y1:y2,x1:x2] = parts == number
                if stats[number, cv2.CC_STAT_AREA] < 24 or min(stats[number, 2:4]) < 3:
                    fragments |= piece
                    continue
                image_owner(piece, 'segmented_visual', proposal['source'], seed=refined)
            if fragments.any():
                image_owner(fragments, 'segmented_visual', proposal['source'], seed=refined)
            continue
        image_owner(mask, "carrier_surface" if proposal['panel'] else "segmented_visual",proposal['source'],seed=refined)

    modules = ((scene.get("vision") or {}).get("reconstructionPlan") or {}).get("modules") or []
    # The VLM plan is a TREE. Previously only root modules were read, so a
    # whole grid claimed all its children's pixels before they could be moved.
    def descendants(items, parent=None):
        for item in items:
            yield {**item, 'parentModule':parent}
            yield from descendants(item.get('children') or [], item.get('moduleId') or item.get('id'))

    # Semantic modules provide grouping only. Their approximate bounding boxes
    # are NOT source contours and must never slice banners or nearby OCR lines.
    module_boxes = [(m,box_of(m,w,h)) for m in descendants(modules)]
    for element in elements:
        eb = box_of(element,w,h)
        if eb is None:
            continue
        candidates = [(m,b) for m,b in module_boxes if b and b[0]-3<=eb[0] and b[1]-3<=eb[1] and b[2]+3>=eb[2] and b[3]+3>=eb[3]]
        if candidates:
            module,_ = min(candidates,key=lambda pair:(pair[1][2]-pair[1][0])*(pair[1][3]-pair[1][1]))
            element['groupId'] = module.get('moduleId') or module.get('id')
            element['parent'] = element['groupId']

    from app.services.reconstruction.carrier_repair import inherit_carrier_groups
    inherit_carrier_groups(elements)

    # Fallback OCR is an image owner, not a hidden duplicate native text object.
    for decision in text_decisions:
        if decision["decision"] != "movable_image" or not decision["bbox"]:
            continue
        x1, y1, x2, y2 = decision["bbox"]
        mask = np.zeros((h, w), bool)
        mask[max(0,y1-2):min(h,y2+2), max(0,x1-2):min(w,x2+2)] = True
        image_owner(mask, "raster_text_fallback", "ocr_unreliable_font")

    # Glyph extraction can isolate little paper islands between Chinese strokes.
    # Keep these in ONE coherent line carrier, not dozens of jagged 1px assets.
    # Existing semantic/SAM carrier owners take precedence; nothing is recopied.
    for decision in text_decisions:
        if decision["decision"] != "editable_text" or not decision["bbox"]:
            continue
        x1,y1,x2,y2 = decision["bbox"]
        mask = np.zeros((h,w),bool)
        # Avoid a separate rectangular patch once a complete carrier already
        # supplies the paper under this line.
        if not np.any(ink[y1:y2,x1:x2] & ~substrate_claimed[y1:y2,x1:x2]):
            continue
        mask[max(0,y1-4):min(h,y2+4),max(0,x1-4):min(w,x2+4)] = True
        image_owner(mask,"text_paper_carrier","ocr_local_paper",decision["id"])

    # A closed detector contour can stop inside an antialiased/soft border.
    # Keep that narrow unclaimed fringe with the module BEFORE residual
    # components connect it to page-wide decoration. No existing owner changes.
    from app.services.reconstruction.carrier_repair import unowned_contour_fringe
    for carrier in list(elements):
        support_file = carrier.get('metadata', {}).get('carrierSupportMask')
        if not support_file or not carrier.get('groupId'):
            continue
        cx,cy,cw,ch = (carrier[k] for k in ('x','y','width','height'))
        radius = max(3,min(6,round(min(cw,ch)*.02)))
        ax,ay,bx,by = max(0,cx-radius),max(0,cy-radius),min(w,cx+cw+radius),min(h,cy+ch+radius)
        support = np.zeros((by-ay,bx-ax),bool)
        support[cy-ay:cy-ay+ch,cx-ax:cx-ax+cw] = image_io.imread(output/support_file,cv2.IMREAD_UNCHANGED)[:,:,3]>0
        fringe = unowned_contour_fringe(support, labels[ay:by,ax:bx]>0, radius)
        if not fringe.any():
            continue
        mask = np.zeros((h,w),bool)
        mask[ay:by,ax:bx] = fringe
        image_owner(mask,'module_contour_fringe','closed_contour_antialias_fringe',carrier['groupId'])

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
    # Tight object masks may exclude a few native-glyph halo pixels that lie
    # just outside an OCR bbox. Attach their already-verified clean paper to one
    # existing carrier, rather than leave holes or copy old glyph pixels back.
    for element in sorted((e for e in elements if e.get('owner')=='movable_image'),
                          key=lambda e:e['width']*e['height'],reverse=True):
        x,y,cw,ch = (element[k] for k in ('x','y','width','height'))
        missing = ink[y:y+ch,x:x+cw] & ~substrate_claimed[y:y+ch,x:x+cw]
        if not missing.any():
            continue
        path = assets/element['src'].split('/')[-1]
        rgba = image_io.imread(path,cv2.IMREAD_UNCHANGED)
        rgba[:,:,:3][missing] = cleaned[y:y+ch,x:x+cw][missing]
        rgba[:,:,3][missing] = 255
        _save_exact(path,rgba)
        paper_path = output/element['metadata']['paperSubstrateMask']
        paper_mask = image_io.imread(paper_path,cv2.IMREAD_UNCHANGED)
        paper_mask[:,:,3][missing] = 255
        _save_exact(paper_path,paper_mask)
        element['metadata']['cleanSubstratePixels'] += int(missing.sum())
        element['metadata']['completedGlyphHaloPixels'] = int(missing.sum())
        substrate_claimed[y:y+ch,x:x+cw] |= missing
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
    # Native glyph ownership is not the layer of the paper beneath the glyph.
    # A new synthetic detail above that paper must not paint a second substrate
    # just because the native text itself renders on the highest layer.
    surface_layers = owner_layers[labels]
    for element in elements:
        metadata = element.get('metadata', {})
        if not metadata.get('paperSubstrateMask'):
            continue
        paper = image_io.imread(output/metadata['paperSubstrateMask'],cv2.IMREAD_UNCHANGED)[:,:,3]>0
        x,y,cw,ch = (element[k] for k in ('x','y','width','height'))
        surface_layers[y:y+ch,x:x+cw][paper] = element['zIndex']
    # Adjacent child masks can form one enormous connected hole in a parent.
    # Measure each already-extracted child's local paper separately instead
    # of trying to extrapolate one field through all cards/modules at once.
    from app.services.reconstruction.editable_text import measured_surface
    for child in reversed(visuals):
        if child['metadata'].get('visualAssetPolicy') not in {
                'supersampled_colour_decontaminated_matte', 'source_verified_photographic_rectangle'}:
            continue
        cx,cy,cw,ch = (child[k] for k in ('x','y','width','height'))
        parents = [e for e in visuals if e['metadata'].get('carrierSupportMask')
                   and e['zIndex']<child['zIndex']
                   and e['x']<=cx and e['y']<=cy
                   and e['x']+e['width']>=cx+cw and e['y']+e['height']>=cy+ch]
        if not parents:
            continue
        model = None
        for pad in (6,10,16):
            ax,ay,bx,by = max(0,cx-pad),max(0,cy-pad),min(w,cx+cw+pad),min(h,cy+ch+pad)
            model = measured_surface(cleaned[ay:by,ax:bx],border=2)
            if model is not None:
                break
        if model is None:
            continue
        parent = min(parents,key=lambda e:e['width']*e['height'])
        if parent['metadata'].get('carrierSurfaceRebuilt'):
            continue  # Its own measured field is complete and silhouette-bounded.
        path = assets/parent['src'].split('/')[-1]
        rgba = image_io.imread(path,cv2.IMREAD_UNCHANGED)
        lx,ly = cx-parent['x'],cy-parent['y']
        target = rgba[ly:ly+ch,lx:lx+cw]
        missing = (target[:,:,3]==0)&(labels[cy:cy+ch,cx:cx+cw]==child['metadata']['exclusiveSourceOwner'])
        target[:,:,:3][missing] = model[0][cy-ay:cy-ay+ch,cx-ax:cx-ax+cw][missing]
        target[:,:,3][missing] = 255
        parent['metadata']['individualChildSurfacePixels'] = parent['metadata'].get('individualChildSurfacePixels',0)+int(missing.sum())
        parent['metadata']['unresolvedChildSurfacePixels'] = max(0,parent['metadata'].get('unresolvedChildSurfacePixels',0)-int(missing.sum()))
        _save_exact(path,rgba)
    # Never let a synthetic floor paint on top of the verified original owner.
    # This layer check is independent of segmentation / VLM insertion order.
    for element in visuals:
        x,y = element["x"],element["y"]
        cw,ch = element["width"],element["height"]
        path = assets / element["src"].split("/")[-1]
        rgba = image_io.imread(path, cv2.IMREAD_UNCHANGED)
        owned = labels[y:y+ch,x:x+cw] == element["metadata"]["exclusiveSourceOwner"]
        glyph_substrate = ink[y:y+ch,x:x+cw]
        own_paper = image_io.imread(output/element['metadata']['paperSubstrateMask'],cv2.IMREAD_UNCHANGED)[:,:,3]>0
        synthetic = (rgba[:,:,3] > 0) & ~owned & ~own_paper
        keep_floor = surface_layers[y:y+ch,x:x+cw] > element["zIndex"]
        rgba[:,:,3][synthetic & ~keep_floor] = 0
        element["metadata"]["edgeSubstratePixels"] = int((synthetic & keep_floor).sum())
        support_file = element['metadata'].get('carrierSupportMask')
        if support_file:
            from app.services.reconstruction.carrier_repair import complete_local_floor
            from app.services.reconstruction.contour_objects import _visible_surface
            support = image_io.imread(output/support_file,cv2.IMREAD_UNCHANGED)[:,:,3]>0
            missing = support & (rgba[:,:,3]==0) & keep_floor
            # After verified child-local paper is installed, an enclosing card
            # may finally have enough supported samples for one smooth field.
            # This is a single bounded fit; it never paints visible source ink.
            field = _visible_surface(rgba[:,:,:3],support & (rgba[:,:,3]>0))
            if field is not None:
                rgba[:,:,:3][missing] = field[missing]
                rgba[:,:,3][missing] = 255
                element['metadata']['finalCarrierSurfacePixels'] = int(missing.sum())
            repair = {}
            before_completion = rgba[:,:,3] == 0
            element['metadata']['localCarrierSurfacePixels'] = complete_local_floor(
                rgba, support, keep_floor, sample_mask=owned & ~glyph_substrate, diagnostics=repair)
            completion_path = mask_dir / f"{element['id']}_completion.png"
            completion_mask = np.zeros_like(rgba)
            completion_mask[:,:,3] = (before_completion & (rgba[:,:,3]>0)).astype(np.uint8)*255
            _save_exact(completion_path,completion_mask)
            repair['mask'] = str(completion_path.relative_to(output)).replace('\\','/')
            element['metadata']['localCarrierRepair'] = repair
            element['metadata']['unresolvedChildSurfacePixels'] = int((support & (rgba[:,:,3]==0) & keep_floor).sum())
            element['metadata']['carrierInteriorHolePixels'] = int((support & (rgba[:,:,3]==0)).sum())
        _save_exact(path,rgba)
    by_id = {e["id"]: e for e in elements}
    for node in nodes:
        node["zIndex"] = by_id[node["id"]].get("zIndex", 0)
        node['parent'] = by_id[node['id']].get('parent')
    for decision in text_decisions:
        box = decision["bbox"]
        if box:
            x1, y1, x2, y2 = box
            decision["replacementOwners"] = [nodes[i-1]["id"] for i in np.unique(labels[y1:y2, x1:x2]) if i > 0]
    graph = {"version": "3.0-exclusive", "policy": "exactly_one_source_owner", "nodes": nodes,
             "textDecisions": text_decisions, "unownedPixelCount": int((labels == 0).sum()),
             "sourcePixelCount": w*h, "automaticRevisionsPaused": True,
             "inpaintCalls":0,
             "contourPlanning":contour_audit,
             "repairScope":"glyph-local measured paper, after persisted owner; never a whole module"}
    graph['objectIntegrity'] = {
        'unresolvedCarrierIds':[e['id'] for e in visuals if e['metadata'].get('carrierSupportMask')
                                and e['metadata'].get('carrierInteriorHolePixels',0)>0],
        'unverifiedCarrierIds':[e['id'] for e in visuals if e['metadata'].get('carrierSupportMask')
                                and not e['metadata'].get('carrierBoundaryVerified')],
        'unrebuiltCarrierIds':[e['id'] for e in visuals if e['metadata'].get('carrierSupportMask')
                               and not e['metadata'].get('carrierSurfaceRebuilt')],
        'photographicObjectCount':sum(e['metadata'].get('visualAssetPolicy')=='source_verified_photographic_rectangle' for e in visuals),
        'scope':'all carrier interiors, boundary evidence and rebuild coverage; zero holes alone is not visual acceptance',
    }
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
        if (item.get("metadata") or {}).get("nativeTextMeasured"):
            # Dense small glyphs bias the old bbox-median estimate. Its dilated
            # mask can include unchanged PAPER pixels, falsely reporting ghosts.
            # Independently remeasure surrounding source paper, not the claimed
            # deletion mask, so genuinely unremoved ink still fails this check.
            from app.services.reconstruction.editable_text import measured_surface
            ax, ay, bx, by = max(0, x1-6), max(0, y1-6), min(w, x2+6), min(h, y2+6)
            measured = measured_surface(source[ay:by, ax:bx])
            if measured is not None:
                paper, _ = measured
                local_paper = paper[y1-ay:y2-ay, x1-ax:x2-ax]
                mask &= np.max(np.abs(patch.astype(float)-local_paper.astype(float)), axis=2) >= 8
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
