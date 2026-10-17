"""Separate editable glyph changes and antialiased borders from visual loss."""
from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.services.visual_qa.analyzer import _font, render_preview
from app.utils import image_io


def inspect_preservation(source_path: Path, layout: dict, root: Path, preview_path: Path) -> dict:
    source = image_io.imread(source_path)
    h,w = source.shape[:2]
    counts = np.zeros((h,w),np.uint16)
    old_glyphs = np.zeros((h,w),bool)
    paper_claims = np.zeros((h,w),np.uint16)
    paper_colors = np.zeros_like(source)
    for node in layout.get("metadata",{}).get("ownershipAudit",{}).get("nodes",[]):
        mask = image_io.imread(root / node["mask"],-1)
        if mask is None:
            raise ValueError("Source owner mask is missing")
        x1,y1,x2,y2 = node["sourceBBox"]
        counts[y1:y2,x1:x2] += mask[:,:,3]>0
        if node["owner"] == "editable_text":
            old_glyphs[y1:y2,x1:x2] |= mask[:,:,3]>0
    visual_edges = np.zeros((h,w),bool)
    surface_candidates = np.zeros((h,w),bool)
    surface_colors = np.zeros_like(source)
    local = copy.deepcopy(layout)
    for e in local["elements"]:
        metadata = e.get("metadata") or {}
        if metadata.get("paperSubstrateMask"):
            paper_mask = image_io.imread(root / metadata["paperSubstrateMask"],-1)
            if paper_mask is None:
                raise ValueError("Glyph paper substrate mask is missing")
            x1,y1,x2,y2 = metadata["paperSubstrateBBox"]
            paper_claims[y1:y2,x1:x2] += paper_mask[:,:,3]>0
        if not e.get("src"):
            if metadata.get('paperSubstrateMask') and e.get('style',{}).get('fill','').startswith('#'):
                fill = e['style']['fill'].lstrip('#')
                color = [int(fill[i:i+2],16) for i in (4,2,0)]
                paper_colors[y1:y2,x1:x2][paper_mask[:,:,3]>0] = color
            continue
        directory = "backgrounds" if e["type"] == "background" else "assets"
        path = root / directory / e["src"].split("/")[-1]
        e["src"] = str(path)
        if e["type"] != "image":
            continue
        asset = image_io.imread(path,-1)
        if asset is None:
            raise ValueError("Visual asset is missing")
        if metadata.get('paperSubstrateMask'):
            x1,y1,x2,y2 = metadata['paperSubstrateBBox']
            paper_colors[y1:y2,x1:x2][paper_mask[:,:,3]>0] = asset[:,:,:3][paper_mask[:,:,3]>0]
        if metadata.get('surfaceApproximationMask'):
            allowed = image_io.imread(root/metadata['surfaceApproximationMask'],-1)[:,:,3]>0
            x,y = round(e['x']),round(e['y'])
            ah,aw = allowed.shape
            owner_mask = image_io.imread(root/metadata['sourceMaskPath'],-1)[:,:,3]>0
            if allowed.shape != asset.shape[:2] or not np.array_equal(
                    metadata['sourceMaskBBox'],[x,y,x+aw,y+ah]):
                raise ValueError('Surface approximation geometry does not match its source owner')
            error = np.max(np.abs(asset[:,:,:3].astype(float)-source[y:y+ah,x:x+aw]),axis=2)
            if np.any(allowed & (~owner_mask | (asset[:,:,3]!=255) | (error>6))):
                raise ValueError('Surface approximation exceeds bounded paper-only changes')
            surface_candidates[y:y+ah,x:x+aw] |= allowed
            surface_colors[y:y+ah,x:x+aw][allowed] = asset[:,:,:3][allowed]
        if (e.get("metadata") or {}).get("visualAssetPolicy") != "supersampled_colour_decontaminated_matte":
            continue
        alpha = asset[:,:,3]
        band = (alpha>0)&(alpha<255)
        band = cv2.dilate(band.astype(np.uint8),np.ones((5,5),np.uint8))>0
        x,y = round(e["x"]),round(e["y"])
        visual_edges[y:y+band.shape[0],x:x+band.shape[1]] |= band
    new_glyph_canvas = Image.new("RGBA",(w,h))
    draw = ImageDraw.Draw(new_glyph_canvas)
    for e in local["elements"]:
        if e.get("type") == "text":
            if (e.get('metadata') or {}).get('fontMatchingVersion')==2:
                from app.services.typography.font_raster import paint_measured_text
                paint_measured_text(new_glyph_canvas,e,'#FFFFFF')
            else:
                draw.multiline_text((e["x"],e["y"]),str(e.get("text") or ""),font=_font(e.get("style",{})),fill='white')
    new_glyphs = np.asarray(new_glyph_canvas)[:,:,3]>0
    blank = {**local,"elements":[e for e in local["elements"] if e.get("type") != "text"]}
    raster_path = root / "editable_text_removed_preview.png"
    background = root / "backgrounds" / next(e["src"].split("/")[-1] for e in local["elements"] if e["type"] == "background")
    render_preview(background,blank,raster_path)
    raster = image_io.imread(raster_path)
    preview = image_io.imread(preview_path)
    paper_changes = np.max(np.abs(source.astype(float)-raster.astype(float)),axis=2)>1
    visible_changes = np.max(np.abs(source.astype(float)-preview.astype(float)),axis=2)>1
    # An approximation may exempt only its own measured paper, with <=6 channel
    # error, and only when that exact surface is visible in the final raster.
    # It cannot hide a missing/covered foreground object elsewhere in the scene.
    surfaces = surface_candidates & (np.max(np.abs(raster.astype(float)-surface_colors),axis=2)<=1)
    surfaces &= np.max(np.abs(preview.astype(float)-surface_colors),axis=2)<=1
    glyph_overpaint = old_glyphs & (paper_claims==1) & ~visual_edges
    glyph_overpaint &= np.max(np.abs(raster.astype(float)-paper_colors),axis=2)>3
    support = old_glyphs|new_glyphs|visual_edges
    unintended = (paper_changes&~(old_glyphs|visual_edges|surfaces))|(visible_changes&~(support|surfaces))
    unintended |= glyph_overpaint
    return {"unownedSourcePixels":int((counts==0).sum()),"multiplyOwnedSourcePixels":int((counts>1).sum()),
            "unownedGlyphPaperPixels":int((old_glyphs & (paper_claims==0)).sum()),
            "multiplyOwnedGlyphPaperPixels":int((paper_claims>1).sum()),
            "unexpectedVisualChangedPixels":int(unintended.sum()),"nativeSourceGlyphPixels":int(old_glyphs.sum()),
            "boundedCarrierSurfacePixels":int(surfaces.sum()),
            "glyphPaperOverpaintPixels":int(glyph_overpaint.sum()),
            "measuredAntialiasEdgePixels":int(visual_edges.sum()),"previewChangedPixels":int(visible_changes.sum()),
            "rasterWithoutNativeText":str(raster_path),"scope":"source owners + non-text interiors; bounded measured paper, font substitution and verified antialias edges are explicit exceptions"}
