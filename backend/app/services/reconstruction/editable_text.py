"""Restore reliable OCR text without requiring an identical raster font.

Ownership proof is OCR + a separable glyph mask + locally measured paper,
not pixel equality with a particular installed font. Font matching ranks the
best editable substitute; it must not turn all normal text into screenshots.
"""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def measured_surface(patch: np.ndarray, border: int = 4) -> tuple[np.ndarray, dict] | None:
    """Robust low-order colour field from surrounding paper, never a white fill."""
    h,w = patch.shape[:2]
    if min(h,w) < border*2+3:
        return None
    yy,xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    terms = np.stack((np.ones_like(xx),xx,yy,xx*yy,xx*xx,yy*yy),axis=-1)
    ring = np.zeros((h,w),bool)
    ring[:border] = ring[-border:] = True
    ring[:,:border] = ring[:,-border:] = True
    coordinates = terms[ring]
    pixels = patch[ring].astype(float)
    keep = np.ones(len(pixels),bool)
    for _ in range(5):
        if keep.sum() < len(keep)*.55:
            return None
        coeff = np.linalg.lstsq(coordinates[keep],pixels[keep],rcond=None)[0]
        error = np.max(np.abs(coordinates@coeff-pixels),axis=1)
        center = float(np.median(error))
        keep = error <= max(3.,center+3*float(np.median(np.abs(error-center))))
    error95 = float(np.quantile(error[keep],.95))
    inliers = float(keep.mean())
    if error95 > 7 or inliers < .7:
        return None
    surface = np.clip(np.rint(terms@coeff),0,255).astype(np.uint8)
    return surface,{"backgroundError95":round(error95,3),"backgroundInlierRatio":round(inliers,3),"backgroundModel":"robust_quadratic_colour_field"}


@lru_cache(maxsize=512)
def _load_font(filename: str, size: int):
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename),size)


def _fit_font(text: str, mask: np.ndarray, preferred: str) -> dict | None:
    ys,xs = np.where(mask)
    th,tw = int(ys.max()-ys.min()+1),int(xs.max()-xs.min()+1)
    target = mask[ys.min():ys.max()+1,xs.min():xs.max()+1]
    choices = [("SimSun","simsun.ttc",400),("Microsoft YaHei","msyh.ttc",400),
               ("Microsoft YaHei","msyhbd.ttc",700),("Arial","arial.ttf",400),("Arial","arialbd.ttf",700),
               ("KaiTi","simkai.ttf",400)]
    choices.sort(key=lambda item: item[0] != preferred)
    best = None
    for family,filename,weight in choices:
        if not (Path("C:/Windows/Fonts") / filename).is_file():
            continue
        for size in range(max(8,round(th*.85)),min(150,round(th*1.5))+1):
            font = _load_font(filename,size)
            bbox = font.getbbox(text)
            fw,fh = bbox[2]-bbox[0],bbox[3]-bbox[1]
            if not fw or not fh or abs(fw-tw)/max(1,tw) > .15 or abs(fh-th)/max(1,th) > .18:
                continue
            canvas = Image.new("L",(fw+8,fh+8))
            ImageDraw.Draw(canvas).text((4-bbox[0],4-bbox[1]),text,font=font,fill=255)
            raw = np.asarray(canvas)>24
            py,px = np.where(raw)
            if not len(px):
                continue
            cropped = raw[py.min():py.max()+1,px.min():px.max()+1]
            normalized = cv2.resize(cropped.astype(np.uint8),(tw,th),interpolation=cv2.INTER_NEAREST)>0
            union = np.count_nonzero(target|normalized)
            similarity = np.count_nonzero(target&normalized)/max(1,union)
            dimension_error = abs(cropped.shape[1]-tw)/tw+abs(cropped.shape[0]-th)/th
            rank = similarity-dimension_error*.8
            if best is None or rank > best[0]:
                best = (rank,{"fontFamily":family,"fontFile":filename,"fontSize":size,"fontWeight":weight,
                              "fontMaskSimilarity":round(float(similarity),4),"fontDimensionError":round(float(dimension_error),4),
                              "fontInkOffset":[int(bbox[0]+px.min()-4),int(bbox[1]+py.min()-4)],
                              "fontInkSize":[int(cropped.shape[1]),int(cropped.shape[0])]})
    if best is None:
        return None
    # A measured layout match is required. Identical glyph rasterization is not.
    return best[1] if best[1]["fontDimensionError"] <= .19 and best[1]["fontMaskSimilarity"] >= .35 else None


def restore_text_candidate(source: np.ndarray, item: dict, hints: dict | None = None,
                           diagnostics: dict | None = None) -> tuple[dict,np.ndarray,np.ndarray] | None:
    report = diagnostics if diagnostics is not None else {}
    report["reason"] = "unreliable_ocr_or_artwork"
    text = str(item.get("text") or "")
    semantic = str((hints or {}).get("componentType") or (hints or {}).get("role") or "").lower()
    if not text.strip() or "\n" in text or float(item.get("confidence") or 0)<.9 or any(t in semantic for t in ("logo","wordmark","artistic")) or "^{" in text or "_{" in text:
        return None
    h,w = source.shape[:2]
    x,y,iw,ih = (float(item.get(k,0)) for k in ("x","y","width","height"))
    if not all(np.isfinite(v) for v in (x,y,iw,ih)) or min(iw,ih)<7:
        return None
    pad = 6
    x1,y1,x2,y2 = max(0,round(x)-pad),max(0,round(y)-pad),min(w,round(x+iw)+pad),min(h,round(y+ih)+pad)
    patch = source[y1:y2,x1:x2]
    report["reason"] = "complex_or_unmeasured_paper"
    model = measured_surface(patch)
    if model is None:
        return None
    paper,evidence = model
    report.update(evidence)
    try:
        color = np.array(list(bytes.fromhex(item.get("style",{}).get("color","#111111").lstrip("#")))[::-1],float)
    except ValueError:
        return None
    pixels = patch.astype(float)
    difference = pixels-paper.astype(float)
    direction = color-paper.astype(float)
    length = np.sum(direction*direction,axis=2)
    alpha = np.sum(difference*direction,axis=2)/np.maximum(1,length)
    alignment = np.max(np.abs(pixels-(paper+alpha[:,:,None]*direction)),axis=2)
    core = np.zeros(patch.shape[:2],bool)
    core[max(0,round(y)-y1):min(patch.shape[0],round(y+ih)-y1),max(0,round(x)-x1):min(patch.shape[1],round(x+iw)-x1)] = True
    strong = core & (alpha>.35) & (alignment<30) & (length>40**2)
    if np.count_nonzero(strong)<8:
        report["reason"] = "no_separable_glyphs"
        return None
    fg = np.median(pixels[strong & (alpha>=np.quantile(alpha[strong],.65))],axis=0)
    direction = fg-paper.astype(float)
    alpha = np.sum(difference*direction,axis=2)/np.maximum(1,np.sum(direction*direction,axis=2))
    alignment = np.max(np.abs(pixels-(paper+alpha[:,:,None]*direction)),axis=2)
    contrast = np.max(np.abs(difference),axis=2)
    seed = core & (alpha>.16) & (contrast>max(10,evidence["backgroundError95"]*2)) & (alignment<22)
    count,labels,stats,_ = cv2.connectedComponentsWithStats(seed.astype(np.uint8),8)
    glyphs = np.zeros(seed.shape,bool)
    rejected_line = False
    for number in range(1,count):
        _,_,cw,ch,area = stats[number]
        if cw>iw*.65 and ch<max(3,ih*.15):
            rejected_line = True  # Grid/axis rules are not glyphs.
            continue
        if area>0:
            glyphs |= labels==number
    report["reason"] = "intersecting_rule_or_visual"
    if rejected_line or not glyphs.any() or np.mean(glyphs)>.5:
        return None
    neighbors = cv2.dilate(glyphs.astype(np.uint8),np.ones((7,7),np.uint8))>0
    # Includes source antialias fringes; changes only glyph-local pixels.
    # JPEG ringing can have the OPPOSITE polarity to the foreground. Filtering
    # by positive foreground alpha leaves white outlines around black glyphs
    # and dark outlines around white glyphs. Include their narrow local halo;
    # never authorize the surrounding card/bbox as an erasure region.
    ink = glyphs | (neighbors & (contrast>.5))
    # The replacement must cover the full source ink, not merely the pixels
    # which best fit the foreground colour model (JPEG/embossed glyphs vary).
    from app.services.reconstruction.owner_gate import tight_text_mask
    legacy_ink = tight_text_mask(patch,core.astype(np.uint8)*255,item.get("style",{}).get("color"))>0
    ink |= legacy_ink & neighbors
    evidence["glyphHaloRadius"] = 3
    evidence["repairScope"] = "glyphs_and_local_jpeg_ringing_only"
    if ink[:2].any() or ink[-2:].any() or ink[:,:2].any() or ink[:,-2:].any():
        report["reason"] = "glyphs_cross_source_boundary"
        return None
    report["reason"] = "font_layout_does_not_fit"
    fit = _fit_font(text,glyphs,item.get("style",{}).get("fontFamily","Microsoft YaHei"))
    if fit is None:
        return None
    ys,xs = np.where(glyphs)
    offset_x,offset_y = fit["fontInkOffset"]
    element = copy.deepcopy(item)
    tx,ty = max(0,float(x1+xs.min()-offset_x)),max(0,float(y1+ys.min()-offset_y))
    width = min(w-tx,max(float(iw)+4,float(fit["fontInkSize"][0])+4))
    height = min(h-ty,max(float(ih)+6,float(fit["fontSize"])*1.3))
    element.update(type="text",owner="editable_text",editable=True,source="ocr",x=tx,y=ty,width=width,height=height)
    fg = np.clip(np.rint(fg),0,255).astype(np.uint8)
    element["style"] = {**element.get("style",{}),"fontFamily":fit["fontFamily"],"fontFile":fit["fontFile"],"fontSize":fit["fontSize"],
                        "fontWeight":fit["fontWeight"],"color":"#"+"".join(f"{v:02X}" for v in fg[::-1]),"align":"left","verticalAlign":"top"}
    element["metadata"] = {"reconstructionStrategy":"editable_text","exclusiveFontVerified":True,
                           "nativeTextMeasured":True,"fontMeasurement":fit,"textErasureEvidence":evidence,
                           "rawOCRBBox":[round(x),round(y),round(x+iw),round(y+ih)],"originalLineCount":1}
    mask = np.zeros((h,w),bool)
    mask[y1:y2,x1:x2] = ink
    fill = source.copy()
    fill[y1:y2,x1:x2] = paper
    report["reason"] = "ocr_glyph_owner_with_measured_paper_and_font_layout"
    report.update({"glyphPixelCount":int(ink.sum()),"fontMaskSimilarity":fit["fontMaskSimilarity"]})
    return element,mask,fill
