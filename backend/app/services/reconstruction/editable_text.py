"""Restore reliable OCR text without requiring an identical raster font.

Ownership proof is OCR + a separable glyph mask + locally measured paper,
not pixel equality with a particular installed font. Font matching ranks the
best editable substitute; it must not turn all normal text into screenshots.
"""
from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np

from app.services.typography.font_raster import font_face, glyph_raster


def measured_surface(patch: np.ndarray, border: int = 4, *, extrapolate: int = 0) -> tuple[np.ndarray, dict] | None:
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
    # Start from the dominant paper instead of fitting neighbouring glyphs /
    # underline colours as if they were background samples.
    distance = np.max(np.abs(pixels-np.median(pixels,axis=0)),axis=1)
    keep = distance <= max(12.,float(np.quantile(distance,.7)))
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
    if extrapolate:
        # Evaluate the SAME fitted field outside a small sampling inset. Do not
        # resize the surface: that would move the gradient underneath the ink.
        ey,ex = np.mgrid[-extrapolate:h+extrapolate, -extrapolate:w+extrapolate]
        ex,ey = ex*2/(w-1)-1,ey*2/(h-1)-1
        terms = np.stack((np.ones_like(ex),ex,ey,ex*ey,ex*ex,ey*ey),axis=-1)
    surface = np.clip(np.rint(terms@coeff),0,255).astype(np.uint8)
    return surface,{"backgroundError95":round(error95,3),"backgroundInlierRatio":round(inliers,3),"backgroundModel":"robust_quadratic_colour_field"}


def measured_text_surface(patch: np.ndarray) -> tuple[np.ndarray, dict] | None:
    """Use nearby paper when the outer OCR halo crosses a card edge/underline.

    Callers supply the existing six-pixel padded OCR crop. Only the sampling
    ring changes; the repair still requires verified glyphs and a font owner.
    Keep the stricter generic surface estimator unchanged for visual assets.
    """
    model = measured_surface(patch)
    if model is not None:
        return model
    for inset in (4, 3, 2):
        if min(patch.shape[:2])-2*inset < 11:
            continue
        model = measured_surface(patch[inset:-inset,inset:-inset],border=2,extrapolate=inset)
        if model is not None:
            paper,evidence = model
            return paper,{**evidence,"backgroundSamplingInset":inset,
                          "backgroundSamplingPolicy":"local_ocr_ring_same_quality_gates"}
    return None


def without_untranscribed_bullet(ink: np.ndarray, text: str, height: float) -> tuple[np.ndarray, bool]:
    """Keep a leading list marker out of an OCR body that did not transcribe it.

    Measure projection islands, not individual connected strokes. The first
    Chinese glyph may have disconnected upper/lower strokes; treating only a
    tall component as the body accidentally groups that glyph with the bullet.
    """
    if not text or not text[0].isalnum():
        return ink,False
    used = ink.any(axis=0)
    starts = np.flatnonzero(used & ~np.r_[False,used[:-1]])
    ends = np.flatnonzero(used & ~np.r_[used[1:],False])+1
    if len(starts)<2:
        return ink,False
    left,right = int(starts[0]),int(ends[0])
    yy,_ = np.where(ink[:,left:right])
    top,bottom = int(yy.min()),int(yy.max()+1)
    cw,ch = right-left,bottom-top
    next_yy,_ = np.where(ink[:,starts[1]:ends[1]])
    next_height = int(np.ptp(next_yy)+1)
    next_center = float((next_yy.min()+next_yy.max()+1)/2)
    if (.2*height<ch<.55*height and .65<cw/max(1,ch)<1.4
            and ink[:,left:right].sum()/max(1,cw*ch)>.25
            and starts[1]-right>height*.2 and next_height>ch*1.6
            and abs((top+bottom)/2-next_center)<height*.22):
        body = ink.copy()
        body[:,left:right] = False
        return body,True
    return ink,False


def _fit_font(text: str, mask: np.ndarray, preferred: str, source_alpha: np.ndarray | None=None) -> dict | None:
    ys,xs = np.where(mask)
    th,tw = int(ys.max()-ys.min()+1),int(xs.max()-xs.min()+1)
    target = mask[ys.min():ys.max()+1,xs.min():xs.max()+1]
    soft_target = np.clip(source_alpha[ys.min():ys.max()+1,xs.min():xs.max()+1],0,1) if source_alpha is not None else target.astype(float)
    choices = [("SimSun","simsun.ttc",400),("SimSun","simsun.ttc",700),
               ("Microsoft YaHei","msyh.ttc",400),("Microsoft YaHei","msyhbd.ttc",700),
               ("DengXian","Deng.ttf",400),("DengXian","Dengb.ttf",700),
               ("Noto Sans SC","NotoSansSC-VF.ttf",400),("Noto Sans SC","NotoSansSC-VF.ttf",700),
               ("Noto Serif SC","NotoSerifSC-VF.ttf",400),("Noto Serif SC","NotoSerifSC-VF.ttf",700),
               ("SimHei","simhei.ttf",400),("FangSong","simfang.ttf",400),
               ("Arial","arial.ttf",400),("Arial","arialbd.ttf",700),
               ("Times New Roman","times.ttf",400),("Times New Roman","timesbd.ttf",700),
               ("KaiTi","simkai.ttf",400)]
    choices.sort(key=lambda item: item[0] != preferred)
    best = None
    for family,filename,weight in choices:
        if not (Path("C:/Windows/Fonts") / filename).is_file():
            continue
        # Latin-only files otherwise score their missing-CJK square glyphs.
        if any('\u2e80' <= c <= '\u9fff' for c in text) and family in ('Arial','Times New Roman'):
            continue
        import unicodedata
        punctuation=all(c.isspace() or unicodedata.category(c).startswith(('P','S')) for c in text)
        for size in range(max(8,round(th*.85)),min(150,round(th*(4.5 if punctuation else 1.5)))+1):
            font = font_face(filename,size,weight)
            bbox = font.getbbox(text)
            fw,fh = bbox[2]-bbox[0],bbox[3]-bbox[1]
            # Font bbox width includes advance whitespace (especially Chinese
            # punctuation). Compare real ink below, not that whitespace.
            if not fw or not fh or abs(fh-th)/max(1,th) > .3 or (not punctuation and abs(fw-tw)/max(1,tw) > .25):
                continue
            stroke = size/48 if filename=='simsun.ttc' and weight==700 else 0
            raster,offset = glyph_raster(text,filename,size,stroke,weight=weight)
            spacing=0.
            if len(text)>1:
                spacing=(tw-raster.shape[1])/(len(text)-1)
                # Do not squeeze a wide substitute until adjacent glyphs collide.
                # Positive tracking is common in headings; negative tracking
                # needs a much tighter limit and is never a width-scale proxy.
                if spacing < -size*.055 or spacing > size*.15:
                    continue
                if abs(spacing)>.15:
                    raster,offset=glyph_raster(text,filename,size,stroke,spacing,weight)
                else:
                    spacing=0.
            cropped = raster>127
            # Score the physical glyphs that will actually be exported. Resizing
            # every candidate to the target used to reward undersized fonts and
            # then export their *unscaled* glyphs, visibly changing the heading.
            ch,cw = raster.shape
            canvas_h,canvas_w = max(th,ch),max(tw,cw)
            normalized_alpha = np.zeros((canvas_h,canvas_w),float)
            normalized_alpha[:ch,:cw] = raster.astype(float)/255
            actual_target = np.zeros_like(normalized_alpha)
            actual_target[:th,:tw] = soft_target
            normalized = normalized_alpha>.5
            target_core = actual_target>.5
            union = np.count_nonzero(target_core|normalized)
            similarity = np.count_nonzero(target_core&normalized)/max(1,union)
            if similarity < .35:
                continue
            dimension_error = abs(cropped.shape[1]-tw)/tw+abs(cropped.shape[0]-th)/th
            if dimension_error>.19:
                continue
            # Binary low-threshold ink rewards fat strokes. Compare calibrated
            # antialias alpha and ink density too, so normal serif is not silently
            # changed into YaHei Bold merely for a few extra matching pixels.
            alpha_error = float(np.mean(np.abs(actual_target-normalized_alpha)))
            density_error = abs(float(target_core.mean())-float(normalized.mean()))
            rank = similarity-alpha_error*.7-density_error*1.2-dimension_error*.7-abs(min(0,spacing))/size*.3
            if best is None or rank > best[0]:
                best = (rank,{"fontFamily":family,"fontFile":filename,"fontSize":size,"fontWeight":weight,
                              "fontStrokeWidth":stroke,"letterSpacing":round(spacing,4),"fontAlphaError":round(alpha_error,4),"fontDensityError":round(density_error,4),
                              "fontMaskSimilarity":round(float(similarity),4),"fontDimensionError":round(float(dimension_error),4),
                              "fontInkOffset":list(offset),
                              "fontInkSize":[int(cropped.shape[1]),int(cropped.shape[0])]})
    if best is None:
        return None
    # A measured layout match is required. Identical glyph rasterization is not.
    return best[1] if best[1]["fontDimensionError"] <= .19 and best[1]["fontMaskSimilarity"] >= .35 else None


def exclude_isolated_crop_fragments(seed: np.ndarray, text: str, height: float) -> np.ndarray:
    """Exclude tiny neighbouring ink from fitting, not from the source image.

    An OCR rectangle may include a one-pixel edge of an untranscribed bullet,
    or compression speckles above the baseline. Neither defines the font size.
    Punctuation/i dots within the body band remain untouched.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
    substantial = [s for s in stats[1:] if s[4] > max(6, height * .35) and s[3] > height * .4]
    if len(substantial) < 3:
        return seed
    top = float(np.median([s[1] for s in substantial]))
    bottom = float(np.median([s[1] + s[3] for s in substantial]))
    left = min(s[0] for s in substantial)
    cleaned = seed.copy()
    for number in range(1, count):
        x, y, w, h, area = stats[number]
        isolated = y + h <= top - 3 or y >= bottom + 3
        clipped_marker = bool(text and text[0].isalnum() and w <= 2 and h < height * .3 and left - (x + w) > height * .3)
        if area <= 3 and (isolated or clipped_marker):
            cleaned[labels == number] = False
    return cleaned


def restore_text_candidate(source: np.ndarray, item: dict, hints: dict | None = None,
                           diagnostics: dict | None = None, *,
                           paper_reference: tuple[int,int,np.ndarray,dict] | None = None) -> tuple[dict,np.ndarray,np.ndarray] | None:
    report = diagnostics if diagnostics is not None else {}
    report["reason"] = "unreliable_ocr_or_artwork"
    text = str(item.get("text") or "")
    semantic = str((hints or {}).get("componentType") or (hints or {}).get("role") or "").lower()
    confidence = float(item.get("confidence") or 0)
    if not text.strip() or "\n" in text or confidence<.85 or any(t in semantic for t in ("logo","wordmark","artistic")) or "^{" in text or "_{" in text:
        return None
    h,w = source.shape[:2]
    x,y,iw,ih = (float(item.get(k,0)) for k in ("x","y","width","height"))
    if not all(np.isfinite(v) for v in (x,y,iw,ih)) or min(iw,ih)<3:
        return None
    pad = 6
    x1,y1,x2,y2 = max(0,round(x)-pad),max(0,round(y)-pad),min(w,round(x+iw)+pad),min(h,round(y+ih)+pad)
    patch = source[y1:y2,x1:x2]
    report["reason"] = "complex_or_unmeasured_paper"
    model = None
    if paper_reference is not None:
        px,py,parent_paper,parent_evidence = paper_reference
        if x1>=px and y1>=py and x2<=px+parent_paper.shape[1] and y2<=py+parent_paper.shape[0]:
            model = (parent_paper[y1-py:y2-py,x1-px:x2-px],
                     {**parent_evidence,"sharedOcrLinePaper":True})
    if model is None:
        model = measured_text_surface(patch)
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
    magnitude = np.sum(difference*difference,axis=2)
    cosine = np.sum(difference*direction,axis=2)/np.maximum(1,np.sqrt(magnitude*length))
    # OCR supplies an approximate colour for the entire line. A differently
    # coloured emphasis run must not estimate its fill from pale AA fringes
    # just because the true solid strokes differ by >30 RGB levels.
    strong = core & (alpha>.35) & (cosine>.94) & (length>40**2)
    if np.count_nonzero(strong)<8:
        report["reason"] = "no_separable_glyphs"
        return None
    fg = np.median(pixels[strong & (magnitude>=np.quantile(magnitude[strong],.65))],axis=0)
    direction = fg-paper.astype(float)
    alpha = np.sum(difference*direction,axis=2)/np.maximum(1,np.sum(direction*direction,axis=2))
    alignment = np.max(np.abs(pixels-(paper+alpha[:,:,None]*direction)),axis=2)
    contrast = np.max(np.abs(difference),axis=2)
    seed = core & (alpha>.16) & (contrast>max(10,evidence["backgroundError95"]*2)) & (alignment<22)
    seed,excluded_bullet = without_untranscribed_bullet(seed,text,ih)
    seed = exclude_isolated_crop_fragments(seed, text, ih)
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
    # A gold/gradient glyph can change colour along its height. Matching only
    # the median foreground leaves its darker connected feet in the old image.
    # Grow the ERASURE support within the authoritative OCR box, using only
    # contrast components attached to verified glyphs. Font fitting is unchanged.
    agreement = np.sum(difference*direction,axis=2)/np.maximum(1,
        np.sqrt(np.sum(difference*difference,axis=2)*np.sum(direction*direction,axis=2)))
    support = core & (contrast>max(10,evidence['backgroundError95']*3)) & (agreement>.80) & (alpha>.12)
    count,parts,stats,_ = cv2.connectedComponentsWithStats(support.astype(np.uint8),connectivity=8)
    complete_glyphs = glyphs.copy()
    for number in range(1,count):
        _,_,cw,ch,area = stats[number]
        component = parts==number
        overlap = np.count_nonzero(component & glyphs)
        if overlap>=max(3,area*.35) and not (cw>iw*.65 and ch<max(3,ih*.15)):
            complete_glyphs |= component
    if complete_glyphs.sum()>glyphs.sum()*1.75:
        complete_glyphs = glyphs
    evidence['connectedGlyphColourPixels'] = int((complete_glyphs & ~glyphs).sum())
    neighbors = cv2.dilate(complete_glyphs.astype(np.uint8),np.ones((7,7),np.uint8))>0
    # Includes source antialias fringes; changes only glyph-local pixels.
    # JPEG ringing can have the OPPOSITE polarity to the foreground. Filtering
    # by positive foreground alpha leaves white outlines around black glyphs
    # and dark outlines around white glyphs. Include their narrow local halo;
    # never authorize the surrounding card/bbox as an erasure region.
    ink = complete_glyphs | (neighbors & (contrast>.5))
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
    fit = _fit_font(text,glyphs,item.get("style",{}).get("fontFamily","Microsoft YaHei"),np.where(core,alpha,0))
    if fit is None:
        return None
    if confidence < .9 and (fit["fontMaskSimilarity"] < .5 or fit["fontAlphaError"] > .32):
        report["reason"] = "borderline_ocr_without_visual_confirmation"
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
                        "fontWeight":fit["fontWeight"],"fontStrokeWidth":fit["fontStrokeWidth"],"letterSpacing":fit["letterSpacing"],"color":"#"+"".join(f"{v:02X}" for v in fg[::-1]),"align":"left","verticalAlign":"top"}
    element["metadata"] = {"reconstructionStrategy":"editable_text","exclusiveFontVerified":True,
                           "nativeTextMeasured":True,"fontMatchingVersion":2,"fontMeasurement":fit,"textErasureEvidence":evidence,
                           "rawOCRBBox":[round(x),round(y),round(x+iw),round(y+ih)],"originalLineCount":1}
    if confidence < .9:
        element["metadata"].update(ocrNeedsReview=True, ocrConfidence=confidence,
            ocrAcceptance="borderline_confidence_with_measured_glyph_confirmation")
    if excluded_bullet:
        element['metadata']['originalOCRBBox'] = element['metadata']['rawOCRBBox'][:]
        element['metadata']['rawOCRBBox'][0] = int(x1+xs.min())
        element['metadata']['excludedLeadingBullet'] = True
    mask = np.zeros((h,w),bool)
    mask[y1:y2,x1:x2] = ink
    fill = source.copy()
    fill[y1:y2,x1:x2] = paper
    report["reason"] = "ocr_glyph_owner_with_measured_paper_and_font_layout"
    report.update({"glyphPixelCount":int(ink.sum()),"fontMaskSimilarity":fit["fontMaskSimilarity"]})
    return element,mask,fill
