"""Coherent photos/carriers, or measured antialiased mattes -- not hard jigsaws."""
from __future__ import annotations

import cv2
import numpy as np

from app.services.reconstruction.editable_text import measured_surface


def refine_edge_alpha(rgb, paper, alpha, contrast):
    """Unmix a narrow exterior fringe using nearby stronger foreground colour.

    Geometry alone cannot measure source antialias opacity. Only accept a
    lower opacity if recomposition onto the measured paper still matches the
    source. Interior white artwork and uncertain edges remain unchanged.
    """
    distance = cv2.distanceTransform(np.uint8(alpha>0),cv2.DIST_L2,3)
    eligible = (alpha>0) & (distance<=2) & (contrast>4)
    best = contrast.copy()
    reference = rgb.astype(float).copy()
    for dy in range(-2,3):
        for dx in range(-2,3):
            shifted = np.roll(rgb,(dy,dx),(0,1)).astype(float)
            strength = np.roll(contrast,(dy,dx),(0,1))
            valid = eligible & (strength>best) & (np.roll(alpha,(dy,dx),(0,1))>=.99)
            if dy>0: valid[:dy]=False
            if dy<0: valid[dy:]=False
            if dx>0: valid[:,:dx]=False
            if dx<0: valid[:,dx:]=False
            reference[valid]=shifted[valid]
            best[valid]=strength[valid]
    direction = reference-paper
    fitted = np.sum((rgb-paper)*direction,axis=2)/np.maximum(1,np.sum(direction*direction,axis=2))
    prediction = paper+fitted[:,:,None]*direction
    accept = eligible & (best>contrast*1.4) & (fitted>.08) & (fitted<alpha-.03)
    accept &= np.max(np.abs(prediction-rgb),axis=2)<=3
    alpha=alpha.copy()
    alpha[accept]=fitted[accept]
    return alpha,int(accept.sum())


def prepare_visual_seed(image: np.ndarray, box: tuple[int,int,int,int], raw: np.ndarray) -> tuple[tuple[int,int,int,int],np.ndarray,dict]:
    x1,y1,x2,y2 = box
    binary = raw > 127
    density = float(binary.mean())
    # Dense rectangular photos, chart cards and carrier plates are coherent
    # crops. Their internal mask pinholes must not become jagged missing bits.
    # An all-opaque proposal is often only a detector bbox. Density alone says
    # nothing about the actual silhouette (a rounded card is also >82% full).
    # Only a mask with enclosed tiny defects AND four complete boundary sides
    # gives independent evidence of an intended rectangular raster surface.
    boundary = np.concatenate((binary[0],binary[-1],binary[:,0],binary[:,-1]))
    if .98 <= density < 1 and boundary.all():
        rgba = cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2BGRA)
        return box,rgba,{"visualAssetPolicy":"coherent_rectangular_crop","seedMaskDensity":round(density,4)}
    h,w = image.shape[:2]
    pad = 6
    ax,ay,bx,by = max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad)
    patch = image[ay:by,ax:bx]
    evidence = measured_surface(patch)
    # The first context ring can cross an adjacent title or contain the AA
    # fringe itself. Try a few bounded rings, never a whole-module whitening.
    if evidence is None:
        for pad in (3, 10, 16):
            ax,ay,bx,by = max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad)
            patch = image[ay:by,ax:bx]
            evidence = measured_surface(patch, border=2)
            if evidence is not None:
                break
    if evidence is None:
        # An unreliable boundary is not permission to chew up an illustration.
        # Retain a coherent crop (with its real local paper) instead of a fake
        # transparent cutout. The policy is recorded for honest QA.
        # A contour proposal can be merely a thin open Canny edge, not a whole
        # object. Preserve its complete source window when paper is unknowable;
        # record the limitation rather than labelling that crop a good matte.
        rgba = cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2BGRA)
        return box,rgba,{"visualAssetPolicy":"coherent_crop_unreliable_matte","seedMaskDensity":round(density,4),
                         "matteNeedsReview":True}
    paper,model = evidence
    mask = np.zeros(patch.shape[:2],np.uint8)
    mask[y1-ay:y2-ay,x1-ax:x2-ax] = binary.astype(np.uint8)*255
    # SAM is a proposal, not the authority for deleting pale appendages or
    # making interior pinholes. Recover source-contrast components attached to
    # the seed, including a small context ring around its predicted bbox.
    contrast = np.max(np.abs(patch.astype(float)-paper.astype(float)),axis=2)
    foreground = contrast > max(5.,model['backgroundError95']*3)
    count,parts,stats,_ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8),8)
    recovered = np.zeros_like(foreground)
    for number in range(1,count):
        component = parts == number
        overlap = np.count_nonzero(component & (mask>0))
        if overlap >= max(2,stats[number,cv2.CC_STAT_AREA]*.05):
            recovered |= component
    added = recovered & (mask==0)
    # A neighbouring connected module is not evidence for expanding this icon.
    # Retain the source crop if the component is unbounded/ambiguous.
    if recovered.sum() > max(binary.sum()*2.5, binary.size*1.15):
        rgba = cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2BGRA)
        return box,rgba,{'visualAssetPolicy':'coherent_crop_ambiguous_component',
                         'seedMaskDensity':round(density,4),'matteNeedsReview':True}
    # Use source boundaries rather than the dilated Canny/SAM bbox boundary.
    # Flood only exterior paper; interior white marks remain part of the object.
    barrier = cv2.morphologyEx(recovered.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    _, spaces = cv2.connectedComponents(1-barrier, connectivity=4)
    exterior_ids = np.unique(np.concatenate((spaces[0],spaces[-1],spaces[:,0],spaces[:,-1])))
    exterior = np.isin(spaces,exterior_ids[exterior_ids>0])
    silhouette = ~exterior
    if not silhouette.any():
        return box,cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2BGRA),{'visualAssetPolicy':'coherent_crop_empty_matte','matteNeedsReview':True}
    mask = silhouette.astype(np.uint8)*255
    # Contour simplification at subpixel scale followed by supersampling. Large
    # holes remain holes; closed tiny mask defects are not fragmented assets.
    contours,hierarchy = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    scale = 4
    hi = np.zeros((mask.shape[0]*scale,mask.shape[1]*scale),np.uint8)
    if hierarchy is not None:
        for index,contour in enumerate(contours):
            simplified = cv2.approxPolyDP(contour,.6,True)
            color = 255 if hierarchy[0,index,3] == -1 else 0
            cv2.drawContours(hi,[simplified*scale],-1,color,-1)
    alpha = cv2.resize(hi,(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_AREA).astype(float)/255
    # Remove the old paper colour from feather pixels. Simply softening alpha
    # while keeping source RGB creates white/pink fringes when moved.
    rgb = patch.astype(float)
    alpha,unmixed_edges = refine_edge_alpha(rgb,paper.astype(float),alpha,contrast)
    band = (alpha>0) & (alpha<1)
    foreground = rgb.copy()
    foreground[band] = (rgb[band]-(1-alpha[band,None])*paper[band])/np.maximum(.05,alpha[band,None])
    foreground = np.clip(np.rint(foreground),0,255).astype(np.uint8)
    # Validate the matte against the actual source before assigning ownership.
    # A smoothed contour must not make dark/coloured source details disappear.
    composite = foreground.astype(float)*alpha[:,:,None]+paper*(1-alpha[:,:,None])
    error = np.max(np.abs(composite-patch),axis=2)
    unsafe = (recovered | (mask>0)) & (error>8)
    alpha[unsafe] = 1
    foreground[unsafe] = patch[unsafe]
    # Supersampling can extend half a pixel onto pure surrounding paper. Do not
    # bake that paper into a translucent white fringe on a movable object.
    paper_only = (alpha>0) & (alpha<1) & (contrast<=max(3.,model['backgroundError95']*1.5))
    alpha[paper_only] = 0
    # Source-like pale details are kept; only pixels outside the proposed object
    # are left for another owner. No original content is erased here.
    rgba = np.dstack((foreground,np.uint8(np.rint(alpha*255))))
    yy,xx = np.where(rgba[:,:,3]>0)
    if not len(xx):
        return box,cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2BGRA),{"visualAssetPolicy":"coherent_crop_empty_matte"}
    cx,cy,dx,dy = xx.min(),yy.min(),xx.max()+1,yy.max()+1
    return (ax+cx,ay+cy,ax+dx,ay+dy),rgba[cy:dy,cx:dx],{
        "visualAssetPolicy":"supersampled_colour_decontaminated_matte","seedMaskDensity":round(density,4),
        "featherPixelCount":int(band.sum()),"matteBackgroundEvidence":model,
        "sourceMeasuredEdgeAlphaPixels":unmixed_edges,
        "recoveredVisualPixelCount":int(added.sum()),"unsafeFeatherPixelsPreserved":int(unsafe.sum()),
        "matteValidation":"source_contrast_components_and_local_composite"}
