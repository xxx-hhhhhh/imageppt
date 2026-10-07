"""Coherent photos/carriers, or measured antialiased mattes -- not hard jigsaws."""
from __future__ import annotations

import cv2
import numpy as np

from app.services.reconstruction.editable_text import measured_surface


def prepare_visual_seed(image: np.ndarray, box: tuple[int,int,int,int], raw: np.ndarray) -> tuple[tuple[int,int,int,int],np.ndarray,dict]:
    x1,y1,x2,y2 = box
    binary = raw > 127
    density = float(binary.mean())
    # Dense rectangular photos, chart cards and carrier plates are coherent
    # crops. Their internal mask pinholes must not become jagged missing bits.
    if density >= .82:
        rgba = cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2BGRA)
        return box,rgba,{"visualAssetPolicy":"coherent_rectangular_crop","seedMaskDensity":round(density,4)}
    h,w = image.shape[:2]
    pad = 6
    ax,ay,bx,by = max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad)
    patch = image[ay:by,ax:bx]
    evidence = measured_surface(patch)
    if evidence is None:
        # An unreliable boundary is not permission to chew up an illustration.
        # Retain a coherent crop (with its real local paper) instead of a fake
        # transparent cutout. The policy is recorded for honest QA.
        rgba = cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2BGRA)
        return box,rgba,{"visualAssetPolicy":"coherent_crop_unreliable_matte","seedMaskDensity":round(density,4)}
    paper,model = evidence
    mask = np.zeros(patch.shape[:2],np.uint8)
    mask[y1-ay:y2-ay,x1-ax:x2-ax] = binary.astype(np.uint8)*255
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
    band = (alpha>0) & (alpha<1)
    foreground = rgb.copy()
    foreground[band] = (rgb[band]-(1-alpha[band,None])*paper[band])/np.maximum(.05,alpha[band,None])
    foreground = np.clip(np.rint(foreground),0,255).astype(np.uint8)
    # Source-like pale details are kept; only pixels outside the proposed object
    # are left for another owner. No original content is erased here.
    rgba = np.dstack((foreground,np.uint8(np.rint(alpha*255))))
    yy,xx = np.where(rgba[:,:,3]>0)
    if not len(xx):
        return box,cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2BGRA),{"visualAssetPolicy":"coherent_crop_empty_matte"}
    cx,cy,dx,dy = xx.min(),yy.min(),xx.max()+1,yy.max()+1
    return (ax+cx,ay+cy,ax+dx,ay+dy),rgba[cy:dy,cx:dx],{
        "visualAssetPolicy":"supersampled_colour_decontaminated_matte","seedMaskDensity":round(density,4),
        "featherPixelCount":int(band.sum()),"matteBackgroundEvidence":model}
