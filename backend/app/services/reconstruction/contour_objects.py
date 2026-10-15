"""Resolve detector proposals into coherent silhouettes before assigning owners.

Boxes are search windows. Source contours supply the mask; nested contour
duplicates are proposals for one object, not permission to cut it into rings.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.reconstruction.editable_text import measured_surface
from app.services.reconstruction.visual_assets import prepare_visual_seed
from app.utils import image_io


def overlap(a, b):
    x1,y1,x2,y2 = max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    return max(0,x2-x1)*max(0,y2-y1)


def area(box):
    return (box[2]-box[0])*(box[3]-box[1])


def carrier_contours(image: np.ndarray):
    """Closed, convex panel boundaries from pixels, including pale rounded cards."""
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Max across channels retains pale cyan/pink panel borders grayscale misses.
    edges = np.maximum.reduce([cv2.Canny(image[:,:,c],12,36) for c in range(3)])
    edges = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_ = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    h,w = gray.shape
    candidates = []
    for contour in contours:
        x,y,cw,ch = cv2.boundingRect(contour)
        size = cw*ch
        actual = cv2.contourArea(contour)
        if min(cw,ch)<24 or size<w*h*.003 or size>w*h*.75 or actual<size*.88:
            continue
        hull = cv2.contourArea(cv2.convexHull(contour))
        if actual/max(1,hull)<.97:
            continue
        # A framed photograph is already a coherent visual, not a paper panel.
        patch = image[y:y+ch,x:x+cw]
        gradient = np.max(np.abs(patch.astype(float)-cv2.GaussianBlur(patch,(5,5),0)),axis=2)
        if float(np.mean(gradient>12))>.20:
            continue
        mask = np.zeros((ch,cw),np.uint8)
        cv2.drawContours(mask,[contour-np.array([[[x,y]]])],-1,255,-1)
        candidates.append(((x,y,x+cw,y+ch),mask))
    return sorted(candidates,key=lambda p: area(p[0]),reverse=True)


def plan_contour_objects(image: np.ndarray, segments: list[dict], assets: Path,
                         text_decisions: list[dict], box_of) -> tuple[list[dict], dict]:
    h,w = image.shape[:2]
    proposals = []
    for segment in segments:
        box = box_of(segment,w,h)
        path = assets/str(segment.get('alphaCrop') or '').replace('\\','/').split('/')[-1]
        asset = image_io.imread(path,cv2.IMREAD_UNCHANGED) if path.is_file() else None
        if not box or asset is None or asset.ndim!=3 or asset.shape[2]!=4:
            continue
        x,y,bx,by = box
        if asset.shape[:2]!=(by-y,bx-x):
            continue
        proposals.append({'id':segment.get('id'),'box':box,'mask':asset[:,:,3],
                          'source':segment.get('source','opencv_segment'),'carrier':False})
    for index,(box,mask) in enumerate(carrier_contours(image)):
        proposals.append({'id':f'closed_carrier_{index}','box':box,'mask':mask,
                          'source':'source_closed_carrier','carrier':True})
    # Prefer the full outer boundary over the inner edge of the same border.
    # Substantially smaller objects (e.g. an icon inside a card) are kept.
    proposals.sort(key=lambda p: area(p['box']),reverse=True)
    unique = []
    duplicates = 0
    for proposal in proposals:
        box = proposal['box']
        near = next((p for p in unique if overlap(box,p['box'])/area(box)>.93 and area(box)/area(p['box'])>.68),None)
        if near:
            # Give the strongest enclosing contour the carrier semantics too.
            near['carrier'] |= proposal['carrier']
            duplicates += 1
            continue
        unique.append(proposal)
    result = []
    crossing = 0
    for proposal in unique:
        box = proposal['box']
        # Never split one raster OCR line among unrelated cutouts. Native lines
        # already have verified glyph owners and are cleaned before this step.
        raster = [d['bbox'] for d in text_decisions if d['decision']=='movable_image' and d.get('bbox')]
        if any(.03 < overlap(box,b)/area(b) < .97 for b in raster):
            crossing += 1
            continue
        refined = prepare_visual_seed(image,box,proposal['mask'])
        refined_box,rgba,evidence = refined
        # An enclosing compact visual owns its internal contour proposals:
        # photos in a stack or line-chart curves remain whole, movable objects.
        # Large panels instead allow independent children above their surface.
        children = [p for p in unique if p is not proposal and area(p['box'])<area(box)*.55
                    and overlap(box,p['box'])/area(p['box'])>.95]
        compact = area(box)<w*h*.035
        is_panel = proposal['carrier'] and (not compact or bool(children))
        if not is_panel:
            parent = next((p for p in result if not p['panel'] and overlap(box,p['originalBox'])/area(box)>.93),None)
            if parent:
                parent['evidence'].setdefault('absorbedContourProposals',[]).append(proposal['id'])
                continue
        evidence = {**evidence,'contourProposalId':proposal['id'],
                    'boundaryAuthority':'source_pixels_not_vlm_bbox'}
        result.append({'box':refined_box,'originalBox':box,'rgba':rgba,'evidence':evidence,
                       'source':proposal['source'],'panel':is_panel})
    result.sort(key=lambda p:(p['panel'],area(p['box']) if p['panel'] else -area(p['box'])))
    return result,{'inputProposals':len(proposals),'duplicateContoursMerged':duplicates,
                   'rasterTextCrossingProposalsRejected':crossing,'plannedObjects':len(result)}


def complete_carrier_holes(cleaned: np.ndarray, rgba: np.ndarray, box, support: np.ndarray,
                           labels: np.ndarray, ink: np.ndarray) -> tuple[np.ndarray,dict]:
    """Fill only verified child-owned holes with independently measured paper.

    Visible original pixels are never changed. This synthesized substrate has
    no claim on source content; its child must render above it. Unknown texture
    remains transparent and is explicitly counted, never fabricated by LaMa.
    """
    x,y,bx,by = box
    owned = labels[y:by,x:bx]>0
    holes = support & owned & ~ink[y:by,x:bx]
    count,parts,stats,_ = cv2.connectedComponentsWithStats(holes.astype(np.uint8),connectivity=8)
    completed = np.zeros(holes.shape,bool)
    for number in range(1,count):
        lx,ly,cw,ch,size = stats[number]
        if size<4:
            continue
        pad = 12
        ax,ay,rx,ry = max(0,x+lx-pad),max(0,y+ly-pad),min(cleaned.shape[1],x+lx+cw+pad),min(cleaned.shape[0],y+ly+ch+pad)
        sample = cleaned[ay:ry,ax:rx]
        local_hole = parts==number
        # Require agreement with the parent's own surrounding pixels too.
        # Skip the immediate antialias/shadow fringe of the child. Those pixels
        # are not clean paper samples even though the detector left them out.
        ring = cv2.dilate(local_hole.astype(np.uint8),np.ones((25,25),np.uint8))>0
        ring &= cv2.dilate(local_hole.astype(np.uint8),np.ones((5,5),np.uint8))==0
        ring &= support & ~owned & ~ink[y:by,x:bx]
        ry0,rx0 = np.where(ring)
        if len(rx0)<12:
            continue
        valid = (rx0+x>=ax)&(rx0+x<rx)&(ry0+y>=ay)&(ry0+y<ry)
        rx0,ry0 = rx0[valid],ry0[valid]
        if len(rx0)<12:
            continue
        measured = measured_surface(sample)
        if measured is None:
            measured = _masked_paper(sample,rx0+x-ax,ry0+y-ay)
        if measured is None:
            continue
        paper,model = measured
        error = np.max(np.abs(cleaned[y+ry0,x+rx0].astype(float)-paper[y+ry0-ay,x+rx0-ax]),axis=1)
        if float(np.quantile(error,.95))>max(5.,model['backgroundError95']*2):
            measured = _masked_paper(sample,rx0+x-ax,ry0+y-ay)
            if measured is None:
                continue
            paper,model = measured
        yy,xx = np.where(local_hole)
        rgba[yy,xx,:3] = paper[y+yy-ay,x+xx-ax]
        rgba[yy,xx,3] = 255
        completed |= local_hole
    return completed,{'completedChildSurfacePixels':int(completed.sum()),
                      'unresolvedChildSurfacePixels':int((holes&~completed).sum()),
                      'childSurfacePolicy':'verified_local_paper_below_child_owner'}


def _masked_paper(patch, xs, ys):
    """Fit only visible parent pixels, excluding neighbouring child owners."""
    h,w = patch.shape[:2]
    if len(xs)<24 or np.ptp(xs)<w*.55 or np.ptp(ys)<h*.55:
        return None
    yy,xx = np.mgrid[-1:1:complex(h),-1:1:complex(w)]
    field = np.stack((np.ones_like(xx),xx,yy,xx*yy),axis=-1)
    coords = field[ys,xs]
    pixels = patch[ys,xs].astype(float)
    keep = np.max(np.abs(pixels-np.median(pixels,axis=0)),axis=1)<18
    for _ in range(4):
        if keep.mean()<.80:
            return None
        coeff = np.linalg.lstsq(coords[keep],pixels[keep],rcond=None)[0]
        error = np.max(np.abs(coords@coeff-pixels),axis=1)
        keep = error<=5
    if keep.mean()<.95:
        return None
    return np.clip(np.rint(field@coeff),0,255).astype(np.uint8),{'backgroundError95':float(np.quantile(error,.95))}
