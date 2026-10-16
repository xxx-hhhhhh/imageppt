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


def photographic_rectangles(image: np.ndarray):
    """Opaque, textured rectangular evidence; thin connecting arrows are not photos.

    These are proposals only. Existing raster-text crossing and overlapping
    stack rules still apply before source ownership is assigned.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = cv2.morphologyEx(np.uint8(gray < 190), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    h, w = gray.shape
    for x, y, cw, ch, pixels in stats[1:]:
        if min(cw, ch) < 40 or max(cw/ch, ch/cw) > 3 or pixels/(cw*ch) < .96 or cw*ch > h*w*.12:
            continue
        patch = image[y:y+ch, x:x+cw]
        texture = np.max(np.abs(patch.astype(float)-cv2.GaussianBlur(patch, (5, 5), 0)), axis=2)
        if np.mean(texture > 12) < .18:
            continue
        yield (max(0, int(x)-1), max(0, int(y)-1), min(w, int(x+cw)+1), min(h, int(y+ch)+1))


def carrier_contours(image: np.ndarray):
    """Closed, convex panel boundaries from pixels, including pale rounded cards."""
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Max across channels retains pale cyan/pink panel borders grayscale misses.
    edges = np.maximum.reduce([cv2.Canny(image[:,:,c],12,36) for c in range(3)])
    edges = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_ = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    # Pale pill edges can be a smooth gradient without a strong Canny edge.
    # Closed chroma regions propose actual curved silhouettes, not OCR boxes.
    chroma = image.max(axis=2).astype(int)-image.min(axis=2)
    for level in (10,20,35):
        regions = np.uint8((chroma>level)&(image.max(axis=2)>140))*255
        extra,_ = cv2.findContours(regions,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(extra)
    h,w = gray.shape
    candidates = []
    for contour in contours:
        x,y,cw,ch = cv2.boundingRect(contour)
        size = cw*ch
        actual = cv2.contourArea(contour)
        if min(cw,ch)<24 or size<w*h*.0015 or size>w*h*.75 or actual<size*.88:
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
        # A large SAM window can contain a whole module plus a disconnected
        # corner ornament. Do not turn the empty gap into one giant asset.
        count,parts,stats,_ = cv2.connectedComponentsWithStats(np.uint8(asset[:,:,3]>127),8)
        large = [(n,s) for n,s in enumerate(stats[1:],1) if s[4]>=250]
        split = area(box)>w*h*.12 and len(large)>1 and max(s[4] for _,s in large)>sum(s[4] for _,s in large)*.9
        if split:
            for number,(lx,ly,cw,ch,_) in large:
                proposals.append({'id':f"{segment.get('id')}_part_{number}",
                    'box':(x+int(lx),y+int(ly),x+int(lx+cw),y+int(ly+ch)),
                    'mask':np.uint8(parts[ly:ly+ch,lx:lx+cw]==number)*255,
                    'source':segment.get('source','opencv_segment'),'carrier':False})
        else:
            proposals.append({'id':segment.get('id'),'box':box,'mask':asset[:,:,3],
                              'source':segment.get('source','opencv_segment'),'carrier':False})
    for index,(box,mask) in enumerate(carrier_contours(image)):
        proposals.append({'id':f'closed_carrier_{index}','box':box,'mask':mask,
                          'source':'source_closed_carrier','carrier':True})
    for index, box in enumerate(photographic_rectangles(image)):
        x,y,bx,by = box
        proposals.append({'id':f'photo_boundary_{index}', 'box':box,
                          'mask':np.full((by-y,bx-x),255,np.uint8),
                          'source':'source_textured_rectangle', 'carrier':False, 'photo':True})
    interiors = []
    for proposal in proposals:
        x,y,bx,by = proposal['box']
        if area(proposal['box'])<w*h*.02 or min(bx-x,by-y)<80:
            continue
        patch = image[y:by,x:bx]
        visible = np.ones(patch.shape[:2],bool)
        for decision in text_decisions:
            b = decision.get('bbox')
            if b and overlap(b,proposal['box']):
                visible[max(0,b[1]-y-3):min(by-y,b[3]-y+3),max(0,b[0]-x-3):min(bx-x,b[2]-x+3)] = False
        paper = _visible_surface(patch,visible)
        if paper is None:
            continue
        foreground = (np.max(np.abs(patch.astype(float)-paper),axis=2)>25)&visible
        joined = cv2.morphologyEx(foreground.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        count,parts,stats,_ = cv2.connectedComponentsWithStats(joined,8)
        for number in range(1,count):
            lx,ly,cw,ch,size = stats[number]
            if min(cw,ch)<24 or size<180 or cw*ch>area(proposal['box'])*.45 or max(cw/ch,ch/cw)>5:
                continue
            box = (x+int(lx),y+int(ly),x+int(lx+cw),y+int(ly+ch))
            interiors.append({'id':f'interior_{len(interiors)}','box':box,
                'mask':np.uint8(parts[ly:ly+ch,lx:lx+cw]==number)*255,
                'source':'measured_paper_foreground_component','carrier':False})
    proposals.extend(interiors)
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
            near['photo'] = near.get('photo',False) or proposal.get('photo',False)
            duplicates += 1
            continue
        unique.append(proposal)
    result = []
    crossing = 0
    for proposal in unique:
        box = proposal['box']
        # An uncertain rectangular SAM crop cannot steal the upper/lower half
        # of a neighbouring complete plate (including plates behind native text).
        if not proposal['carrier'] and any(p['carrier'] and area(p['box'])<area(box)*.7
                and .08<overlap(box,p['box'])/area(p['box'])<.92 for p in unique):
            crossing += 1
            continue
        # Never split one raster OCR line among unrelated cutouts. Native lines
        # already have verified glyph owners and are cleaned before this step.
        raster = [d['bbox'] for d in text_decisions if d['decision']=='movable_image' and d.get('bbox')]
        if any(.03 < overlap(box,b)/area(b) < .97 for b in raster):
            crossing += 1
            continue
        # An enclosing compact visual owns its internal contour proposals:
        # photos in a stack or line-chart curves remain whole, movable objects.
        # Large panels instead allow independent children above their surface.
        children = [p for p in unique if p is not proposal and area(p['box'])<area(box)*.55
                    and overlap(box,p['box'])/area(p['box'])>.95]
        compact = area(box)<w*h*.035
        photo_children = [p for p in children if p.get('photo')]
        composite = any(overlap(a['box'],b['box']) == 0 for i,a in enumerate(photo_children) for b in photo_children[i+1:])
        is_panel = (proposal['carrier'] and (not compact or bool(children))) or composite
        if is_panel:
            # A carrier's silhouette is its OUTER boundary, not a foreground
            # matte with the labels and previously extracted objects punched out.
            contours, _ = cv2.findContours((proposal['mask']>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            support = np.zeros(proposal['mask'].shape,np.uint8)
            cv2.drawContours(support,contours,-1,255,-1)
            x,y,bx,by = box
            refined_box,rgba = box,np.dstack((image[y:by,x:bx],support))
            evidence = {'visualAssetPolicy':'closed_carrier_outer_contour',
                        'carrierSilhouette':'external_contour_without_child_holes'}
        elif proposal.get('photo'):
            x,y,bx,by = box
            refined_box,rgba = box,cv2.cvtColor(image[y:by,x:bx],cv2.COLOR_BGR2BGRA)
            evidence = {'visualAssetPolicy':'source_verified_photographic_rectangle'}
        else:
            refined_box,rgba,evidence = prepare_visual_seed(image,box,proposal['mask'])
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
    holes = support & owned & (rgba[:,:,3]==0)
    visible = support & ~owned & ~ink[y:by,x:bx]
    # Fit the same carrier surface across its visible area. A 12px ring often
    # crossed adjacent cards or contained only a shadow, leaving large holes.
    field = _visible_surface(cleaned[y:by,x:bx],visible)
    if field is not None:
        rgba[:,:,:3][holes] = field[holes]
        rgba[:,:,3][holes] = 255
        return holes,{'completedChildSurfacePixels':int(holes.sum()),
                      'unresolvedChildSurfacePixels':0,
                      'childSurfacePolicy':'robust_visible_carrier_surface_below_verified_owners'}
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


def _visible_surface(patch, visible):
    """Measured smooth paper only; reject texture and extrapolation without support."""
    h,w = visible.shape
    safe = cv2.erode(visible.astype(np.uint8),np.ones((5,5),np.uint8))>0
    ys,xs = np.where(safe)
    if len(xs)<max(80,h*w*.005) or np.ptp(xs)<w*.7 or np.ptp(ys)<h*.7:
        return None
    step = max(1,len(xs)//10000)
    xs,ys = xs[::step],ys[::step]
    xx,yy = xs/max(1,w-1)*2-1,ys/max(1,h-1)*2-1
    coords = np.stack((np.ones_like(xx),xx,yy,xx*yy,xx*xx,yy*yy),axis=-1)
    pixels = patch[ys,xs].astype(float)
    keep = np.ones(len(xs),bool)
    for _ in range(6):
        if keep.mean()<.85:
            return None
        coeff = np.linalg.lstsq(coords[keep],pixels[keep],rcond=None)[0]
        errors = np.max(np.abs(coords@coeff-pixels),axis=1)
        keep = errors<=max(4.,float(np.median(errors))*2.5)
    if keep.mean()<.90 or np.quantile(errors[keep],.95)>4:
        return None
    yy,xx = np.mgrid[-1:1:complex(h),-1:1:complex(w)]
    field = np.stack((np.ones_like(xx),xx,yy,xx*yy,xx*xx,yy*yy),axis=-1)@coeff
    if field.min()<pixels[keep].min()-12 or field.max()>pixels[keep].max()+12:
        return None
    return np.clip(np.rint(field),0,255).astype(np.uint8)


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
