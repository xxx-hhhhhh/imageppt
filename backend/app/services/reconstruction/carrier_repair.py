"""One-pass local floor completion, only below already-persisted child owners.

Never erase or repaint a visible pixel. No global whitening, source-text copy,
or generative inpainting. Smooth paper may be approximated when an exact
gradient cannot be recovered; surfaces without paper evidence stay unresolved.
"""
import cv2
import numpy as np


def unowned_contour_fringe(support: np.ndarray, owned: np.ndarray, radius: int = 3) -> np.ndarray:
    """Attach only a narrow contour's unclaimed antialias fringe, not its bbox.

    The caller pads the local support before passing it, and persists the new
    edge owner before assigning the remaining page pixels to fallback assets.
    """
    expanded = cv2.dilate(support.astype(np.uint8), np.ones((radius*2+1, radius*2+1), np.uint8)) > 0
    return expanded & ~support & ~owned


def inherit_carrier_groups(elements: list[dict]) -> None:
    """Use agreeing contained detail owners when a semantic bbox misses a rim."""
    by_id = {e['id']: e for e in elements}
    for carrier in elements:
        metadata = carrier.get('metadata', {})
        ids = metadata.get('carrierDetailIds', [])
        if carrier.get('groupId') or not metadata.get('carrierSurfaceRebuilt') or not ids:
            continue
        details = [by_id[i] for i in ids if i in by_id]
        groups = {e.get('groupId') for e in details}
        if len(details)!=len(ids) or len(groups)!=1 or None in groups:
            continue
        x,y,w,h = (carrier[k] for k in ('x','y','width','height'))
        if any(e['x']<x or e['y']<y or e['x']+e['width']>x+w or e['y']+e['height']>y+h for e in details):
            continue
        dx,dy = min(e['x'] for e in details),min(e['y'] for e in details)
        bx,by = max(e['x']+e['width'] for e in details),max(e['y']+e['height'] for e in details)
        if (bx-dx)*(by-dy)<w*h*.7:
            continue
        carrier['groupId'] = carrier['parent'] = next(iter(groups))
        metadata['moduleAssignmentEvidence'] = 'contained_detail_owners_agree_covering_carrier_interior'


def separate_carrier_surface(rgb: np.ndarray, support: np.ndarray,
                             available: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray] | None:
    """Measure a simple plate from its rim, preserving stronger content first.

    Only a caller with an independently verified closed boundary may use this.
    Returns a smooth interior, independent detail silhouettes, and the interior
    allowed to be rebuilt. Existing outer border pixels remain untouched.
    """
    from app.services.reconstruction.contour_objects import _visible_surface

    h, w = support.shape
    if min(h, w) < 100:
        return None
    inner = cv2.erode(support.astype(np.uint8), np.ones((7, 7), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    radius = max(12, min(24, round(min(h, w)*.05)))
    core = cv2.erode(support.astype(np.uint8), np.ones((radius*2+1, radius*2+1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    sample = inner & ~core & available & (rgb.min(axis=2)>220)
    if sample.sum() < 100:
        return None
    median = np.median(rgb[sample], axis=0)
    sample &= np.max(np.abs(rgb.astype(float)-median), axis=2)<=8
    field = _visible_surface(rgb, sample)
    if field is None:
        return None
    # Everything differing materially from the measured paper needs a separate
    # persisted owner. A two-pixel fringe preserves faint antialiasing as well.
    detail = inner & (np.max(np.abs(rgb.astype(float)-field), axis=2)>6)
    detail = cv2.dilate(detail.astype(np.uint8), np.ones((5, 5), np.uint8))>0
    detail &= inner
    contours, _ = cv2.findContours(detail.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masks = []
    fragments = np.zeros_like(support)
    for contour in contours:
        mask = np.zeros(support.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, -1)
        mask = (mask>0) & inner
        yy, xx = np.where(mask & available)
        if len(xx):
            if len(xx)<128 or np.ptp(xx)<16 or np.ptp(yy)<12:
                fragments |= mask
            else:
                masks.append(mask)
    # Keep tiny residual decorations together; do not create dozens of 1px
    # editable objects. Their real alpha remains disjoint, never a bbox crop.
    if fragments.any():
        masks.append(fragments)
    rebuilt = rgb.copy()
    rebuilt[inner] = field[inner]
    return rebuilt, masks, inner


def _approximate_paper(original: np.ndarray, known: np.ndarray,
                       support: np.ndarray, holes: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Diffuse only measured, smooth pale colours, never source image content.

    A coarse-to-fine harmonic field bridges even large missing interiors. Its
    donors are opaque smooth paper patches or coherent narrow paper fringes,
    excluding ink, saturated borders and texture. Each silhouette needs its
    own evidence. The field is
    only a colour proposal: the caller writes exclusively to authorized holes.
    """
    kernel = np.ones((3, 3), np.uint8)
    rgb = original.astype(np.uint8)
    safe = cv2.erode(known.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                     borderValue=0) > 0
    spread = cv2.dilate(rgb, kernel).astype(np.int16) - cv2.erode(rgb, kernel)
    pale = known & (rgb.min(axis=2) > 175)
    donors = safe & pale & (spread.max(axis=2) <= 10)
    # A photo can leave only a one-pixel paper fringe. Measure neighbours that
    # actually belong to that fringe, without consulting RGB hidden by alpha.
    high = cv2.dilate(np.where(pale[:, :, None], rgb, 0).astype(np.uint8), kernel)
    low = cv2.erode(np.where(pale[:, :, None], rgb, 255).astype(np.uint8), kernel)
    neighbors = cv2.filter2D(pale.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    narrow = pale & (neighbors >= 3) & ((high.astype(np.int16)-low).max(axis=2) <= 10)
    result = np.zeros_like(rgb)
    completed = np.zeros_like(holes)
    donor_count = 0
    count, parts, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), 8)
    for number in range(1, count):
        x, y, w, h, _ = stats[number]
        region = parts[y:y+h, x:x+w] == number
        target = holes[y:y+h, x:x+w] & region
        sample = donors[y:y+h, x:x+w] & region
        colors = original[y:y+h, x:x+w].astype(np.float32)
        if sample.sum() < 24:
            fringe = narrow[y:y+h, x:x+w] & region
            if fringe.sum() >= 16 and np.max(np.ptp(np.percentile(colors[fringe], [5, 95], axis=0), axis=0)) <= 16:
                sample = fringe
        # A few isolated smooth pixels inside a photograph are not a floor.
        if not target.any() or sample.sum() < max(16, int((pale[y:y+h, x:x+w] & region).sum() * .15)):
            continue
        weights = sample.astype(np.float32)
        levels = [(colors * weights[:, :, None], weights)]
        while max(levels[-1][1].shape) > 32:
            values, weight = levels[-1]
            size = (max(1, (weight.shape[1]+1)//2), max(1, (weight.shape[0]+1)//2))
            levels.append((cv2.resize(values, size, interpolation=cv2.INTER_AREA),
                           cv2.resize(weight, size, interpolation=cv2.INTER_AREA)))
        field = None
        median = np.median(colors[sample], axis=0)
        for values, weight in reversed(levels):
            size = (weight.shape[1], weight.shape[0])
            if field is None:
                field = np.broadcast_to(median, (*weight.shape, 3)).copy()
            else:
                field = cv2.resize(field, size, interpolation=cv2.INTER_LINEAR)
            # Continuous confidence at coarse levels preserves narrow strips.
            # At full resolution measured donors are fixed Dirichlet data.
            for _ in range(120):
                field = cv2.blur(field, (3, 3)) * (1-weight[:, :, None]) + values
        result[y:y+h, x:x+w][target] = np.clip(np.rint(field[target]), 0, 255).astype(np.uint8)
        completed[y:y+h, x:x+w] |= target
        donor_count += int(sample.sum())
    return result, completed, donor_count


def complete_local_floor(rgba: np.ndarray, support: np.ndarray,
                         verified_child_above: np.ndarray, *,
                         sample_mask: np.ndarray | None = None,
                         diagnostics: dict | None = None) -> int:
    holes = support & (rgba[:, :, 3] == 0) & verified_child_above
    # Snapshot samples prevents sequential fills from becoming new evidence.
    known = support & (rgba[:, :, 3] == 255)
    original = rgba[:, :, :3].astype(float).copy()
    initial_holes = holes.copy()
    filled = 0
    light = known & (original.min(axis=2) > 175)
    # A thin cyan border must not invalidate an otherwise verified pale floor.
    # Require >=98% light evidence, spatial coverage and the same strict smooth
    # field residual gates; exclude that border from samples, never repaint it.
    if holes.any() and light.sum() >= max(80, known.sum()*.98):
        from app.services.reconstruction.contour_objects import _visible_surface
        field = _visible_surface(rgba[:, :, :3], light)
        if field is not None:
            rgba[:, :, :3][holes] = field[holes]
            rgba[:, :, 3][holes] = 255
            filled = int(holes.sum())
            holes = np.zeros_like(holes)
    count, parts, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    h, w = support.shape
    for number in range(1, count):
        x, y, cw, ch, size = stats[number]
        if size < 4 or size > h*w*.25:
            continue
        pad = min(48, max(16, min(cw, ch)))
        ax, ay, bx, by = max(0, x-pad), max(0, y-pad), min(w, x+cw+pad), min(h, y+ch+pad)
        hole = parts[ay:by, ax:bx] == number
        ring = cv2.dilate(hole.astype(np.uint8), np.ones((pad*2+1, pad*2+1), np.uint8)) > 0
        ring &= cv2.dilate(hole.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
        samples = ring & known[ay:by, ax:bx]
        yy, xx = np.where(samples)
        if len(xx) < 80 or np.ptp(xx) < cw*.7 or np.ptp(yy) < min(ch*.4, 12):
            continue
        colors = original[ay+yy, ax+xx]
        # Floor evidence must be a coherent light carrier, not a photograph,
        # ink, a blue title stripe, or mixed surfaces across a card boundary.
        light = colors.min(axis=1) > 175
        if light.mean() < .90:
            continue
        xx, yy, colors = xx[light], yy[light], colors[light]
        coords = np.column_stack((np.ones(len(xx)), xx/max(cw, 1), yy/max(ch, 1)))
        coeff = np.linalg.lstsq(coords, colors, rcond=None)[0]
        error = np.max(np.abs(coords@coeff-colors), axis=1)
        good = error <= 4
        if good.mean() < .95:
            continue
        coeff = np.linalg.lstsq(coords[good], colors[good], rcond=None)[0]
        hy, hx = np.where(hole)
        field = np.column_stack((np.ones(len(hx)), hx/max(cw, 1), hy/max(ch, 1)))@coeff
        if np.any(field < colors[good].min(axis=0)-8) or np.any(field > colors[good].max(axis=0)+8):
            continue
        rgba[ay+hy, ax+hx, :3] = np.clip(np.rint(field), 0, 255).astype(np.uint8)
        rgba[ay+hy, ax+hx, 3] = 255
        filled += len(hx)
    strict_filled = filled
    remaining = initial_holes & (rgba[:, :, 3] == 0)
    donor_count = 0
    if remaining.any():
        evidence = known if sample_mask is None else known & sample_mask
        field, approximate, donor_count = _approximate_paper(original, evidence, support, remaining)
        rgba[:, :, :3][approximate] = field[approximate]
        rgba[:, :, 3][approximate] = 255
        filled += int(approximate.sum())
    if diagnostics is not None:
        diagnostics.update(strictGradientPixels=strict_filled,
                           approximatePaperPixels=filled-strict_filled,
                           approximatePaperSamples=donor_count,
                           unresolvedPixels=int(initial_holes.sum())-filled,
                           policy='same_carrier_smooth_paper_harmonic_fallback_below_verified_owner')
    return filled
