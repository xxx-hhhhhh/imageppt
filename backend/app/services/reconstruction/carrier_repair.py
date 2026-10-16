"""One-pass local floor completion, only below already-persisted child owners.

Never erase or repaint a visible pixel. No global whitening, source-text copy,
or generative inpainting. Uncertain surfaces stay explicitly unresolved.
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


def complete_local_floor(rgba: np.ndarray, support: np.ndarray,
                         verified_child_above: np.ndarray) -> int:
    holes = support & (rgba[:, :, 3] == 0) & verified_child_above
    # Snapshot samples prevents sequential fills from becoming new evidence.
    known = support & (rgba[:, :, 3] == 255)
    original = rgba[:, :, :3].astype(float).copy()
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
    return filled
