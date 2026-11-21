"""One editable OCR line with measured internal typography, not child objects."""
import copy

import numpy as np


def pack_text_line(item, candidates):
    if len(candidates) < 2:
        return candidates
    parts = [p[0] for p in candidates]
    if ''.join(p['text'] for p in parts) != item['text']:
        return candidates
    mask = np.zeros_like(candidates[0][1])
    fill = candidates[0][2].copy()
    for _, owned, paper in candidates:
        overlap = mask & owned
        if overlap.any() and np.max(np.abs(fill[overlap].astype(float)-paper[overlap])) > 7:
            return candidates
        fill[owned & ~mask] = paper[owned & ~mask]
        mask |= owned
    element = copy.deepcopy(max(parts, key=lambda p: len(p['text'].strip())))
    x, y = min(p['x'] for p in parts), min(p['y'] for p in parts)
    w = max(p['x']+p['width'] for p in parts)-x
    h = max(p['y']+p['height'] for p in parts)-y
    runs = [{'text': p['text'], 'x': p['x']-x, 'y': p['y']-y,
             'width': p['width'], 'height': p['height'], 'style': copy.deepcopy(p['style'])}
            for p in parts]
    element.update(id=item['id'], text=item['text'], x=x, y=y, width=w, height=h, lines=[])
    element['metadata'].update(textEditingUnit='whole_ocr_line', sourceOcrIds=[item['id']],
        rawOCRBBox=[round(item['x']), round(item['y']), round(item['x']+item['width']), round(item['y']+item['height'])],
        measuredTextLine={'text':item['text'], 'style':copy.deepcopy(element['style']),
                          'width':w, 'height':h, 'runs':runs})
    return [(element, mask, fill)]


def measured_line_runs(element):
    """Edits to content or typography intentionally use the current whole-line style."""
    data = element.get('metadata', {}).get('measuredTextLine')
    if not data or data['text'] != element.get('text'):
        return []
    style = element.get('style', {})
    keys = ('fontFamily','fontWeight','fontStyle','color','letterSpacing','align')
    if any(style.get(k) != data['style'].get(k) for k in keys):
        return []
    scale = float(style.get('fontSize',24))/data['style']['fontSize']
    sx, sy = element['width']/data['width'], element['height']/data['height']
    return [{**r, 'x':element['x']+r['x']*sx, 'y':element['y']+r['y']*sy,
             'style':{**r['style'], 'fontSize':r['style']['fontSize']*scale,
                      'letterSpacing':r['style'].get('letterSpacing',0)*scale}}
            for r in data['runs']]
