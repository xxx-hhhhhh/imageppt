"""Align authoritative OCR characters to ink, then preserve mixed font spans.

No generated text or fixed slide coordinates. Ambiguous alignment retains the
original raster owner. Font-size discontinuity does not discard the line tail.
"""
import copy
import math
import unicodedata

import numpy as np

from app.services.reconstruction.editable_text import measured_surface, restore_text_candidate


def _kind(char):
    if unicodedata.category(char).startswith('P'):
        return 'punct'
    return 'han' if '\u2e80'<=char<='\u9fff' else 'latin'


def mixed_spans(source: np.ndarray,item: dict) -> list[dict]:
    text = str(item.get('text') or '')
    if len(text)<2 or len(text)>80 or '\n' in text or float(item.get('confidence') or 0)<.9:
        return []
    chars = [(index,char) for index,char in enumerate(text) if not char.isspace()]
    raw=[float(item[k]) for k in ('x','y','width','height')]
    if not all(np.isfinite(v) for v in raw) or min(raw[2:])<=0:
        return []
    x,y,w,h = [round(v) for v in raw]
    if x<0 or y<0 or x+w>source.shape[1] or y+h>source.shape[0]:
        return []
    ax,ay,bx,by = max(0,x-6),max(0,y-6),min(source.shape[1],x+w+6),min(source.shape[0],y+h+6)
    patch = source[ay:by,ax:bx]
    measured = measured_surface(patch)
    if measured is None:
        return []
    paper,_ = measured
    try:
        fg = np.array(list(bytes.fromhex(item.get('style',{}).get('color','#111111')[1:]))[::-1],float)
    except ValueError:
        return []
    pixels = patch.astype(float)
    direction = fg-paper
    alpha = np.sum((pixels-paper)*direction,axis=2)/np.maximum(1,np.sum(direction*direction,axis=2))
    ink = (alpha>.4)[y-ay:y+h-ay,x-ax:x+w-ax]
    used = ink.any(axis=0)
    starts = np.where(used & ~np.r_[False,used[:-1]])[0]
    ends = np.where(used & ~np.r_[used[1:],False])[0]+1
    blocks = list(zip(starts,ends))
    if not len(chars)<=len(blocks)<=len(chars)*3:
        return []
    # DP joins disconnected strokes (e.g. 门/川) rather than treating each
    # projection island as another character. All OCR chars must align in order.
    states = {(0,0):(0.,[])}
    for index,(_,char) in enumerate(chars):
        for (number,start),(cost,path) in list(states.items()):
            if number!=index:
                continue
            for end in range(start+1,min(len(blocks),start+3)+1):
                if len(blocks)-end<len(chars)-index-1:
                    continue
                left,right = blocks[start][0],blocks[end-1][1]
                yy,_ = np.where(ink[:,left:right])
                top,bottom = int(yy.min()),int(yy.max()+1)
                width,height = right-left,bottom-top
                kind = _kind(char)
                ratio = width/max(1,height)
                ideal = .94 if kind=='han' else .62 if kind=='latin' else .45
                limits = (.48,1.45) if kind=='han' else (.14,1.08) if kind=='latin' else (.08,1.3)
                if not limits[0]<=ratio<=limits[1]:
                    continue
                score = cost+math.log(max(.01,ratio)/ideal)**2+(end-start-1)*.03
                key = (index+1,end)
                if key not in states or score<states[key][0]:
                    states[key] = (score,path+[(int(left),top,int(right),bottom)])
    match = states.get((len(chars),len(blocks)))
    if match is None or match[0]/len(chars)>.38:
        return []
    boxes = match[1]
    sizes=[]
    for (_,char),(left,top,right,bottom) in zip(chars,boxes):
        sizes.append((bottom-top)/(.93 if _kind(char)=='han' else .70) if _kind(char)!='punct' else None)
    known = [s for s in sizes if s]
    if not known or max(known)/min(known)<1.28:
        return []
    for i,size in enumerate(sizes):
        if size is None:
            sizes[i]=next((sizes[j] for j in range(i-1,-1,-1) if sizes[j]),next(s for s in sizes if s))
    groups=[]
    for i,size in enumerate(sizes):
        if not groups or _kind(chars[i][1])=='punct' or _kind(chars[i-1][1])=='punct' or abs(math.log(size/np.median([sizes[j] for j in groups[-1]])))>math.log(1.22):
            groups.append([i])
        else:
            groups[-1].append(i)
    if len(groups)<2:
        return []
    result=[]
    for index,group in enumerate(groups):
        start=chars[group[0]][0] if index else 0
        stop=chars[group[-1]+1][0] if group[-1]+1<len(chars) else len(text)
        subset=[boxes[j] for j in group]
        left,top,right,bottom=min(b[0] for b in subset),min(b[1] for b in subset),max(b[2] for b in subset),max(b[3] for b in subset)
        child=copy.deepcopy(item)
        child.update(id=f"{item['id']}_span_{index+1}",text=text[start:stop],x=x+left,y=y+top,width=right-left,height=bottom-top)
        child['lines']=[]
        child['metadata']={**child.get('metadata',{}),'sourceOcrIds':[item['id']],
                           'sourceCharRange':[start,stop],'mixedTypography':True}
        result.append(child)
    assert ''.join(s['text'] for s in result)==text
    return result


def restore_text_candidates(source,item,hints=None,diagnostics=None):
    report=diagnostics if diagnostics is not None else {}
    children=mixed_spans(source,item)
    if children:
        candidates=[]
        reasons=[]
        for child in children:
            evidence={}
            candidate=restore_text_candidate(source,child,hints,evidence)
            reasons.append({'id':child['id'],'text':child['text'],'reason':evidence.get('reason')})
            if candidate is None:
                # Do not lose the rest of a line because ONE size is uncertain.
                continue
            element,mask,fill=candidate
            element['metadata'].update(child['metadata'])
            candidates.append((element,mask,fill))
        report.update(reason='mixed_typography_spans',spans=reasons)
        if candidates:
            return candidates
    candidate=restore_text_candidate(source,item,hints,report)
    return [candidate] if candidate else []
