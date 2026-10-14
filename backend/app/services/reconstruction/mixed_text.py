"""Align authoritative OCR characters to ink, then preserve mixed font spans.

No generated text or fixed slide coordinates. Ambiguous alignment retains the
original raster owner. Font-size discontinuity does not discard the line tail.
"""
import copy
import math
import re
import unicodedata

import numpy as np

from app.services.reconstruction.editable_text import (
    measured_text_surface,
    restore_text_candidate,
    without_untranscribed_bullet,
)
from app.services.typography.font_raster import font_face


def _kind(char):
    if unicodedata.category(char).startswith(('P','S')):
        return 'punct'
    return 'han' if '\u2e80'<=char<='\u9fff' else 'latin'


def mixed_spans(source: np.ndarray,item: dict, *, typography_runs: bool=False) -> list[dict]:
    text = str(item.get('text') or '')
    if len(text)<2 or len(text)>80 or '\n' in text or float(item.get('confidence') or 0)<.85:
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
    measured = measured_text_surface(patch)
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
    ink,_ = without_untranscribed_bullet(ink,text,h)
    used = ink.any(axis=0)
    starts = np.where(used & ~np.r_[False,used[:-1]])[0]
    ends = np.where(used & ~np.r_[used[1:],False])[0]+1
    blocks = list(zip(starts,ends))
    if not len(chars)<=len(blocks)<=len(chars)*3:
        return []
    # DP joins disconnected strokes (e.g. 门/川) rather than treating each
    # projection island as another character. All OCR chars must align in order.
    states = {(0,0):(0.,[])}
    reference = font_face('msyh.ttc',100)
    for index,(_,char) in enumerate(chars):
        reference_box = reference.getmask(char).getbbox() or reference.getbbox(char)
        reference_ratio = (reference_box[2]-reference_box[0])/max(1,reference_box[3]-reference_box[1])
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
                # 一 / em dash / quotes are not square Han glyphs. Use their
                # real ink aspect, with the old broad bounds for normal glyphs.
                ideal = reference_ratio if reference_ratio>1.45 or kind=='punct' else .94 if kind=='han' else .62
                limits = (ideal*.3,ideal*2) if reference_ratio>1.45 else (.48,1.45) if kind=='han' else (.14,1.08) if kind=='latin' else (ideal*.45,ideal*1.8)
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
    # Capitals, x-height, ascenders and punctuation have different ink heights
    # at ONE font size. Normalize by the measured glyph, not a fixed .70.
    for (_,char),(left,top,right,bottom) in zip(chars,boxes):
        glyph_box = reference.getmask(char).getbbox() or reference.getbbox(char)
        nominal_height = max(1,glyph_box[3]-glyph_box[1])/100
        if char.isdigit():
            nominal_height = .70
        sizes.append((bottom-top)/nominal_height if _kind(char)!='punct' and nominal_height>.45 else None)
    known = [s for s in sizes if s]
    if not known or (max(known)/min(known)<1.4 and not typography_runs):
        return []
    for i,size in enumerate(sizes):
        if size is None:
            sizes[i]=next((sizes[j] for j in range(i-1,-1,-1) if sizes[j]),next(s for s in sizes if s))
    groups=[]
    for i,size in enumerate(sizes):
        if typography_runs:
            char = chars[i][1]
            previous = chars[i-1][1] if i else ''
            # Keep punctuation with its word; a colon starts a new text run.
            # Script transitions and real source sizes use measured positions,
            # not one squeezed font across Chinese + English + a heading label.
            gap = boxes[i][0]-boxes[i-1][2] if i else 0
            boundary = bool(i and (previous in ':：' or (gap>max(6,size*.4) and len(chars)-i>2)
                or (_kind(char) != 'punct' and (
                    (_kind(previous) != 'punct' and _kind(previous) != _kind(char))
                    or abs(math.log(size/np.median([sizes[j] for j in groups[-1]]))) > math.log(1.22)
                    or (len(groups[-1])>=8 and len(chars)-i>2)))))
        else:
            boundary = bool(i and (_kind(chars[i][1])=='punct' or _kind(chars[i-1][1])=='punct' or abs(math.log(size/np.median([sizes[j] for j in groups[-1]])))>math.log(1.22)))
        if not groups or boundary:
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
    # Prefer one reliable editable line. Character-level partitioning is a
    # fallback for a genuine size discontinuity, never the default for Latin.
    whole=restore_text_candidate(source,item,hints,report)
    fit = whole[0]['metadata']['fontMeasurement'] if whole else {}
    needs_runs = whole is None or fit.get('fontAlphaError',0)>.22 or fit.get('letterSpacing',0)<-.03*fit.get('fontSize',1)
    if whole is not None and not needs_runs:
        return [whole]
    paper_reference=_line_paper(source,item)
    options=[]
    colon = _colon_candidates(source,item,hints,paper_reference)
    if colon:
        options.append((colon,{'reason':'measured_label_value_typography','editableRunCount':len(colon)}))
    seen=set()
    for typography_runs in (False,True):
        children=mixed_spans(source,item,typography_runs=typography_runs)
        signature=tuple(child['text'] for child in children)
        if not children or signature in seen:
            continue
        seen.add(signature)
        candidates=[]
        reasons=[]
        for child in children:
            evidence={}
            candidate=(restore_text_candidate(source,child,hints,evidence,paper_reference=paper_reference)
                       if paper_reference is not None else restore_text_candidate(source,child,hints,evidence))
            reasons.append({'id':child['id'],'text':child['text'],'reason':evidence.get('reason')})
            if candidate is None:
                # Atomic OCR line: incomplete span recovery must not leave a
                # sentence alternating between pictures and editable letters.
                report.update(reason='whole_line_preserved_after_run_rejection' if whole else 'incomplete_mixed_line_retained',spans=reasons)
                break
            element,mask,fill=candidate
            element['metadata'].update(child.get('metadata',{}))
            candidates.append((element,mask,fill))
        if len(candidates)==len(children):
            options.append((candidates,{'reason':'mixed_typography_spans','spans':reasons}))
    if options:
        # A colon split may still contain a long, poorly matched tail. Compare
        # complete measured alternatives instead of accepting the first one.
        candidates,evidence=min(options,key=lambda option:_run_error(option[0])+.004*(len(option[0])-1))
        if whole is None or _run_error(candidates)+.025 < _run_error([whole]):
            report.update(evidence)
            return candidates
    if whole is not None:
        report.update(reason='whole_line_fit_better_than_runs')
    return [whole] if whole is not None else []


def _run_error(candidates):
    total=0.
    count=0
    for element,_,_ in candidates:
        fit=element['metadata']['fontMeasurement']
        n=len(element['text'])
        total+=n*(fit['fontAlphaError']+.5*(1-fit['fontMaskSimilarity'])+fit['fontDimensionError']*.5)
        count+=n
    return total/max(1,count)


def _line_paper(source,item):
    if not all(k in item for k in ('x','y','width','height')):
        return None
    values=[float(item[k]) for k in ('x','y','width','height')]
    if not all(np.isfinite(v) for v in values) or min(values[2:])<3:
        return None
    x,y,w,h=map(round,values)
    ax,ay,bx,by=max(0,x-6),max(0,y-6),min(source.shape[1],x+w+6),min(source.shape[0],y+h+6)
    if bx<=ax or by<=ay:
        return None
    model=measured_text_surface(source[ay:by,ax:bx])
    return (ax,ay,*model) if model is not None else None


def _colon_candidates(source,item,hints,paper_reference=None):
    """Measure label/value runs even when Latin letters touch in projection.

    OCR remains authoritative. A punctuation boundary only proposes a split;
    both sides must independently fit real ink and preserve the full string.
    """
    text = str(item.get('text') or '')
    match = re.match(r'^(.{2,}?[：:])(\s*.{2,})$',text)
    if not match or '\n' in text:
        return []
    x,y,w,h = [round(float(item[k])) for k in ('x','y','width','height')]
    if min(x,y)<0 or min(w,h)<=0 or x+w>source.shape[1] or y+h>source.shape[0]:
        return []
    ax,ay,bx,by=max(0,x-6),max(0,y-6),min(source.shape[1],x+w+6),min(source.shape[0],y+h+6)
    patch=source[ay:by,ax:bx]
    model=measured_text_surface(patch)
    if model is None:
        return []
    contrast=np.max(np.abs(patch.astype(float)-model[0].astype(float)),axis=2)
    ink=(contrast>30)[y-ay:y+h-ay,x-ax:x+w-ax]
    used=ink.any(axis=0)
    gaps=np.where(~used & np.r_[False,used[:-1]])[0]
    lengths=[sum(1 if _kind(c)=='han' else .55 for c in part) for part in match.groups()]
    expected=w*lengths[0]/sum(lengths)
    cuts=sorted((int(c) for c in gaps if .12*w<c<.88*w and abs(c-expected)<.22*w),key=lambda c:abs(c-expected))[:6]
    best=None
    for cut in cuts:
        candidates=[]
        for index,(left,right,body) in enumerate(((0,cut,match[1]),(cut,w,match[2]))):
            yy,xx=np.where(ink[:,left:right])
            if not len(xx):
                break
            child=copy.deepcopy(item)
            start=0 if index==0 else len(match[1])
            child.update(id=f"{item['id']}_run_{index+1}",text=body,
                         x=x+left+int(xx.min()),y=y+int(yy.min()),
                         width=int(xx.max()-xx.min()+1),height=int(yy.max()-yy.min()+1),lines=[])
            candidate=restore_text_candidate(source,child,hints,{},paper_reference=paper_reference)
            if candidate is None:
                break
            candidate[0]['metadata'].update(sourceOcrIds=[item['id']],sourceCharRange=[start,start+len(body)],mixedTypography=True)
            candidates.append(candidate)
        if len(candidates)!=2:
            continue
        score=_run_error(candidates)
        if best is None or score<best[0]:
            best=(score,candidates)
    return best[1] if best else []
