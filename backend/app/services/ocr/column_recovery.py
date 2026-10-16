"""Bounded local re-reading of uncertain vertical stacks, never guessed labels."""
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import cv2
import numpy as np

from app.services.ocr.provider import OCRResult, _style_from_region
from app.utils import image_io


def recover_columns(image_path: Path, results: list[OCRResult], provider) -> list[OCRResult]:
    image = image_io.imread(image_path)
    if image is None:
        return results
    recovered = []
    attempts = 0
    started = monotonic()
    for original in results:
        x,y,bx,by = [round(v) for v in original.bbox]
        w,h = bx-x,by-y
        if not (original.confidence<.85 and h>w*1.8 and 6<=w<=60 and 20<=h<=240) or attempts>=12 or monotonic()-started>20:
            recovered.append(original)
            continue
        x,y,bx,by = max(0,x-3),max(0,y-3),min(image.shape[1],bx+3),min(image.shape[0],by+3)
        patch=image[y:by,x:bx]
        paper=np.median(np.concatenate((patch[:,0],patch[:,-1])),axis=0)
        used=(np.max(np.abs(patch.astype(float)-paper),axis=2)>40).sum(axis=1)>=2
        starts=np.flatnonzero(used & ~np.r_[False,used[:-1]])
        ends=np.flatnonzero(used & ~np.r_[used[1:],False])+1
        bands=[(max(0,int(a)-3),min(by-y,int(b)+3)) for a,b in zip(starts,ends) if b-a>=5]
        if not 2<=len(bands)<=8 or attempts+len(bands)>12:
            recovered.append(original)
            continue
        rows=[]
        try:
            with TemporaryDirectory(prefix='image2ppt-column-') as directory:
                for number,(top,bottom) in enumerate(bands):
                    if monotonic()-started>20:
                        break
                    attempts+=1
                    crop=cv2.resize(patch[top:bottom],None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
                    border=32
                    crop=cv2.copyMakeBorder(crop,border,border,border,border,cv2.BORDER_CONSTANT,value=paper.tolist())
                    path=Path(directory)/f'row_{number}.png'
                    image_io.imwrite(path,crop)
                    candidates=provider.recognize(path)
                    if len(candidates)!=1 or candidates[0].confidence<.85 or not candidates[0].text.strip():
                        break
                    result=candidates[0]
                    left,upper,right,lower=result.bbox
                    mapped=[x+(left-border)/4,y+top+(upper-border)/4,x+(right-border)/4,y+top+(lower-border)/4]
                    if mapped[0]<x-3 or mapped[1]<y-3 or mapped[2]>bx+3 or mapped[3]>by+3:
                        break
                    mapped=[max(0,mapped[0]),max(0,mapped[1]),min(image.shape[1],mapped[2]),min(image.shape[0],mapped[3])]
                    style=_style_from_region(image_path,mapped,result.text,result.confidence)
                    style['ocrRecoveryPolicy']='bounded_4x_vertical_column_reread'
                    rows.append(OCRResult(result.text,mapped,result.confidence,style))
        except Exception:
            recovered.append(original)
            continue
        rows.sort(key=lambda r:r.bbox[1])
        complete = (len(rows)==len(bands) and all(a.bbox[3]<=b.bbox[1] for a,b in zip(rows,rows[1:]))
                    and sum(r.bbox[3]-r.bbox[1] for r in rows)>=h*.35)
        recovered.extend(rows if complete else [original])
    return recovered
