"""Shared measured glyph raster for font matching, preview and preservation QA."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path('C:/Windows/Fonts')


@lru_cache(maxsize=1024)
def font_face(filename: str,size: int):
    if Path(filename).name != filename:
        raise ValueError('Font must be an installed face name')
    return ImageFont.truetype(str(FONT_DIR/filename),size)


def glyph_raster(text: str,filename: str,size: int,stroke: float=0,spacing: float=0) -> tuple[np.ndarray,tuple[int,int]]:
    scale = 4 if stroke or spacing else 1
    font = font_face(filename,size*scale)
    bbox = font.getbbox(text)
    pad = max(8,round(stroke*scale)+4)
    image = Image.new('L',(max(1,bbox[2]-bbox[0]+round(spacing*scale*max(0,len(text)-1)))+pad*2,bbox[3]-bbox[1]+pad*2))
    draw=ImageDraw.Draw(image)
    if spacing:
        for i,char in enumerate(text):
            px=pad-bbox[0]+font.getlength(text[:i])+i*spacing*scale
            draw.text((px,pad-bbox[1]),char,font=font,fill=255,stroke_width=round(stroke*scale),stroke_fill=255)
    else:
        draw.text((pad-bbox[0],pad-bbox[1]),text,font=font,fill=255,
                  stroke_width=round(stroke*scale),stroke_fill=255)
    raw = np.asarray(image)
    # Maintain the FONT origin, including stroke/serif bearings, after AA.
    if scale>1:
        raw = cv2.resize(raw,(round(raw.shape[1]/scale),round(raw.shape[0]/scale)),interpolation=cv2.INTER_AREA)
    yy,xx = np.where(raw>8)
    if not len(xx):
        return np.zeros((1,1),np.uint8),(0,0)
    ox,oy = round((bbox[0]-pad)/scale)+int(xx.min()),round((bbox[1]-pad)/scale)+int(yy.min())
    return raw[yy.min():yy.max()+1,xx.min():xx.max()+1],(ox,oy)


def paint_measured_text(base: Image.Image,element: dict,color: str | None=None) -> None:
    style = element.get('style') or {}
    # fontFile is measurement provenance, not a second authoritative font. A
    # manual family/weight/size edit must also change the backend preview.
    weight=int(style.get('fontWeight',400))
    family=style.get('fontFamily','Microsoft YaHei')
    faces={'SimSun':'simsun.ttc','宋体':'simsun.ttc','KaiTi':'simkai.ttf','楷体':'simkai.ttf',
           'SimHei':'simhei.ttf','黑体':'simhei.ttf','FangSong':'simfang.ttf','仿宋':'simfang.ttf',
           'Microsoft YaHei':'msyhbd.ttc' if weight>=600 else 'msyh.ttc',
           'Arial':'arialbd.ttf' if weight>=600 else 'arial.ttf',
           'Times New Roman':'timesbd.ttf' if weight>=600 else 'times.ttf'}
    filename=faces.get(family,style.get('fontFile','msyh.ttc'))
    size=round(style['fontSize'])
    stroke=size/48 if filename=='simsun.ttc' and weight>=600 else 0
    alpha,offset = glyph_raster(str(element.get('text') or ''),filename,size,stroke,float(style.get('letterSpacing',0)))
    layer = Image.new('RGBA',(alpha.shape[1],alpha.shape[0]),color or style.get('color','#111827'))
    layer.putalpha(Image.fromarray(alpha))
    x,y = round(element['x']+offset[0]),round(element['y']+offset[1])
    base.alpha_composite(layer,(x,y))
