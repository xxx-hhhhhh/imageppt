"""General mixed-label and incomplete-segmentation fixtures, not slide hacks."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from app.services.reconstruction.editable_text import _fit_font
from app.services.reconstruction.mixed_text import restore_text_candidates
from app.services.reconstruction.visual_assets import prepare_visual_seed
from app.services.typography.font_raster import font_face, glyph_raster
from app.services.validation.reconstruction_quality import reconstruction_quality
from PIL import Image, ImageDraw, ImageFont


def test_label_value_keeps_separate_sizes_without_squeezing():
    path=Path('C:/Windows/Fonts/msyhbd.ttc')
    if not path.exists():
        pytest.skip('Installed Windows font required')
    image=Image.new('RGB',(700,100),'#E8F5EF')
    draw=ImageDraw.Draw(image)
    parts=[('实验定义：',22,35),('传感器迁移学习',30,27)]
    x=24
    boxes=[]
    for text,size,y in parts:
        font=ImageFont.truetype(str(path),size)
        draw.text((x,y),text,font=font,fill='#17365D')
        boxes.append(draw.textbbox((x,y),text,font=font))
        x+=round(draw.textlength(text,font=font))+12
    left,top=min(b[0] for b in boxes),min(b[1] for b in boxes)
    right,bottom=max(b[2] for b in boxes),max(b[3] for b in boxes)
    item={'id':'line','type':'text','text':''.join(p[0] for p in parts),'confidence':.99,
          'x':left,'y':top,'width':right-left,'height':bottom-top,
          'style':{'color':'#17365D','fontFamily':'Microsoft YaHei'}}
    source=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    candidates=restore_text_candidates(source,item)
    assert ''.join(e['text'] for e,_,_ in candidates)==item['text']
    sizes=[e['style']['fontSize'] for e,_,_ in candidates]
    assert len(candidates)>=2 and max(sizes)-min(sizes)>=6
    assert all(e['style']['letterSpacing']>=-.055*e['style']['fontSize'] for e,_,_ in candidates)


def test_matte_recovers_attached_detail_missing_from_segment():
    image=np.full((120,140,3),255,np.uint8)
    cv2.circle(image,(65,55),20,(40,80,130),-1,cv2.LINE_AA)
    # Thin appendage is visually attached but absent from the proposed mask.
    cv2.line(image,(65,73),(65,93),(60,90,120),3,cv2.LINE_AA)
    raw=np.zeros((80,80),np.uint8)
    cv2.circle(raw,(40,30),20,255,-1)
    box,rgba,metadata=prepare_visual_seed(image,(25,25,105,105),raw)
    x,y,_,_=box
    assert rgba[89-y,65-x,3]>0
    assert metadata['recoveredVisualPixelCount']>0
    assert metadata['matteValidation']=='source_contrast_components_and_local_composite'
    assert rgba[0,0,3]==0  # Not a white square fallback.


def test_variable_font_weight_matches_requested_not_thin_default():
    if not Path('C:/Windows/Fonts/NotoSansSC-VF.ttf').exists():
        pytest.skip('Optional installed variable face')
    regular=font_face('NotoSansSC-VF.ttf',28,400)
    bold=font_face('NotoSansSC-VF.ttf',28,700)
    assert bytes(regular.getmask('文字'))!=bytes(bold.getmask('文字'))


def test_small_ink_symbol_and_trailing_space_are_not_lost():
    if not Path('C:/Windows/Fonts/msyh.ttc').exists():
        pytest.skip('Installed Windows font required')
    alpha,_=glyph_raster('× ','msyh.ttc',30)
    fit=_fit_font('× ',alpha>127,'Microsoft YaHei',alpha.astype(float)/255)
    assert fit is not None
    assert fit['fontMaskSimilarity']>.8


def test_qa_does_not_label_bad_font_fit_as_visual_success():
    layout={'slide':{'width':500,'height':200},'elements':[{
        'id':'t','owner':'editable_text','style':{'fontSize':28,'letterSpacing':-3.75},
        'metadata':{'fontMeasurement':{'fontMaskSimilarity':.43,'fontAlphaError':.39}}}]}
    quality=reconstruction_quality(layout,{}, {})
    assert quality['fontApproximationWarningCount']==1
    assert quality['aggressiveTrackingCount']==1
    assert any(i['problem']=='approximateFontNeedsVisualReview' for i in quality['issuesAfter'])
