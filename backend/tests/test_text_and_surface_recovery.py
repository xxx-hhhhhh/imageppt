from pathlib import Path

import numpy as np
import pytest
from app.services.reconstruction.contour_objects import (
    _visible_surface,
    complete_carrier_holes,
)
from app.services.reconstruction.editable_text import _fit_font
from app.services.reconstruction.text_cohorts import prepare_text_lines
from app.services.typography.font_raster import glyph_raster


def test_middle_dot_is_measured_by_ink_not_line_height():
    if not Path('C:/Windows/Fonts/msyh.ttc').is_file():
        pytest.skip('Windows font fixture')
    alpha, _ = glyph_raster('·', 'msyh.ttc', 50)
    fit = _fit_font('·', alpha > 40, 'Microsoft YaHei', alpha.astype(float)/255)
    assert fit is not None
    assert fit['fontMaskSimilarity'] >= .35
    assert fit['fontDimensionError'] <= .19


def test_carrier_recovers_large_occluded_gradient_without_touching_visible_pixels():
    yy,xx = np.mgrid[:140,:280]
    paper = np.dstack((240+xx*.03,225+yy*.04,210+xx*.02)).astype(np.uint8)
    labels = np.zeros((140,280),np.int32)
    labels[30:100,60:220] = 1
    source = paper.copy()
    source[labels>0] = (60,100,30)
    rgba = np.dstack((source,np.uint8(labels==0)*255))
    before = rgba.copy()
    filled, evidence = complete_carrier_holes(source,rgba,(0,0,280,140),
        np.ones(labels.shape,bool),labels,np.zeros(labels.shape,bool))
    assert filled[labels>0].all()
    assert evidence['unresolvedChildSurfacePixels']==0
    assert np.max(np.abs(rgba[:,:,:3][labels>0].astype(int)-paper[labels>0]))<=2
    assert np.array_equal(rgba[labels==0],before[labels==0])


def test_surface_rejects_textured_or_unsupported_extrapolation():
    rng=np.random.default_rng(17)
    patch=rng.integers(0,256,(70,120,3),dtype=np.uint8)
    assert _visible_surface(patch,np.ones((70,120),bool)) is None
    patch[:]=240
    support=np.zeros((70,120),bool)
    support[:10]=True
    assert _visible_surface(patch,support) is None


def test_font_cohort_uses_majority_only_with_individual_fit(monkeypatch):
    import app.services.reconstruction.text_cohorts as module
    def restore(source,item,hints,evidence):
        family=(hints or {}).get('_cohortFontFamily') or ('SimSun' if item['id']=='a' else 'Microsoft YaHei')
        fit={'fontAlphaError':.18,'fontMaskSimilarity':.7,'fontDimensionError':0}
        return [({'text':item['text'],'style':{'fontFamily':family},'metadata':{'fontMeasurement':fit}},None,None)]
    monkeypatch.setattr(module,'restore_text_candidates',restore)
    items=[{'id':key,'text':'标签测试','x':i*110,'y':20,'width':80,'height':25,
            'style':{'fontClass':'bold-sans'}} for i,key in enumerate('abc')]
    result=prepare_text_lines(None,items,{})
    assert {r[0][0][0]['style']['fontFamily'] for r in result.values()}=={'Microsoft YaHei'}
    # Spatially unrelated text must not get grouped merely for sharing a role.
    items[-1]['x']=900
    result=prepare_text_lines(None,items,{})
    assert result['a'][0][0][0]['style']['fontFamily']=='SimSun'
