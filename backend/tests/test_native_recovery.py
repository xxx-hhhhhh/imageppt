"""Generic text/matte regressions; no real-page coordinates or provider fakes."""
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
from app.services.reconstruction.editable_text import restore_text_candidate
from app.services.reconstruction.exclusive_ownership import (
    audit_raster_text,
    build_exclusive_scene,
)
from app.services.reconstruction.visual_assets import prepare_visual_seed
from app.utils import image_io
from PIL import Image, ImageDraw, ImageFont


def make_text(tmp_path, text='编辑文字', size=24, *, jpeg=False):
    face = Path('C:/Windows/Fonts/msyhbd.ttc')
    if not face.exists():
        pytest.skip('Windows font fixture')
    yy, xx = np.mgrid[:100, :340]
    paper = np.stack((240-xx*.018, 245-yy*.025, 253-xx*.009), axis=-1).astype(np.uint8)
    image = Image.fromarray(paper)
    font = ImageFont.truetype(str(face), size)
    draw = ImageDraw.Draw(image)
    draw.text((25, 25), text, font=font, fill='#17365D')
    box = draw.textbbox((25, 25), text, font=font)
    source = tmp_path / ('input.jpg' if jpeg else 'input.png')
    image.save(source, quality=88)
    item = {'id':'ocr_001', 'type':'text', 'text':text, 'confidence':.99,
            'x':box[0], 'y':box[1], 'width':box[2]-box[0], 'height':box[3]-box[1],
            'style':{'fontFamily':'Microsoft YaHei', 'color':'#17365D'}}
    return source, item


@pytest.mark.parametrize('jpeg', [False, True])
def test_gradient_chinese_text_native_and_old_ink_audit_independent(tmp_path, jpeg):
    source, item = make_text(tmp_path, jpeg=jpeg)
    output = tmp_path / 'candidate'
    layout = build_exclusive_scene(source, {'elements':[item]}, {}, [], output, 'abc', 1)
    native = next(e for e in layout['elements'] if e.get('owner') == 'editable_text')
    assert native['text'] == item['text']
    assert native['metadata']['nativeTextMeasured']
    assert audit_raster_text(source,layout,output)['rasterNativeDuplicateTextCount'] == 0
    # Inject the old whole screenshot as a NEW top raster layer: auditing must
    # detect real old ink, even though every text owner already has a mask.
    image_io.imwrite(output / 'assets/old.png', image_io.imread(source))
    corrupt = deepcopy(layout)
    corrupt['elements'].append({'id':'old_source', 'type':'image', 'src':str(output/'assets/old.png'),
                               'x':0,'y':0,'width':340,'height':100,'zIndex':999})
    assert audit_raster_text(source,corrupt,output)['rasterNativeDuplicateTextCount'] == 1


def test_only_glyph_local_pixels_change_not_whole_card(tmp_path):
    source, item = make_text(tmp_path, jpeg=True)
    image = image_io.imread(source)
    native, mask, paper = restore_text_candidate(image,item)
    assert native['owner'] == 'editable_text'
    cleaned = image.copy()
    cleaned[mask] = paper[mask]
    assert np.array_equal(cleaned[~mask], image[~mask])
    x,y,w,h = [int(item[k]) for k in ('x','y','width','height')]
    # Authorized support never reaches another module or a page-wide strip.
    assert not mask[:y-4].any() and not mask[y+h+4:].any()
    assert not mask[:,:x-4].any() and not mask[:,x+w+4:].any()
    assert mask.sum() < image.shape[0]*image.shape[1]*.12
    assert native['metadata']['textErasureEvidence']['glyphHaloRadius'] == 3


def test_unreliable_or_logo_text_is_not_erased(tmp_path):
    source,item = make_text(tmp_path)
    image = image_io.imread(source)
    before = image.copy()
    item['confidence'] = .5
    assert restore_text_candidate(image,item) is None
    item['confidence'] = .99
    assert restore_text_candidate(image,item,{'role':'logo'}) is None
    assert np.array_equal(image,before)


def test_chart_crop_keeps_dense_mask_pinholes(tmp_path):
    image = np.full((100,150,3),235,np.uint8)
    image[30:60,40:70] = (180,50,20)
    mask = np.full((80,130),255,np.uint8)
    mask[25:28,35:38] = 0
    box,rgba,metadata = prepare_visual_seed(image,(10,10,140,90),mask)
    assert box == (10,10,140,90)
    assert metadata['visualAssetPolicy'] == 'coherent_rectangular_crop'
    assert np.all(rgba[:,:,3] == 255)
    assert np.array_equal(rgba[:,:,:3],image[10:90,10:140])


def test_round_visual_has_real_alpha_and_no_paper_fringe():
    image = np.full((100,100,3),255,np.uint8)
    cv2.circle(image,(50,50),22,(90,45,20),-1,cv2.LINE_AA)
    mask = np.zeros((50,50),np.uint8)
    cv2.circle(mask,(25,25),22,255,-1)
    _,rgba,metadata = prepare_visual_seed(image,(25,25,75,75),mask)
    assert metadata['visualAssetPolicy'] == 'supersampled_colour_decontaminated_matte'
    assert rgba[0,0,3] == 0
    alpha = rgba[:,:,3]
    assert np.any((alpha>0)&(alpha<255))
    band = (alpha>0)&(alpha<255)
    assert not np.any(np.all(rgba[:,:,:3][band]>250,axis=1))


def test_raster_only_quality_failure_is_actionable_http409(monkeypatch,tmp_path):
    import app.main as api
    from app.models.project_store import ProjectStore
    from app.services.reconstruction.pipeline import ReconstructionQualityError
    from fastapi.testclient import TestClient
    store = ProjectStore(tmp_path/'outputs')
    record = store.create('quality gate')
    store.add_image(record['id'], {'path':'unused.png'})
    monkeypatch.setattr(api,'store',store)
    class FailedPipeline:
        def __init__(self,store):
            pass
        def analyze_project(self,*args):
            raise ReconstructionQualityError('未生成原生文本框；候选已保留，原结果未覆盖。')
    monkeypatch.setattr(api,'ReconstructionPipeline',FailedPipeline)
    response = TestClient(api.app).post(f"/api/projects/{record['id']}/analyze")
    assert response.status_code == 409
    assert response.json()['detail']['code'] == 'EDITABLE_TEXT_RECONSTRUCTION_INCOMPLETE'
    assert not (store.root/record['id']/'slides').exists()
