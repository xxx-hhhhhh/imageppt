"""Generic object tests; no user slide coordinates or provider fixtures."""
import cv2
import numpy as np
from app.services.reconstruction.contour_objects import plan_contour_objects
from app.services.reconstruction.exclusive_ownership import box_of
from app.services.reconstruction.visual_assets import refine_edge_alpha
from app.utils import image_io


def test_broad_segment_cannot_absorb_independent_textured_photos(tmp_path):
    image=np.full((260,700,3),250,np.uint8)
    rng=np.random.default_rng(12)
    boxes=[]
    for x in (60,210,360,510):
        image[80:160,x:x+90]=rng.integers(20,145,(80,90,3),dtype=np.uint8)
        boxes.append((x,80,x+90,160))
    assets=tmp_path/'assets'
    assets.mkdir()
    image_io.imwrite(assets/'broad.png',cv2.cvtColor(image[50:200,30:640],cv2.COLOR_BGR2BGRA))
    planned,_=plan_contour_objects(image,[{'id':'broad','bbox':[30,50,640,200],
        'alphaCrop':'broad.png'}],assets,[],box_of)
    photos=[p for p in planned if p['evidence']['visualAssetPolicy']=='source_verified_photographic_rectangle']
    assert len(photos)==4
    for x,y,bx,by in boxes:
        assert sum(p['box'][0]<=x and p['box'][1]<=y and p['box'][2]>=bx and p['box'][3]>=by for p in photos)==1
    broad=next(p for p in planned if p['evidence']['contourProposalId']=='broad')
    assert broad['panel'] and planned.index(broad)>max(planned.index(p) for p in photos)


def test_edge_alpha_unmixes_old_paper_without_touching_interior():
    paper=np.full((20,20,3),245.,float)
    rgb=paper.copy()
    alpha=np.zeros((20,20),float)
    alpha[5:15,5:15]=1
    rgb[5:15,5:15]=[40,120,190]
    rgb[5,5:15]=paper[5,5:15]*.6+np.array([40,120,190])*.4
    contrast=np.max(np.abs(rgb-paper),axis=2)
    refined,count=refine_edge_alpha(rgb,paper,alpha,contrast)
    assert count>=8
    assert np.allclose(refined[5,6:14],.4)
    assert refined[10,10]==1
    assert np.array_equal(refined[alpha==0],alpha[alpha==0])


def test_disconnected_photo_strip_is_not_promoted_to_a_carrier(tmp_path, monkeypatch):
    from app.services.reconstruction import contour_objects
    image = np.full((200, 500, 3), 250, np.uint8)
    mask = np.zeros((90, 400), np.uint8)
    mask[:, 10:90] = 255
    mask[:, 140:220] = 255
    mask[:, 270:350] = 255
    image_io.imwrite(tmp_path/'strip.png', np.dstack((image[50:140, 30:430], mask)))
    monkeypatch.setattr(contour_objects, 'carrier_contours', lambda _: [])
    monkeypatch.setattr(contour_objects, 'photographic_rectangles', lambda _: [
        (40, 50, 120, 140), (170, 50, 250, 140), (300, 50, 380, 140)])
    planned, _ = plan_contour_objects(image, [{'id': 'strip', 'bbox': [30, 50, 430, 140],
                                               'alphaCrop': 'strip.png'}], tmp_path, [], box_of)
    strip = next(p for p in planned if p['evidence']['contourProposalId'] == 'strip')
    assert not strip['panel']
    assert strip['evidence']['visualAssetPolicy'] == 'retained_disconnected_visual'
    assert np.array_equal(strip['rgba'][:, :, 3], mask)
