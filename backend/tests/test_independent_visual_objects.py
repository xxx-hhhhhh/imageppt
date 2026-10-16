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
