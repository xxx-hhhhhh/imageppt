"""Object-level regressions: silhouettes, complete surfaces and OCR boundaries."""
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
from app.services.reconstruction.contour_objects import plan_contour_objects
from app.services.reconstruction.editable_text import restore_text_candidate
from app.services.reconstruction.exclusive_ownership import (
    box_of,
    build_exclusive_scene,
)
from app.services.reconstruction.preservation_checks import inspect_preservation
from app.services.reconstruction.visual_assets import prepare_visual_seed
from app.services.visual_qa.analyzer import render_preview
from app.utils import image_io
from PIL import Image, ImageDraw, ImageFont


def test_opaque_bbox_is_not_evidence_for_a_square_icon():
    image = np.full((140,180,3),245,np.uint8)
    cv2.circle(image,(80,70),35,(30,140,50),-1,cv2.LINE_AA)
    # Internal white mark must stay opaque; only exterior is transparent.
    cv2.rectangle(image,(75,60),(85,80),(245,245,245),-1)
    box,rgba,meta = prepare_visual_seed(image,(40,30,120,110),np.full((80,80),255,np.uint8))
    assert rgba[0,0,3]==0
    assert rgba[70-box[1],80-box[0],3]==255
    assert np.any((rgba[:,:,3]>0)&(rgba[:,:,3]<255))
    assert meta['visualAssetPolicy']=='supersampled_colour_decontaminated_matte'
    # Moving onto dark paper should not carry the original light square.
    alpha = rgba[:,:,3:4].astype(float)/255
    moved = rgba[:,:,:3]*alpha+np.array([25,35,45])*(1-alpha)
    assert np.array_equal(moved[0,0],[25,35,45])


def test_dense_rounded_rectangle_keeps_round_corners():
    image = np.full((130,240,3),255,np.uint8)
    cv2.rectangle(image,(40,30),(200,90),(170,90,30),-1)
    cv2.rectangle(image,(30,40),(210,80),(170,90,30),-1)
    for point in ((40,40),(200,40),(40,80),(200,80)):
        cv2.circle(image,point,10,(170,90,30),-1,cv2.LINE_AA)
    box,rgba,_ = prepare_visual_seed(image,(28,28,213,93),np.full((65,185),255,np.uint8))
    assert rgba[0,0,3]==0
    assert rgba[60-box[1],100-box[0],3]==255
    assert np.mean(rgba[:,:,3]>0)>.85


def test_thin_ridge_proposal_recovers_filled_illustration():
    image=np.full((120,240,3),255,np.uint8)
    points=np.array([[25,85],[65,35],[110,65],[160,25],[215,85]],np.int32)
    cv2.fillPoly(image,[points],(215,190,160),cv2.LINE_AA)
    raw=np.zeros((70,205),np.uint8)
    cv2.polylines(raw,[points-[18,20]],False,255,2)
    box,rgba,meta=prepare_visual_seed(image,(18,20,223,90),raw)
    assert rgba[72-box[1],65-box[0],3]==255
    assert meta.get('recoveredVisualPixelCount',0)>100
    assert rgba[0,0,3]==0


def test_overlapping_photo_contours_are_one_complete_asset(tmp_path):
    image = np.full((400,600,3),255,np.uint8)
    image[70:150,70:190] = (130,60,20)
    image[80:160,60:180] = (50,160,30)
    assets = tmp_path/'assets'
    assets.mkdir()
    segments = []
    for name,box in [('stack',(58,68,193,163)),('inner',(60,80,180,160))]:
        x,y,bx,by=box
        path=assets/f'{name}.png'
        image_io.imwrite(path,cv2.cvtColor(image[y:by,x:bx],cv2.COLOR_BGR2BGRA))
        segments.append({'id':name,'bbox':list(box),'alphaCrop':str(path)})
    planned,audit = plan_contour_objects(image,segments,assets,[],box_of)
    covering=[p for p in planned if p['box'][0]<=70 and p['box'][2]>=180]
    assert len(covering)==1
    assert audit['duplicateContoursMerged']>=1
    rgba=covering[0]['rgba']
    assert rgba[70-covering[0]['box'][1],185-covering[0]['box'][0],3]>0


def test_raster_text_cannot_be_cut_between_two_objects(tmp_path):
    image=np.full((150,250,3),240,np.uint8)
    cv2.line(image,(45,70),(165,70),(30,30,30),4)
    assets=tmp_path/'assets'
    assets.mkdir()
    image_io.imwrite(assets/'bad.png',cv2.cvtColor(image[30:90,30:110],cv2.COLOR_BGR2BGRA))
    decision={'bbox':[40,60,170,80],'decision':'movable_image'}
    planned,audit=plan_contour_objects(image,[{'id':'bad','bbox':[30,30,110,90],
        'alphaCrop':'bad.png'}],assets,[decision],box_of)
    assert audit['rasterTextCrossingProposalsRejected']>=1
    assert not any(p['evidence']['contourProposalId']=='bad' for p in planned)


def test_card_under_icon_is_complete_when_icon_moves(tmp_path):
    image=np.full((200,360,3),255,np.uint8)
    cv2.rectangle(image,(25,25),(335,175),(240,225,210),-1)
    cv2.circle(image,(85,90),25,(30,120,50),-1,cv2.LINE_AA)
    output=tmp_path/'out'
    assets=output/'assets'
    assets.mkdir(parents=True)
    source=tmp_path/'source.png'
    image_io.imwrite(source,image)
    mask=np.zeros((54,54),np.uint8)
    cv2.circle(mask,(27,27),25,255,-1)
    image_io.imwrite(assets/'icon.png',np.dstack((image[63:117,58:112],mask)))
    scene=build_exclusive_scene(source,{'elements':[]},{},[{'id':'icon','bbox':[58,63,112,117],
        'alphaCrop':'icon.png'}],output,'fixture',1)
    cards=[e for e in scene['elements'] if e.get('metadata',{}).get('completedChildSurfacePixels',0)>0]
    assert cards
    card=max(cards,key=lambda e:e['width']*e['height'])
    rgba=image_io.imread(assets/card['src'].split('/')[-1],-1)
    assert rgba[90-card['y'],85-card['x'],3]==255
    assert np.max(np.abs(rgba[90-card['y'],85-card['x'],:3].astype(int)-[240,225,210]))<=1
    local=deepcopy(scene)
    for e in local['elements']:
        if e.get('src'):
            e['src']=str(output/('backgrounds' if e['type']=='background' else 'assets')/e['src'].split('/')[-1])
    preview=output/'preview.png'
    render_preview(output/'backgrounds/page_1.png',local,preview)
    check=inspect_preservation(source,scene,output,preview)
    assert check['unownedSourcePixels']==check['multiplyOwnedSourcePixels']==0
    assert check['unexpectedVisualChangedPixels']==0


def test_gradient_glyph_feet_do_not_remain_inside_image():
    face=Path('C:/Windows/Fonts/arial.ttf')
    if not face.is_file():
        pytest.skip('Windows font fixture')
    glyph=Image.new('L',(280,110))
    draw=ImageDraw.Draw(glyph)
    font=ImageFont.truetype(str(face),42)
    draw.text((24,16),'7082',font=font,fill=255)
    x,y,bx,by=draw.textbbox((24,16),'7082',font=font)
    alpha=np.asarray(glyph).astype(float)/255
    paper=np.full((110,280,3),(10,10,180),np.uint8)
    colour=np.full(paper.shape,(210,245,255),np.uint8)
    colour[by-6:]=(85,210,255)
    image=np.rint(paper*(1-alpha[:,:,None])+colour*alpha[:,:,None]).astype(np.uint8)
    item={'id':'gradient','type':'text','text':'7082','confidence':.99,
          'x':x,'y':y,'width':bx-x,'height':by-y,'style':{'fontFamily':'Arial','color':'#FFF5D2'}}
    result=restore_text_candidate(image,item)
    assert result is not None
    _,mask,fill=result
    assert np.all(mask[alpha>.5])
    assert np.max(np.abs(fill[mask].astype(int)-paper[mask].astype(int)))<6
