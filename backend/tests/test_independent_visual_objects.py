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
    photos = [p for p in planned if p['evidence']['visualAssetPolicy'] == 'source_verified_photographic_rectangle']
    assert len(photos) == 3
    assert all(planned.index(p) < planned.index(strip) for p in photos)


def test_convex_repair_needs_independent_edges_on_every_side():
    from app.services.reconstruction.contour_objects import _supported_convex_boundary
    # A near-rectangular contour with a notch may close only if the source
    # independently shows the missing top edge; no user image coordinates.
    contour = np.array([[[20, 20]], [[90, 20]], [[90, 30]], [[120, 30]],
                        [[120, 20]], [[219, 20]], [[219, 139]], [[20, 139]]], np.int32)
    edges = np.zeros((160, 240), np.uint8)
    cv2.rectangle(edges, (20, 20), (219, 139), 255, 1)
    distance = cv2.distanceTransform(np.uint8(edges == 0), cv2.DIST_L2, 5)
    recovered = _supported_convex_boundary(contour, distance, (20, 20, 200, 120))
    assert recovered is not None
    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [recovered], -1, 1, -1)
    assert mask[21, 100] == 1  # A diagonal hull across a title is not a full frame.
    # An open U cannot be turned into a plate just from its bounding window.
    edges[20, 21:219] = 0
    distance = cv2.distanceTransform(np.uint8(edges == 0), cv2.DIST_L2, 5)
    assert _supported_convex_boundary(contour, distance, (20, 20, 200, 120)) is None


def test_dark_binary_diagram_is_not_a_pale_carrier():
    from app.services.reconstruction.contour_objects import carrier_contours
    image = np.full((260, 420, 3), 250, np.uint8)
    image[70:190, 90:320] = 10
    cv2.line(image, (130, 150), (280, 100), (255, 255, 255), 15)
    candidates = carrier_contours(image)
    assert not any(x <= 95 and y <= 75 and bx >= 315 and by >= 185 for (x, y, bx, by), _ in candidates)


def test_photo_owns_pale_shadow_but_not_neighboring_blue_arrow(tmp_path):
    image = np.full((230, 400, 3), 250, np.uint8)
    image[76:164, 56:154] = 238
    rng = np.random.default_rng(5)
    image[80:160, 60:150] = rng.integers(20, 140, (80, 90, 3), dtype=np.uint8)
    image[114:126, 155:165] = [230, 130, 10]
    planned, _ = plan_contour_objects(image, [], tmp_path, [], box_of)
    photo = next(p for p in planned if p['evidence']['visualAssetPolicy']=='source_verified_photographic_rectangle')
    x, y, _, _ = photo['box']
    assert photo['rgba'][120-y, 153-x, 3] == 255
    assert photo['rgba'][120-y, 155-x, 3] == 0
