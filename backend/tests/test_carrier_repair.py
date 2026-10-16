import numpy as np
from app.services.reconstruction.carrier_repair import (
    complete_local_floor,
    unowned_contour_fringe,
)


def fixture():
    a = np.full((120, 240, 4), 255, np.uint8)
    a[:, :, :3] = [252, 244, 235]
    a[20:55, 70:150, 3] = 0
    support = np.ones(a.shape[:2], bool)
    owner = np.zeros_like(support)
    owner[20:55, 70:150] = True
    return a, support, owner


def test_verified_child_hole_only_and_existing_pixels_unchanged():
    a, support, owner = fixture()
    before = a.copy()
    assert complete_local_floor(a, support, owner) == 2800
    assert np.array_equal(a[~owner], before[~owner])
    assert np.all(a[owner, :3] == [252, 244, 235])
    assert np.all(a[owner, 3] == 255)


def test_no_owner_no_completion():
    a, support, owner = fixture()
    before = a.copy()
    assert complete_local_floor(a, support, ~owner) == 0
    assert np.array_equal(a, before)


def test_outside_silhouette_is_never_filled():
    a, support, owner = fixture()
    support[owner] = False
    assert complete_local_floor(a, support, owner) == 0


def test_texture_or_mixed_dark_visual_is_not_paper():
    a, support, owner = fixture()
    a[:, :, :3] = np.random.default_rng(1).integers(30, 255, a[:, :, :3].shape, dtype=np.uint8)
    before = a.copy()
    assert complete_local_floor(a, support, owner) == 0
    assert np.array_equal(a, before)


def test_thin_colored_border_does_not_block_pale_floor_but_stays_intact():
    a, support, owner = fixture()
    a[0, :, :3] = [255, 140, 70]
    before = a.copy()
    assert complete_local_floor(a, support, owner) == 2800
    assert np.array_equal(a[~owner], before[~owner])


def test_mottled_light_surface_cannot_be_claimed_as_smooth_paper():
    a, support, owner = fixture()
    a[:, :, :3] = np.random.default_rng(4).integers(180, 255, a[:, :, :3].shape, dtype=np.uint8)
    before = a.copy()
    assert complete_local_floor(a, support, owner) == 0
    assert np.array_equal(a, before)


def test_fringe_keeps_curved_silhouette_and_never_takes_existing_owner():
    import cv2
    support = np.zeros((80, 80), np.uint8)
    cv2.circle(support, (40, 40), 20, 1, -1)
    owned = np.zeros_like(support, dtype=bool)
    owned[:, 40:] = True
    fringe = unowned_contour_fringe(support > 0, owned)
    assert fringe.any()
    assert not fringe[owned].any()
    assert not fringe[support > 0].any()
    assert not fringe[17, 17]  # A padded rectangular crop would claim this corner.


def test_duplicate_carrier_merges_measured_outline_not_only_semantic_flag(tmp_path, monkeypatch):
    from app.services.reconstruction import contour_objects
    from app.services.reconstruction.exclusive_ownership import box_of
    from app.utils import image_io

    image = np.full((300, 500, 3), 240, np.uint8)
    rgba = np.full((200, 400, 4), 255, np.uint8)
    rgba[:, 340:, 3] = 0  # SAM has an incomplete right edge in its search window.
    image_io.imwrite(tmp_path/'sam.png', rgba)
    monkeypatch.setattr(contour_objects, 'carrier_contours', lambda _: [
        ((50, 50, 448, 250), np.full((200, 398), 255, np.uint8))])
    planned, _ = contour_objects.plan_contour_objects(image, [
        {'id': 'sam', 'bbox': [50, 50, 450, 250], 'alphaCrop': 'sam.png'}], tmp_path, [], box_of)
    carrier = next(p for p in planned if p['panel'])
    assert carrier['rgba'][100, 390, 3] == 255
