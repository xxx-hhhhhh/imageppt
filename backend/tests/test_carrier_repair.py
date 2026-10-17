import numpy as np
from app.services.reconstruction.carrier_repair import (
    complete_local_floor,
    inherit_carrier_groups,
    separate_carrier_surface,
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


def test_large_hole_in_nonlinear_paper_is_completed_without_repainting_outline():
    import cv2
    h, w = 160, 260
    support = np.zeros((h, w), np.uint8)
    cv2.ellipse(support, (130, 80), (128, 78), 0, 0, 360, 1, -1)
    yy, xx = np.indices((h, w))
    rgb = np.stack((230 + 15*np.sin(xx/30), 239 + 8*np.cos(yy/24), np.full((h, w), 253)), axis=2)
    a = np.dstack((rgb.astype(np.uint8), support*255))
    owner = (xx > 40) & (xx < 220) & (yy > 25) & (yy < 135)
    a[owner, 3] = 0
    before = a.copy()
    report = {}
    expected = owner & support.astype(bool)
    assert complete_local_floor(a, support > 0, owner, diagnostics=report) == expected.sum()
    assert report['approximatePaperPixels'] > 0
    assert report['unresolvedPixels'] == 0
    assert np.array_equal(a[~expected], before[~expected])
    assert a[expected, :3].min() >= 210
    # Large interiors must be smoothly bridged, not filled with noisy source patches.
    center = a[45:115, 65:195, :3].astype(float)
    assert np.abs(np.diff(center, axis=0)).max() <= 2
    assert np.abs(np.diff(center, axis=1)).max() <= 2


def test_sparse_paper_strip_excludes_saturated_ink_and_hidden_rgb():
    a = np.zeros((50, 200, 4), np.uint8)
    a[:, :, :3] = [255, 0, 255]  # Hidden source RGB is not colour evidence.
    a[5:10, 5:195] = [249, 244, 236, 255]
    a[20:28, 20:80] = [25, 95, 210, 255]
    support = np.ones(a.shape[:2], bool)
    owner = a[:, :, 3] == 0
    before = a.copy()
    report = {}
    assert complete_local_floor(a, support, owner, diagnostics=report) == owner.sum()
    assert report['approximatePaperPixels'] == owner.sum()
    assert np.all(a[owner, :3] == [249, 244, 236])
    assert np.array_equal(a[~owner], before[~owner])


def test_disconnected_silhouette_without_samples_is_not_filled_from_neighbor():
    a = np.zeros((80, 180, 4), np.uint8)
    a[:30, :60] = [250, 240, 230, 255]
    support = np.zeros(a.shape[:2], bool)
    support[:60, :60] = True
    support[:60, 120:] = True
    owner = support & (a[:, :, 3] == 0)
    report = {}
    complete_local_floor(a, support, owner, diagnostics=report)
    assert np.all(a[:60, 120:, 3] == 0)
    assert report['unresolvedPixels'] == 3600


def test_fallback_uses_only_allowed_source_samples_and_preserves_unowned_holes():
    a = np.zeros((60, 200, 4), np.uint8)
    a[2:10, :90] = [240, 245, 250, 255]
    a[2:10, 110:] = [180, 190, 200, 255]  # Excluded synthetic/foreign substrate.
    support = np.ones(a.shape[:2], bool)
    owner = a[:, :, 3] == 0
    owner[30:40, 80:120] = False
    samples = np.zeros_like(owner)
    samples[:, :100] = True
    before = a.copy()
    report = {}
    complete_local_floor(a, support, owner, sample_mask=samples, diagnostics=report)
    assert report['approximatePaperPixels'] > 0
    assert np.all(a[owner, :3] == [240, 245, 250])
    assert np.array_equal(a[~owner], before[~owner])


def test_one_pixel_paper_fringe_can_supply_large_occluded_surface():
    a = np.zeros((40, 100, 4), np.uint8)
    a[:, :, :3] = [0, 255, 0]
    a[0, 10:40] = [249, 251, 253, 255]
    support = np.ones(a.shape[:2], bool)
    owner = a[:, :, 3] == 0
    report = {}
    assert complete_local_floor(a, support, owner, diagnostics=report) == owner.sum()
    assert report['approximatePaperPixels'] == owner.sum()
    assert np.all(a[owner, :3] == [249, 251, 253])


def test_isolated_pale_specks_are_not_enough_evidence():
    a = np.zeros((50, 100, 4), np.uint8)
    a[::5, ::5] = [249, 251, 253, 255]
    before = a.copy()
    support = np.ones(a.shape[:2], bool)
    assert complete_local_floor(a, support, a[:, :, 3] == 0) == 0
    assert np.array_equal(a, before)


def test_surface_reconstruction_separates_artwork_before_repainting_paper():
    import cv2
    support = np.zeros((180, 320), np.uint8)
    cv2.rectangle(support, (2, 2), (317, 177), 1, -1)
    rgb = np.full((180, 320, 3), [250, 248, 245], np.uint8)
    rgb[40:140, 40:130] = [240, 220, 200]
    cv2.line(rgb, (180, 60), (270, 120), (10, 80, 180), 5)
    original = rgb.copy()
    rebuilt, details, interior = separate_carrier_surface(rgb, support > 0, support > 0)
    detail = np.logical_or.reduce(details)
    strong = np.max(np.abs(original.astype(float)-rebuilt), axis=2)>6
    assert np.all(detail[strong])
    assert detail[80, 80] and detail[90, 225]
    assert np.all(rebuilt[interior] == [250, 248, 245])
    assert np.array_equal(rebuilt[~interior], original[~interior])
    assert np.array_equal(rgb, original)


def test_unknown_rim_or_missing_samples_cannot_authorize_surface_replacement():
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (140, 260, 3), dtype=np.uint8)
    support = np.ones(rgb.shape[:2], bool)
    assert separate_carrier_surface(rgb, support, support) is None
    rgb[:] = 250
    assert separate_carrier_surface(rgb, support, np.zeros_like(support)) is None


def test_surface_does_not_duplicate_details_already_owned_by_children():
    rgb = np.full((180, 320, 3), 250, np.uint8)
    rgb[50:130, 70:250] = [50, 100, 200]
    support = np.ones(rgb.shape[:2], bool)
    available = support.copy()
    available[45:135, 65:255] = False
    rebuilt, details, _ = separate_carrier_surface(rgb, support, available)
    assert details == []
    assert np.all(rebuilt[60:120, 80:240] == 250)


def test_short_banner_also_separates_icon_and_returns_clean_paper():
    rgb = np.full((64, 400, 3), 248, np.uint8)
    rgb[18:46, 25:55] = [20, 90, 180]
    support = np.ones(rgb.shape[:2], bool)
    rebuilt, details, interior = separate_carrier_surface(rgb, support, support)
    assert np.all(rebuilt[interior] == 248)
    assert np.logical_or.reduce(details)[30, 40]


def test_rebuilt_carrier_follows_agreeing_detail_module_despite_bbox_rim():
    carrier = {'id': 'plate', 'x': 0, 'y': 0, 'width': 300, 'height': 200,
               'metadata': {'carrierSurfaceRebuilt': True, 'carrierDetailIds': ['a', 'b']}}
    a = {'id': 'a', 'x': 5, 'y': 5, 'width': 140, 'height': 190, 'groupId': 'module'}
    b = {'id': 'b', 'x': 155, 'y': 5, 'width': 140, 'height': 190, 'groupId': 'module'}
    inherit_carrier_groups([carrier, a, b])
    assert carrier['groupId'] == carrier['parent'] == 'module'


def test_carrier_cannot_take_conflicting_modules_or_one_small_icon_module():
    for b_group, size in [('other', 190), ('module', 20)]:
        carrier = {'id': 'plate', 'x': 0, 'y': 0, 'width': 300, 'height': 200,
                   'metadata': {'carrierSurfaceRebuilt': True, 'carrierDetailIds': ['a', 'b']}}
        a = {'id': 'a', 'x': 5, 'y': 5, 'width': 20, 'height': size, 'groupId': 'module'}
        b = {'id': 'b', 'x': 35, 'y': 5, 'width': 20, 'height': size, 'groupId': b_group}
        inherit_carrier_groups([carrier, a, b])
        assert not carrier.get('groupId')
