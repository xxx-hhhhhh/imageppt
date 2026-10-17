import numpy as np
import pytest
from app.services.reconstruction.preservation_checks import inspect_preservation
from app.utils import image_io


def audit_fixture(tmp_path, delta=3, claimed=True, owned=True, new_text=False, foreign_overlay=False):
    source = np.full((32, 48, 3), 250, np.uint8)
    image_io.imwrite(tmp_path/'source.png', source)
    (tmp_path/'assets').mkdir()
    (tmp_path/'backgrounds').mkdir()
    image_io.imwrite(tmp_path/'backgrounds/page_1.png', source)
    asset = np.dstack((source-delta, np.full((32, 48), 255, np.uint8)))
    image_io.imwrite(tmp_path/'assets/plate.png', asset)
    image_io.imwrite(tmp_path/'preview.png', asset[:, :, :3])
    mask = np.zeros_like(asset)
    mask[:, :, 3] = 255
    image_io.imwrite(tmp_path/'approx.png', mask)
    if not owned:
        mask[0, 0, 3] = 0
    image_io.imwrite(tmp_path/'owner.png', mask)
    metadata = {'sourceMaskPath': 'owner.png', 'sourceMaskBBox': [0, 0, 48, 32]}
    if claimed:
        metadata['surfaceApproximationMask'] = 'approx.png'
    plate = {'id': 'plate', 'type': 'image', 'x': 0, 'y': 0, 'width': 48, 'height': 32,
             'zIndex': 1, 'src': '/media/assets/test/plate.png', 'metadata': metadata}
    background = {**plate, 'id': 'background', 'type': 'background', 'zIndex': 0,
                  'src': '/media/backgrounds/test/page_1.png', 'metadata': {}}
    scene = {'slide': {'width': 48, 'height': 32}, 'elements': [background, plate],
             'metadata': {'ownershipAudit': {'nodes': [
                 {'owner': 'movable_image', 'mask': 'owner.png', 'sourceBBox': [0, 0, 48, 32]}]}}}
    if new_text:
        from copy import deepcopy

        from app.services.visual_qa.analyzer import render_preview
        scene['elements'].append({'id': 'text', 'type': 'text', 'text': 'A', 'x': 10, 'y': 8,
            'width': 25, 'height': 20, 'zIndex': 3, 'style': {'fontSize': 16, 'color': '#000000'}})
        local = deepcopy(scene)
        local['elements'][0]['src'] = str(tmp_path/'backgrounds/page_1.png')
        local['elements'][1]['src'] = str(tmp_path/'assets/plate.png')
        render_preview(tmp_path/'backgrounds/page_1.png', local, tmp_path/'preview.png')
    if foreign_overlay:
        preview = image_io.imread(tmp_path/'preview.png')
        preview[0, 0] = 0
        image_io.imwrite(tmp_path/'preview.png', preview)
    return inspect_preservation(tmp_path/'source.png', scene, tmp_path, tmp_path/'preview.png')


def test_only_declared_small_surface_changes_are_counted_separately(tmp_path):
    result = audit_fixture(tmp_path)
    assert result['boundedCarrierSurfacePixels'] == 32*48
    assert result['unexpectedVisualChangedPixels'] == 0


def test_undeclared_changes_are_still_unexpected(tmp_path):
    result = audit_fixture(tmp_path, claimed=False)
    assert result['boundedCarrierSurfacePixels'] == 0
    assert result['unexpectedVisualChangedPixels'] == 32*48


def test_new_glyph_over_valid_surface_does_not_invalidate_text_free_paper(tmp_path):
    result = audit_fixture(tmp_path, new_text=True)
    assert result['unexpectedVisualChangedPixels'] == 0


def test_surface_claim_still_detects_foreign_preview_overlay(tmp_path):
    result = audit_fixture(tmp_path, new_text=True, foreign_overlay=True)
    assert result['unexpectedVisualChangedPixels'] == 1


def test_child_local_completion_cannot_leak_outside_rebuilt_round_plate(tmp_path, monkeypatch):
    import cv2
    from app.services.reconstruction import contour_objects
    from app.services.reconstruction.exclusive_ownership import build_exclusive_scene

    source = np.full((220, 360, 3), 250, np.uint8)
    source[20:80, 20:90] = [20, 80, 130]
    support = np.zeros((180, 320), np.uint8)
    cv2.rectangle(support, (20, 0), (299, 179), 255, -1)
    cv2.rectangle(support, (0, 20), (319, 159), 255, -1)
    for center in [(20, 20), (299, 20), (20, 159), (299, 159)]:
        cv2.circle(support, center, 20, 255, -1)

    def proposals(cleaned, *args):
        return [
            {'box': (20, 20, 90, 80), 'rgba': np.dstack((cleaned[20:80, 20:90], np.full((60, 70), 255, np.uint8))),
             'panel': False, 'source': 'photo', 'evidence': {'visualAssetPolicy': 'source_verified_photographic_rectangle'}},
            {'box': (20, 20, 340, 200), 'rgba': np.dstack((cleaned[20:200, 20:340], support)),
             'panel': True, 'source': 'plate', 'evidence': {'visualAssetPolicy': 'closed_carrier_outer_contour',
                'carrierBoundaryVerified': True, 'contourProposalId': 'round_plate'}}], {}

    monkeypatch.setattr(contour_objects, 'plan_contour_objects', proposals)
    image_io.imwrite(tmp_path/'source.png', source)
    root = tmp_path/'output'
    scene = build_exclusive_scene(tmp_path/'source.png', {'elements': []}, {}, [], root, 'test', 1)
    plate = next(e for e in scene['elements'] if e.get('source') == 'plate')
    assert plate['metadata']['carrierSurfaceRebuilt']
    asset = image_io.imread(root/'assets'/plate['src'].split('/')[-1], -1)
    mask = image_io.imread(root/plate['metadata']['carrierSupportMask'], -1)[:, :, 3] > 0
    assert not asset[:, :, 3][~mask].any()
    assert np.all(asset[:, :, 3][mask] == 255)


def test_foreground_cannot_underpaint_independent_photo_with_arrow_colour(tmp_path, monkeypatch):
    from app.services.reconstruction import contour_objects
    from app.services.reconstruction.exclusive_ownership import build_exclusive_scene

    source = np.full((150, 180, 3), 250, np.uint8)
    source[30:90, 30:90] = [40, 60, 80]
    source[55:65, 90:120] = [230, 120, 10]
    source[100, 35] = [230, 120, 10]
    mask = np.zeros((150, 180), np.uint8)
    mask[55:65, 90:120] = 255
    mask[100, 35] = 255

    def proposals(cleaned, *args):
        return [
            {'box': (30, 30, 90, 90), 'rgba': np.dstack((cleaned[30:90,30:90], np.full((60,60),255,np.uint8))),
             'panel': False, 'source': 'photo', 'evidence': {'visualAssetPolicy':'source_verified_photographic_rectangle'}},
            {'box': (0, 0, 180, 150), 'rgba': np.dstack((cleaned, mask)),
             'panel': False, 'source': 'arrow', 'evidence': {'visualAssetPolicy':'independent_carrier_detail'}}], {}

    monkeypatch.setattr(contour_objects, 'plan_contour_objects', proposals)
    image_io.imwrite(tmp_path/'source.png', source)
    root = tmp_path/'output'
    scene = build_exclusive_scene(tmp_path/'source.png', {'elements': []}, {}, [], root, 'test', 1)
    arrow = next(e for e in scene['elements'] if e.get('source')=='arrow')
    asset = image_io.imread(root/'assets'/arrow['src'].split('/')[-1], -1)
    assert asset[58-arrow['y'], 88-arrow['x'], 3] == 0
    assert asset[58-arrow['y'], 92-arrow['x'], 3] == 255


@pytest.mark.parametrize('delta,owned', [(20, True), (3, False)])
def test_surface_claim_cannot_hide_content_loss_or_take_another_owner(tmp_path, delta, owned):
    with pytest.raises(ValueError, match='bounded paper-only'):
        audit_fixture(tmp_path, delta=delta, owned=owned)


def test_synthetic_detail_cannot_cover_another_objects_glyph_paper(tmp_path, monkeypatch):
    from copy import deepcopy

    from app.services.reconstruction import contour_objects, text_cohorts
    from app.services.reconstruction.exclusive_ownership import build_exclusive_scene
    from app.services.visual_qa.analyzer import render_preview

    image = np.full((140, 260, 3), 250, np.uint8)
    blue = [220, 80, 10]
    image[20:80, 20:220] = blue
    glyph = np.zeros(image.shape[:2], bool)
    glyph[40:60, 80:160] = True
    image[glyph] = 255
    banner = np.ones((60, 200), bool)
    for x, y in [(25, 25), (210, 25), (25, 70), (210, 70)]:
        image[y:y+5, x:x+5] = 250
        banner[y-20:y-15, x-20:x-15] = False
    text = {'id': 't', 'type': 'text', 'text': 'TITLE', 'x': 80, 'y': 40,
            'width': 80, 'height': 20, 'zIndex': 1, 'rotation': 0,
            'style': {'fontFamily': 'Arial', 'fontSize': 16, 'color': '#FFFFFF'},
            'metadata': {'reconstructionStrategy': 'editable_text'}}
    fill = image.copy()
    fill[glyph] = blue
    monkeypatch.setattr(text_cohorts, 'prepare_text_lines', lambda *args: {
        't': ([(deepcopy(text), glyph, fill)], {})})

    def proposals(cleaned, *args):
        return [
            {'box': (20, 20, 220, 80), 'rgba': np.dstack((cleaned[20:80, 20:220], banner*255)).astype(np.uint8),
             'panel': False, 'source': 'banner', 'evidence': {'visualAssetPolicy': 'test_banner'}},
            {'box': (25, 25, 215, 75), 'rgba': np.dstack((cleaned[25:75, 25:215], np.full((50, 190), 255))).astype(np.uint8),
             'panel': False, 'source': 'detail', 'evidence': {'visualAssetPolicy': 'independent_carrier_detail'}},
        ], {}

    monkeypatch.setattr(contour_objects, 'plan_contour_objects', proposals)
    source = tmp_path/'source.png'
    image_io.imwrite(source, image)
    root = tmp_path/'output'
    scene = build_exclusive_scene(source, {'elements': [text]}, {}, [], root, 'test', 1)
    local = deepcopy(scene)
    local['elements'] = [e for e in local['elements'] if e['type'] != 'text']
    for e in local['elements']:
        e['src'] = str(root/('backgrounds' if e['type']=='background' else 'assets')/e['src'].split('/')[-1])
    preview = root/'preview.png'
    render_preview(root/'backgrounds/page_1.png', local, preview)
    assert np.array_equal(image_io.imread(preview)[50, 120], blue)
    detail = next(e for e in scene['elements'] if e.get('source') == 'detail')
    path = root/'assets'/detail['src'].split('/')[-1]
    asset = image_io.imread(path, -1)
    assert asset[50-detail['y'], 120-detail['x'], 3] == 0
    # A synthetic white patch over the native glyph used to pass preservation
    # checks because all glyph regions were exempt. It must now be detected.
    asset[50-detail['y'], 120-detail['x']] = [250, 250, 250, 255]
    image_io.imwrite(path, asset)
    render_preview(root/'backgrounds/page_1.png', local, preview)
    audit = inspect_preservation(source, scene, root, preview)
    assert audit['glyphPaperOverpaintPixels'] > 0
    assert audit['unexpectedVisualChangedPixels'] > 0
