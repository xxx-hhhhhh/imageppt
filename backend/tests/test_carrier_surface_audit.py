import numpy as np
import pytest
from app.services.reconstruction.preservation_checks import inspect_preservation
from app.utils import image_io


def audit_fixture(tmp_path, delta=3, claimed=True, owned=True):
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
    return inspect_preservation(tmp_path/'source.png', scene, tmp_path, tmp_path/'preview.png')


def test_only_declared_small_surface_changes_are_counted_separately(tmp_path):
    result = audit_fixture(tmp_path)
    assert result['boundedCarrierSurfacePixels'] == 32*48
    assert result['unexpectedVisualChangedPixels'] == 0


def test_undeclared_changes_are_still_unexpected(tmp_path):
    result = audit_fixture(tmp_path, claimed=False)
    assert result['boundedCarrierSurfacePixels'] == 0
    assert result['unexpectedVisualChangedPixels'] == 32*48


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
