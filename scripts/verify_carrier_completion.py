"""Compare saved carrier assets and inspect them without foreground objects.

No provider requests or project writes. Browser mutations affect only its
isolated in-memory store; generated evidence lives under the supplied output.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from app.services.pptx.renderer import PPTXRenderer
from app.services.reconstruction.contour_objects import _visible_surface
from app.services.reconstruction.preservation_checks import inspect_preservation
from app.services.visual_qa.analyzer import render_preview


def rgba(path):
    return np.array(Image.open(path).convert('RGBA'))


def checkerboard(asset):
    yy, xx = np.indices(asset.shape[:2])
    paper = np.where(((xx//12 + yy//12) % 2)[:, :, None], 210, 245)
    alpha = asset[:, :, 3:4]/255
    return Image.fromarray(np.rint(asset[:, :, :3]*alpha + paper*(1-alpha)).astype(np.uint8))


def inspect_browser(project, layout, targets, output):
    from playwright.sync_api import expect, sync_playwright

    errors, missing, writes = [], [], []
    module_checks = []
    single_moves = []
    photos = [e for e in layout['elements'] if e.get('metadata', {}).get('visualAssetPolicy')
              == 'source_verified_photographic_rectangle']
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1800, 'height': 1100})
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('response', lambda r: missing.append(r.url) if r.status >= 400 and '/media/' in r.url else None)
        page.on('request', lambda r: writes.append(r.url) if r.method not in ('GET', 'HEAD', 'OPTIONS') else None)
        page.goto('http://127.0.0.1:5173/?project='+project)
        page.wait_for_function("""async () => {
          const m=performance.getEntriesByType('resource').find(e=>e.name.includes('/src/stores/useProjectStore.ts'));
          if (!m) return false; const {useProjectStore}=await import(m.name);
          return Boolean(useProjectStore.getState().slides[0]?.elements?.length) && !useProjectStore.getState().busy;
        }""")
        visible = [e for e in layout['elements'] if e['type'] != 'group' and not any(
            e.get('metadata', {}).get(k) for k in ('suppressed', 'suppressRender', 'ownedBy'))]
        expect(page.locator('.canvas-overlay [data-element-id]')).to_have_count(len(visible))
        page.evaluate("""async () => {
          await document.fonts.ready;
          await Promise.all([...document.querySelectorAll('.canvas-overlay img')].map(i=>i.decode()));
        }""")
        page.screenshot(path=str(output/'browser-full.png'), full_page=True)
        rebuilt = [e for e in targets if e.get('metadata', {}).get('carrierSurfaceRebuilt')
                   and e.get('groupId') and e['metadata'].get('carrierDetailIds')]
        module_targets = [max(rebuilt, key=lambda e: e['width']*e['height'])] if rebuilt else []
        photo_parents = [e for e in rebuilt if photos and all(e['x']<=p['x'] and e['y']<=p['y']
                         and e['x']+e['width']>=p['x']+p['width']
                         and e['y']+e['height']>=p['y']+p['height'] for p in photos)]
        if photo_parents:
            photo_parent = min(photo_parents, key=lambda e:e['width']*e['height'])
            if photo_parent not in module_targets:
                module_targets.append(photo_parent)
        for carrier in module_targets:
            page.evaluate("""async ({layout, id}) => {
              const m=performance.getEntriesByType('resource').find(e=>e.name.includes('/src/stores/useProjectStore.ts'));
              (await import(m.name)).useProjectStore.setState({slides:[layout],activePage:0,selectedIds:[id]});
            }""", {'layout': layout, 'id': carrier['id']})
            page.get_by_role('button', name='选择所属模块', exact=True).click()
            selected = page.evaluate("""async () => {
              const m=performance.getEntriesByType('resource').find(e=>e.name.includes('/src/stores/useProjectStore.ts'));
              return (await import(m.name)).useProjectStore.getState().selectedIds;
            }""")
            assert {carrier['id'], *carrier['metadata']['carrierDetailIds']} <= set(selected)
            original_x = float(page.get_by_label('X', exact=True).input_value())
            page.get_by_label('X', exact=True).fill(str(original_x+8))
            page.get_by_label('X', exact=True).press('Tab')
            moved = page.evaluate("""async () => {
              const m=performance.getEntriesByType('resource').find(e=>e.name.includes('/src/stores/useProjectStore.ts'));
              return (await import(m.name)).useProjectStore.getState().slides[0];
            }""")
            old = {e['id']: e for e in layout['elements']}
            for e in moved['elements']:
                assert abs(e['x']-old[e['id']]['x']-(8 if e['id'] in selected else 0)) < .01
            page.screenshot(path=str(output/(carrier['id']+'-module-moved.png')), full_page=True)
            module_checks.append({'carrier': carrier['id'], 'members': len(selected),
                                  'allNewDetailsMoveWithCarrier': True, 'deltaX': 8})
        for target in [*targets, *photos]:
            page.evaluate("""async ({layout, id}) => {
              const m=performance.getEntriesByType('resource').find(e=>e.name.includes('/src/stores/useProjectStore.ts'));
              (await import(m.name)).useProjectStore.setState({slides:[layout],activePage:0,selectedIds:[id]});
            }""", {'layout': layout, 'id': target['id']})
            original_x = float(page.get_by_label('X', exact=True).input_value())
            page.get_by_label('X', exact=True).fill(str(original_x+8))
            page.get_by_label('X', exact=True).press('Tab')
            actual = page.evaluate("""async () => {
              const m=performance.getEntriesByType('resource').find(e=>e.name.includes('/src/stores/useProjectStore.ts'));
              return (await import(m.name)).useProjectStore.getState().slides[0];
            }""")
            old_elements = {e['id']: e for e in layout['elements']}
            for e in actual['elements']:
                assert abs(e['x']-old_elements[e['id']]['x']-(8 if e['id']==target['id'] else 0)) < .01
            single_moves.append(target['id'])
            moved = copy.deepcopy(layout)
            for e in moved['elements']:
                if e['id'] != target['id'] and e['type'] not in ('background', 'group'):
                    e['x'] += layout['slide']['width'] * 2
                    if e.get('bbox'):
                        e['bbox']['left'] = e['x']
            page.evaluate("""async layout => {
              const m=performance.getEntriesByType('resource').find(e=>e.name.includes('/src/stores/useProjectStore.ts'));
              (await import(m.name)).useProjectStore.setState({slides:[layout],activePage:0,selectedIds:[]});
            }""", moved)
            node = page.locator('[data-element-id="'+target['id']+'"]')
            expect(node).to_be_visible()
            page.wait_for_function("""id => {
              const n=document.querySelector('[data-element-id="'+id+'"] img');
              return n && n.complete && n.naturalWidth>0;
            }""", arg=target['id'])
            node.locator('img').evaluate('(img) => img.decode()')
            node.screenshot(path=str(output/(target['id']+'-browser-exposed.png')))
        moved = copy.deepcopy(layout)
        photo_ids = {e['id'] for e in photos}
        for e in moved['elements']:
            if e['id'] in photo_ids:
                e['x'] += layout['slide']['width']*2
        page.evaluate("""async layout => {
          const m=performance.getEntriesByType('resource').find(e=>e.name.includes('/src/stores/useProjectStore.ts'));
          (await import(m.name)).useProjectStore.setState({slides:[layout],activePage:0,selectedIds:[]});
        }""", moved)
        page.screenshot(path=str(output/'browser-photos-removed.png'), full_page=True)
        assert not errors and not missing and not writes, (errors, missing, writes)
        browser.close()
    return {'errors': errors, 'missingAssets': missing, 'writeRequests': writes,
            'foregroundMovedInMemory': True, 'exposedCarrierCount': len(targets),
            'individuallyMovedCarrierIds': [i for i in single_moves if i not in photo_ids],
            'individuallyMovedPhotoIds': [i for i in single_moves if i in photo_ids],
            'rebuiltModuleMovements': module_checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--browser', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    old = json.loads((args.before/'scene_final_1.json').read_text('utf-8'))
    layout = json.loads((args.after/'scene_final_1.json').read_text('utf-8'))
    previous = [e for e in old['elements'] if e.get('metadata', {}).get('carrierSupportMask')]
    def overlap_ratio(a, b):
        x, y = max(a['x'], b['x']), max(a['y'], b['y'])
        bx, by = min(a['x']+a['width'], b['x']+b['width']), min(a['y']+a['height'], b['y']+b['height'])
        intersection = max(0, bx-x)*max(0, by-y)
        return intersection/max(a['width']*a['height'], b['width']*b['height'])
    rows, targets = [], []
    for e in layout['elements']:
        meta = e.get('metadata', {})
        if not meta.get('carrierSupportMask'):
            continue
        before = max(previous, key=lambda p: overlap_ratio(e, p))
        assert overlap_ratio(e, before) > .65, 'New carrier needs an explicit visual baseline'
        a = rgba(args.before/'assets'/before['src'].split('/')[-1])
        b = rgba(args.after/'assets'/e['src'].split('/')[-1])
        support = rgba(args.after/meta['carrierSupportMask'])[:, :, 3] > 0
        old_support = rgba(args.before/before['metadata']['carrierSupportMask'])[:, :, 3] > 0
        same_geometry = all(e[k] == before[k] for k in ('x', 'y', 'width', 'height'))
        same_outline = same_geometry and np.array_equal(old_support, support)
        rebuilt = bool(meta.get('carrierSurfaceRebuilt'))
        old_owner = rgba(args.before/before['metadata']['sourceMaskPath'])[:, :, 3]>0
        new_owner = rgba(args.after/meta['sourceMaskPath'])[:, :, 3]>0
        if not rebuilt and same_geometry:
            retained_source = old_owner & new_owner
            # Reclassification can change which children supply synthetic
            # paper. Compare actual source pixels owned by both versions.
            assert np.array_equal(a[retained_source], b[retained_source]), 'Retained source pixels repainted'
        if rebuilt:
            assert not b[:, :, 3][~support].any(), 'Rebuilt carrier leaks outside its measured silhouette'
        elif same_outline:
            assert np.array_equal(a[~support], b[~support]), 'Exterior transparency changed'
        holes_before = int((old_support & (a[:, :, 3] == 0)).sum())
        holes_after = int((support & (b[:, :, 3] == 0)).sum())
        inner = cv2.erode(support.astype(np.uint8), np.ones((9, 9), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        field = _visible_surface(b[:, :, :3], inner & (b[:, :, 3] == 255))
        residual = None if field is None else int(np.max(np.abs(b[:, :, :3].astype(float)-field)[inner]))
        rows.append({'id': e['id'], 'holesBefore': holes_before, 'holesAfter': holes_after,
                     'visiblePixelsUnchanged': bool(same_geometry and np.array_equal(a[a[:, :, 3]>0], b[a[:, :, 3]>0])),
                     'outlineChanged': not same_outline,
                     'boundaryVerified': bool(meta.get('carrierBoundaryVerified')),
                     'smoothInteriorMaxError': residual,
                     'surfaceRebuilt': rebuilt, 'independentDetailIds': meta.get('carrierDetailIds', []),
                     'exteriorUnchanged': bool(same_outline and np.array_equal(a[~support], b[~support])),
                     'exteriorTransparent': not bool(b[:, :, 3][~support].any()),
                     'repair': meta.get('localCarrierRepair')})
        # Every carrier is inspected, including unchanged, zero-hole and
        # uncertain assets. Those were precisely the previous blind spot.
        targets.append(e)
        w, h = max(a.shape[1], b.shape[1]), max(a.shape[0], b.shape[0])
        board = Image.new('RGB', (w, h*2+56), 'white')
        draw = ImageDraw.Draw(board)
        draw.text((6, 5), f"BEFORE: {holes_before} transparent pixels inside silhouette", fill='black')
        board.paste(checkerboard(a), (0, 24))
        caption = 'rim-measured surface; foreground retained separately' if rebuilt else 'existing pixels unchanged'
        draw.text((6, h+30), f"AFTER: {holes_after} / {caption}", fill='black')
        board.paste(checkerboard(b), (0, h+56))
        board.save(args.output/(e['id']+'-comparison.png'))
        single = copy.deepcopy(e)
        single.update(x=0, y=0, src=str((args.after/'assets'/e['src'].split('/')[-1]).resolve()))
        isolated = {**layout, 'slide': {'width': e['width'], 'height': e['height']}, 'elements': [single]}
        export = args.output/e['id']
        PPTXRenderer().render_project('isolated-carrier-proof', [isolated], output_dir=export)
        (export/'slides').mkdir(exist_ok=True)
        (export/'slides/page_1.json').write_text(json.dumps(isolated, ensure_ascii=False), encoding='utf-8')
    rejected = [e['id'] for e in layout['elements']
                if e.get('metadata', {}).get('carrierRejectedReason')]
    report = {'carriers': rows, 'rejectedFalseCarriers': rejected,
              'totalHolesBefore': sum(r['holesBefore'] for r in rows),
              'totalHolesAfter': sum(r['holesAfter'] for r in rows)}
    root = args.after.resolve()
    photo_free = copy.deepcopy(layout)
    photo_free['elements'] = [e for e in photo_free['elements'] if e.get('metadata', {}).get('visualAssetPolicy')
                             != 'source_verified_photographic_rectangle']
    for e in photo_free['elements']:
        if e.get('src'):
            e['src'] = str(root/('backgrounds' if e['type']=='background' else 'assets')/e['src'].split('/')[-1])
    render_preview(root/'backgrounds/page_1.png', photo_free, args.output/'photos-removed-preview.png')
    revealed = np.array(Image.open(args.output/'photos-removed-preview.png').convert('RGB'))
    report['photoRevealChecks'] = []
    for photo in layout['elements']:
        if photo.get('metadata', {}).get('visualAssetPolicy') != 'source_verified_photographic_rectangle':
            continue
        x, y, w, h = (photo[k] for k in ('x', 'y', 'width', 'height'))
        parents = [e for e in targets if e['x']<=x and e['y']<=y
                   and e['x']+e['width']>=x+w and e['y']+e['height']>=y+h]
        assert parents, 'Photograph has no independently inspected background plate'
        parent = min(parents, key=lambda e:e['width']*e['height'])
        base = rgba(root/'assets'/parent['src'].split('/')[-1])
        expected = base[y-parent['y']:y-parent['y']+h, x-parent['x']:x-parent['x']+w, :3]
        owned = rgba(root/photo['metadata']['sourceMaskPath'])[:, :, 3]>0
        error = np.max(np.abs(revealed[y:y+h,x:x+w].astype(float)-expected),axis=2)
        report['photoRevealChecks'].append({'id':photo['id'], 'residualPixels':int((owned & (error>6)).sum()),
                                           'maxError':int(error[owned].max())})
    photo_dir = args.output/'photos-removed'
    PPTXRenderer().render_project('photo-movement-proof', [photo_free], output_dir=photo_dir)
    (photo_dir/'slides').mkdir(exist_ok=True)
    (photo_dir/'slides/page_1.json').write_text(json.dumps(photo_free, ensure_ascii=False), encoding='utf-8')
    report['preservation'] = inspect_preservation(root/'original.png', layout, root, root/'reconstructed_preview.png')
    assert report['preservation']['unexpectedVisualChangedPixels'] == 0
    assert report['preservation']['glyphPaperOverpaintPixels'] == 0
    if args.browser:
        report['browser'] = inspect_browser(args.after.name, layout, targets, args.output)
    (args.output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    assert report['totalHolesAfter'] == 0, 'Some carrier interiors remain transparent'
    assert all(r['boundaryVerified'] and r['smoothInteriorMaxError'] is not None
               and r['smoothInteriorMaxError'] <= 6 for r in rows), 'Some carriers still contain unverified boundaries or foreground residuals'
    assert all(r['residualPixels']==0 for r in report['photoRevealChecks']), 'Moving photos leaves foreign frames or invented paper behind'


if __name__ == '__main__':
    main()
