"""Compare saved carrier assets and inspect them without foreground objects.

No provider requests or project writes. Browser mutations affect only its
isolated in-memory store; generated evidence lives under the supplied output.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from app.services.pptx.renderer import PPTXRenderer
from app.services.reconstruction.preservation_checks import inspect_preservation


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
    module_check = None
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
        if rebuilt:
            carrier = max(rebuilt, key=lambda e: e['width']*e['height'])
            page.evaluate("""async id => {
              const m=performance.getEntriesByType('resource').find(e=>e.name.includes('/src/stores/useProjectStore.ts'));
              (await import(m.name)).useProjectStore.setState({selectedIds:[id]});
            }""", carrier['id'])
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
            page.screenshot(path=str(output/'browser-rebuilt-module-moved.png'), full_page=True)
            module_check = {'carrier': carrier['id'], 'members': len(selected),
                            'allNewDetailsMoveWithCarrier': True, 'deltaX': 8}
        for target in targets:
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
        assert not errors and not missing and not writes, (errors, missing, writes)
        browser.close()
    return {'errors': errors, 'missingAssets': missing, 'writeRequests': writes,
            'foregroundMovedInMemory': True, 'exposedCarrierCount': len(targets),
            'rebuiltModuleMovement': module_check}


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
    previous = {e['metadata']['contourProposalId']: e for e in old['elements']
                if e.get('metadata', {}).get('carrierSupportMask')}
    rows, targets = [], []
    for e in layout['elements']:
        meta = e.get('metadata', {})
        if not meta.get('carrierSupportMask'):
            continue
        before = previous[meta['contourProposalId']]
        a = rgba(args.before/'assets'/before['src'].split('/')[-1])
        b = rgba(args.after/'assets'/e['src'].split('/')[-1])
        support = rgba(args.after/meta['carrierSupportMask'])[:, :, 3] > 0
        old_support = rgba(args.before/before['metadata']['carrierSupportMask'])[:, :, 3] > 0
        assert np.array_equal(old_support, support), 'Carrier outline changed'
        assert a.shape == b.shape
        rebuilt = bool(meta.get('carrierSurfaceRebuilt'))
        old_owner = rgba(args.before/before['metadata']['sourceMaskPath'])[:, :, 3]>0
        new_owner = rgba(args.after/meta['sourceMaskPath'])[:, :, 3]>0
        retained_source = old_owner & new_owner
        if not rebuilt:
            # Reclassification can change which children supply synthetic
            # paper. Compare actual source pixels owned by both versions.
            assert np.array_equal(a[retained_source], b[retained_source]), 'Retained source pixels repainted'
        if rebuilt:
            assert not b[:, :, 3][~support].any(), 'Rebuilt carrier leaks outside its measured silhouette'
        else:
            assert np.array_equal(a[~support], b[~support]), 'Exterior transparency changed'
        holes_before = int((support & (a[:, :, 3] == 0)).sum())
        holes_after = int((support & (b[:, :, 3] == 0)).sum())
        rows.append({'id': e['id'], 'holesBefore': holes_before, 'holesAfter': holes_after,
                     'visiblePixelsUnchanged': bool(np.array_equal(a[a[:, :, 3]>0], b[a[:, :, 3]>0])),
                     'surfaceRebuilt': rebuilt, 'independentDetailIds': meta.get('carrierDetailIds', []),
                     'exteriorUnchanged': bool(np.array_equal(a[~support], b[~support])),
                     'exteriorTransparent': not bool(b[:, :, 3][~support].any()),
                     'repair': meta.get('localCarrierRepair')})
        if holes_before == 0 and not rebuilt:
            continue
        targets.append(e)
        w, h = e['width'], e['height']
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
        isolated = {**layout, 'slide': {'width': w, 'height': h}, 'elements': [single]}
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
    report['preservation'] = inspect_preservation(root/'original.png', layout, root, root/'reconstructed_preview.png')
    assert report['preservation']['unexpectedVisualChangedPixels'] == 0
    assert report['preservation']['glyphPaperOverpaintPixels'] == 0
    if args.browser:
        report['browser'] = inspect_browser(args.after.name, layout, targets, args.output)
    (args.output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    assert report['totalHolesAfter'] == 0, 'Some carrier interiors remain transparent'


if __name__ == '__main__':
    main()
