"""Real saved scene, isolated browser state; no saves, analysis or paid calls."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

STORE = "import(performance.getEntriesByType('resource').find(e=>e.name.includes('/src/stores/useProjectStore.ts')).name)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project')
    parser.add_argument('--output', type=Path, default=Path('temp/module-interaction'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    errors, missing, writes = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1800, 'height': 1100})
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('response', lambda r: missing.append(r.url) if r.status >= 400 and '/media/' in r.url else None)
        page.on('request', lambda r: writes.append(r.url) if r.method not in ('GET', 'HEAD', 'OPTIONS') else None)
        page.goto('http://127.0.0.1:5173/?project=' + args.project)
        page.wait_for_function("""async () => {
          const m=performance.getEntriesByType('resource').find(e=>e.name.includes('/src/stores/useProjectStore.ts'));
          if (!m) return false; const {useProjectStore}=await import(m.name);
          return Boolean(useProjectStore.getState().slides[0]?.elements?.length) && !useProjectStore.getState().busy;
        }""")
        page.wait_for_function("() => [...document.querySelectorAll('.canvas-overlay img')].every(i=>i.complete && i.naturalWidth>0)")
        page.evaluate('() => document.fonts.ready')
        expect(page.locator('[data-element-id]').first).to_be_visible()
        original = page.request.get('http://127.0.0.1:5173/api/projects/'+args.project+'/slides/1')
        assert original.ok
        original = original.json()
        page.evaluate(f"async layout => (await {STORE}).useProjectStore.setState({{slides:[layout],activePage:0,selectedIds:[]}})", original)
        contracts = page.evaluate("""async () => {
          const {moduleIds,transformSelection}=await import('/src/editor/selectionGeometry.ts');
          const item=(id,groupId,x)=>({id,groupId,type:'text',x,y:10,width:20,height:20,rotation:0,zIndex:1,text:id,style:{fontSize:10}});
          const p={version:'1',slide:{width:100,height:100},elements:[item('a','root',10),item('b','child',40),item('c','other',70)],
            metadata:{reconstructionPlan:{modules:[{id:'root',children:[{id:'child'}]}]}}};
          const ids=moduleIds(p,'a'), moved=transformSelection(p,ids,{x:1000});
          const resized=transformSelection(p,ids,{width:25,height:10});
          return {nested:ids.join(',')==='a,b',clamped:moved.elements[1].x===80,
            untouched:moved.elements[2]===p.elements[2],scaled:resized.elements[0].style.fontSize===5,
            invalid:transformSelection(p,ids,{x:NaN})===p};
        }""")
        assert all(contracts.values()), contracts
        photo = next(e for e in original['elements'] if e.get('metadata', {}).get('visualAssetPolicy') == 'source_verified_photographic_rectangle')
        members = [e for e in original['elements'] if e.get('groupId') == photo['groupId'] and e['type'] not in ('group', 'background')]
        assert len(members) > 2
        node = page.locator('[data-element-id="' + photo['id'] + '"]')
        node.click()
        expect(page.get_by_label('宽度', exact=True)).to_have_value(str(round(photo['width'])))
        page.screenshot(path=str(args.output/'initial.png'), full_page=True)
        page.get_by_role('button', name='选择所属模块', exact=True).click()
        expect(page.get_by_role('heading', name=f'已选 {len(members)} 个对象 · 联动编辑')).to_be_visible()
        bounds = node.bounding_box()
        page.mouse.move(bounds['x'] + bounds['width']/2, bounds['y'] + bounds['height']/2)
        page.mouse.down()
        page.mouse.move(bounds['x'] + bounds['width']/2 - 18, bounds['y'] + bounds['height']/2 + 12, steps=6)
        page.mouse.up()
        moved = json.loads(page.evaluate(f"async () => JSON.stringify((await {STORE}).useProjectStore.getState().slides[0])"))
        old = {e['id']: e for e in original['elements']}
        ids = {e['id'] for e in members}
        changes = []
        for e in moved['elements']:
            before = old[e['id']]
            delta = (round(e['x']-before['x'], 5), round(e['y']-before['y'], 5))
            if e['id'] in ids:
                changes.append(delta)
            else:
                assert delta == (0, 0), 'Unselected object moved'
        assert len(set(changes)) == 1 and changes[0] != (0, 0), 'Module members did not move together'
        # This screenshot exposes title/text baselines and holes after translation.
        page.screenshot(path=str(args.output/'module-moved.png'), full_page=True)
        # Atomic group scaling, with AST geometry and fonts following it.
        width = float(page.get_by_label('宽度', exact=True).input_value())
        page.get_by_label('宽度', exact=True).fill(str(round(width * .9)))
        page.get_by_label('宽度', exact=True).press('Tab')
        scaled = json.loads(page.evaluate(f"async () => JSON.stringify((await {STORE}).useProjectStore.getState().slides[0])"))
        for e in scaled['elements']:
            if e['id'] not in ids:
                assert e == old[e['id']]
            elif e.get('bbox'):
                assert abs(e['bbox']['left']-e['x']) < .001
                assert abs(e['bbox']['width']-e['width']) < .001
        (args.output/'transformed-layout.json').write_text(json.dumps(scaled, ensure_ascii=False), encoding='utf-8')
        # Reset only the test tab's in-memory state, then confirm individual movement.
        page.evaluate(f"async layout => (await {STORE}).useProjectStore.setState({{slides:[layout],activePage:0,selectedIds:[]}})", original)
        node.click()
        page.get_by_label('X', exact=True).fill(str(photo['x'] - 12))
        single = json.loads(page.evaluate(f"async () => JSON.stringify((await {STORE}).useProjectStore.getState().slides[0])"))
        assert [e['id'] for e in single['elements'] if e != old[e['id']]] == [photo['id']]
        page.screenshot(path=str(args.output/'individual-moved.png'), full_page=True)
        # Inspect real rendered alpha at overlapping boxes; click a transparent
        # top image and assert routing reaches the lower actual visible owner.
        page.evaluate(f"async layout => (await {STORE}).useProjectStore.setState({{slides:[layout],selectedIds:[]}})", original)
        hit = page.evaluate("""async () => {
          const {hitElement}=await import('/src/editor/alphaHitTest.ts');
          const overlay=document.querySelector('.canvas-overlay'), b=overlay.getBoundingClientRect();
          for(let y=Math.max(0,b.top)+4;y<Math.min(innerHeight,b.bottom);y+=5)
          for(let x=Math.max(0,b.left)+4;x<Math.min(innerWidth,b.right);x+=5) {
            const box=document.elementFromPoint(x,y)?.closest('[data-element-id]');
            const actual=hitElement(overlay,x,y);
            if(box && actual && box!==actual && box.classList.contains('visual-image'))
              return {x,y,top:box.dataset.elementId,expected:actual.dataset.elementId};
          } return null;
        }""")
        assert hit, 'No transparent-overlap case found in real fixture'
        page.mouse.click(hit['x'], hit['y'])
        selected = page.evaluate(f"async () => (await {STORE}).useProjectStore.getState().selectedIds")
        assert selected == [hit['expected']], (hit, selected)
        # Read the existing UI at a narrower viewport as well.
        page.set_viewport_size({'width': 1280, 'height': 900})
        page.screenshot(path=str(args.output/'narrow.png'), full_page=True)
        assert not errors and not missing and not writes, (errors, missing, writes)
        result = {'geometryContracts': contracts, 'moduleMemberCount': len(members), 'moduleTranslation': changes[0],
                  'singlePhotoOnly': True, 'moduleScaleAndBbox': True, 'transparentHit': hit,
                  'errors': errors, 'missingAssets': missing, 'writeRequests': writes}
        (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False))
        browser.close()


if __name__ == '__main__':
    main()
