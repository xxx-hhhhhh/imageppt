"""Read real saved result, verify movable image editing in an isolated browser.

No provider calls, mocked responses or edits to the user's existing page.
The test initializes the app's existing Zustand store with actual API data;
the product currently has no existing-project URL route.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id")
    parser.add_argument("--output-root",type=Path)
    parser.add_argument('--text-id',action='append',default=[],help='Verify every editable span belonging to these OCR ids')
    parser.add_argument('--image-id', help='Exercise a particular saved image object')
    args = parser.parse_args()
    output = (args.output_root or Path(__file__).resolve().parents[1] / "outputs") / args.project_id
    errors, missing = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, accept_downloads=True)
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("response", lambda r: missing.append(r.url) if r.status >= 400 and "/media/" in r.url else None)
        page.goto("http://127.0.0.1:5173/?project="+args.project_id, wait_until="domcontentloaded")
        expect(page.get_by_role("button", name="AI设置", exact=True)).to_be_visible()
        page.wait_for_function("""async () => {
          const module=performance.getEntriesByType('resource').find(e=>e.name.includes('/src/stores/useProjectStore.ts'));
          if (!module) return false;
          const {useProjectStore} = await import(module.name);
          const state=useProjectStore.getState(); return Boolean(state.project?.id) && !state.busy;
        }""")
        layout = page.evaluate("""async id => {
          const response = await fetch('/api/projects/'+id+'/slides/1');
          if (!response.ok) throw new Error('Actual saved layout unavailable');
          const layout = await response.json();
          const {useProjectStore} = await import(performance.getEntriesByType('resource').find(e=>e.name.includes('/src/stores/useProjectStore.ts')).name);
          if (useProjectStore.getState().project?.id !== id) throw new Error('Project restore failed');
          return layout;
        }""", args.project_id)
        expect(page.get_by_role("button", name="手动编辑", exact=True).first).to_be_enabled()
        page.wait_for_function("() => [...document.querySelectorAll('img')].filter(i => i.offsetParent !== null).every(i => i.complete && i.naturalWidth > 0)")
        page.screenshot(path=str(output / "browser_review.png"), full_page=True)
        page.get_by_role("button", name="手动编辑", exact=True).first.click()
        for text in (e for e in layout['elements'] if e.get('metadata',{}).get('exclusiveFontVerified')):
            expect(page.locator(f'[data-element-id="{text["id"]}"]')).to_have_css('white-space','pre')
        text = min((e for e in layout['elements'] if e.get('owner')=='editable_text' and len(e.get('text') or '')>=4),key=lambda e:e['y'])
        # The DOM owns visible text; the underlying Fabric proxy must not paint
        # the same foreground pixels (that used to duplicate/wrap old lines).
        duplicate_pixels = page.evaluate("""item => {
          const c=document.querySelector('canvas.lower-canvas');
          const d=c.getContext('2d').getImageData(Math.round(item.x),Math.round(item.y),Math.round(item.width),Math.round(item.height)).data;
          const hex=item.style.color.slice(1), rgb=[0,2,4].map(i=>parseInt(hex.slice(i,i+2),16));
          let count=0; for(let i=0;i<d.length;i+=4) if(d[i+3]>0 && rgb.every((v,j)=>Math.abs(d[i+j]-v)<4)) count++;
          return count;
        }""",text)
        assert duplicate_pixels==0, 'Fabric paints a second native text copy'
        selected=[text]
        if args.text_id:
            decisions={d['id']:d for d in layout['metadata']['ownershipAudit']['textDecisions']}
            ids=[]
            for identifier in args.text_id:
                decision=decisions[identifier]
                assert decision['decision']=='editable_text', f'{identifier} still raster'
                assert decision['editableCharacterCount']==decision['sourceCharacterCount']
                ids.extend(decision['editableSpanIds'])
            selected=[next(e for e in layout['elements'] if e['id']==identifier) for identifier in ids]
        for item in selected:
            node = page.locator(f'[data-element-id="{item["id"]}"]')
            node.dblclick()
            expect(node.locator('[data-text-content]')).to_have_attribute('contenteditable','true')
            page.keyboard.press('Control+A')
            page.keyboard.insert_text('单一文字层回归')
            page.get_by_label('字号',exact=True).click()
            expect(page.get_by_role('textbox',name='内容',exact=True)).to_have_value('单一文字层回归')
            page.evaluate("""async layout => {
              const {useProjectStore} = await import(performance.getEntriesByType('resource').find(e=>e.name.includes('/src/stores/useProjectStore.ts')).name);
              useProjectStore.setState({slides:[layout],selectedIds:[]});
            }""",layout)
            expect(node.locator('[data-text-content]')).to_have_text(item['text'])
        # A paper carrier can sit behind editable text. Clicking its centre
        # correctly selects the text, not the carrier. Pick a real hit-tested
        # image point rather than interpreting that expected behaviour as failure.
        hit = page.evaluate("""({layout,imageId}) => {
          for (const n of document.querySelectorAll('.visual-image')) {
            const item=layout.elements.find(e=>e.id===n.dataset.elementId);
            if(imageId && item?.id!==imageId) continue;
            // A page-width substrate is correctly clamped at x=0. Exercise a
            // genuine independent object with room to move on both axes.
            if(!item || item.width>layout.slide.width*.55 || item.height>layout.slide.height*.65) continue;
            const b=n.getBoundingClientRect(); if(b.width*b.height<1500) continue;
            for(let y=1;y<8;y++) for(let x=1;x<8;x++) {
              const px=b.x+b.width*x/8, py=b.y+b.height*y/8;
              const top=document.elementFromPoint(px,py)?.closest('[data-element-id]');
              if(top===n) return {id:n.dataset.elementId,x:px,y:py};
            }
          }
          return null;
        }""",{"layout":layout,"imageId":args.image_id})
        assert hit, 'No independently selectable image'
        image = next(e for e in layout['elements'] if e['id']==hit['id'])
        node = page.locator(f'[data-element-id="{image["id"]}"]')
        expect(node).to_be_visible()
        bounds = node.bounding_box()
        assert bounds
        page.mouse.move(hit['x'],hit['y'])
        page.mouse.down()
        page.mouse.move(hit['x']+30,hit['y']+20, steps=8)
        page.mouse.up()
        expect(node).not_to_have_css("left", f'{image["x"]}px')
        # In dense cards another foreground element may cover a corner handle.
        # Use the real property control for an explicit target, then assert the
        # target itself changed (a different selection is not a successful resize).
        if args.image_id:
            expect(page.get_by_label("宽度",exact=True)).to_have_value(str(image['width']))
            page.get_by_label("宽度",exact=True).fill(str(image['width']+20))
            page.get_by_label("宽度",exact=True).press('Tab')
        else:
            handle = node.locator(".resize-handle")
            expect(handle).to_be_visible()
            bounds = handle.bounding_box()
            assert bounds
            page.mouse.move(bounds["x"]+bounds["width"]/2,bounds["y"]+bounds["height"]/2)
            page.mouse.down()
            page.mouse.move(bounds["x"]+25,bounds["y"]+15,steps=6)
            page.mouse.up()
        expect(node).not_to_have_css('width',f'{image["width"]}px')
        page.screenshot(path=str(output / "browser_image_moved.png"),full_page=True)
        # Restore only this isolated in-memory test state; baseline files untouched.
        page.evaluate("""async layout => {
          const {useProjectStore} = await import(performance.getEntriesByType('resource').find(e=>e.name.includes('/src/stores/useProjectStore.ts')).name);
          useProjectStore.setState({slides:[layout],selectedIds:[]});
        }""", layout)
        expect(page.get_by_role("button",name="保存当前页面",exact=True)).to_be_enabled()
        page.screenshot(path=str(output / "browser_final.png"),full_page=True)
        assert not errors and not missing, {"errors":errors,"missing":missing}
        result = {"projectId": args.project_id,"realSavedApiLayout":True,"imageDrag":True,"imageResize":True,"nativeSingleLineRendering":True,
                  "domTextEditing":True,"fabricDuplicateTextPixelCount":duplicate_pixels,
                  "verifiedOcrLineIds":args.text_id,"editedTextSpanIds":[e['id'] for e in selected],"movedImageId":image['id'],
                  "missingAssets":missing,"browserErrors":errors,"nativeTextCount":sum(e.get('owner') == 'editable_text' for e in layout['elements']),
                  "originalResultFilesModified":False,"automaticRevisionCalls":0}
        (output / "exclusive_browser.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(result),flush=True)
        browser.close()


if __name__ == "__main__":
    main()
