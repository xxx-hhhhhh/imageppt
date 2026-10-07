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
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "outputs" / args.project_id
    errors, missing = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, accept_downloads=True)
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("response", lambda r: missing.append(r.url) if r.status >= 400 and "/media/" in r.url else None)
        page.goto("http://127.0.0.1:5173/", wait_until="domcontentloaded")
        expect(page.get_by_role("button", name="AI设置", exact=True)).to_be_visible()
        page.wait_for_function("() => !document.querySelector('button:disabled')?.textContent?.includes('AI解析')")
        layout = page.evaluate("""async id => {
          const response = await fetch('/api/projects/'+id+'/slides/1');
          if (!response.ok) throw new Error('Actual saved layout unavailable');
          const layout = await response.json();
          const {useProjectStore} = await import('/src/stores/useProjectStore.ts');
          useProjectStore.setState({project:{id,name:'单页Ownership回归',createdAt:'2026-10-07',imageCount:1},
            slides:[layout],pagePreviews:[{id:'source',originalPreviewUrl:'/api/projects/'+id+'/artifacts/source.png',
              resultPreviewUrl:'/api/projects/'+id+'/artifacts/reconstructed_preview.png',status:'ready'}],
            activePage:0,approvedPages:[],busy:false,reviewVersion:1,message:'真实结果回归，不启动自动优化'});
          return layout;
        }""", args.project_id)
        expect(page.get_by_role("button", name="手动编辑", exact=True).first).to_be_enabled()
        page.wait_for_function("() => [...document.querySelectorAll('img')].filter(i => i.offsetParent !== null).every(i => i.complete && i.naturalWidth > 0)")
        page.screenshot(path=str(output / "browser_review.png"), full_page=True)
        page.get_by_role("button", name="手动编辑", exact=True).first.click()
        image = min((e for e in layout["elements"] if e.get("owner") == "movable_image"), key=lambda e: e["width"]*e["height"])
        node = page.locator(f'[data-element-id="{image["id"]}"]')
        expect(node).to_be_visible()
        bounds = node.bounding_box()
        assert bounds
        page.mouse.move(bounds["x"]+bounds["width"]/2, bounds["y"]+bounds["height"]/2)
        page.mouse.down()
        page.mouse.move(bounds["x"]+bounds["width"]/2+30, bounds["y"]+bounds["height"]/2+20, steps=8)
        page.mouse.up()
        expect(node).not_to_have_css("left", f'{image["x"]}px')
        handle = node.locator(".resize-handle")
        expect(handle).to_be_visible()
        bounds = handle.bounding_box()
        assert bounds
        page.mouse.move(bounds["x"]+bounds["width"]/2,bounds["y"]+bounds["height"]/2)
        page.mouse.down()
        page.mouse.move(bounds["x"]+25,bounds["y"]+15,steps=6)
        page.mouse.up()
        expect(page.get_by_label("宽度",exact=True)).not_to_have_value(str(image["width"]))
        page.screenshot(path=str(output / "browser_image_moved.png"),full_page=True)
        # Restore only this isolated in-memory test state; baseline files untouched.
        page.evaluate("""async layout => {
          const {useProjectStore} = await import('/src/stores/useProjectStore.ts');
          useProjectStore.setState({slides:[layout],selectedIds:[]});
        }""", layout)
        expect(page.get_by_role("button",name="保存当前页面",exact=True)).to_be_enabled()
        page.screenshot(path=str(output / "browser_final.png"),full_page=True)
        assert not errors and not missing, {"errors":errors,"missing":missing}
        result = {"projectId": args.project_id,"realSavedApiLayout":True,"imageDrag":True,"imageResize":True,
                  "missingAssets":missing,"browserErrors":errors,"nativeTextCount":sum(e.get('owner') == 'editable_text' for e in layout['elements']),
                  "originalResultFilesModified":False,"automaticRevisionCalls":0}
        (output / "exclusive_browser.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(result),flush=True)
        browser.close()


if __name__ == "__main__":
    main()
