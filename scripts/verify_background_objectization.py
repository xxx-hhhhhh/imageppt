"""Verify a stored offline reconstruction in the real editor, without AI calls.

Usage: python scripts/verify_background_objectization.py PROJECT_ID
Uses the project's real scene and media; replaces only upload/analysis dispatch.
Dragging and exporting affect this supplied regression project, so use a test copy.
"""

import argparse
import base64
import json
import re
from pathlib import Path
from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "outputs" / args.project_id
    layout = json.loads((root / "slides/page_1.json").read_text(encoding="utf-8"))
    project = {"id": args.project_id, "name": "Background objectization regression", "imageCount": 1}
    source = root / "source.png"
    source_url = "data:image/png;base64," + base64.b64encode(source.read_bytes()).decode()
    assets = [item for item in layout["elements"] if item.get("type") == "image"
              and item.get("metadata", {}).get("reconstructionStrategySource") == "background_objectization"
              and not any(item.get("metadata", {}).get(k) for k in ("suppressed", "suppressRender", "ownedBy"))]
    assert assets, "Project has no extracted background objects"
    target = max(assets, key=lambda item: item["width"] * item["height"])

    def api(route):
        request = route.request
        url = request.url
        if url.endswith("/api/projects") and request.method == "POST":
            body = project
        elif "/images" in url and request.method == "POST":
            body = {"project": project, "images": [{"id": "image-1", "source_url": source_url}]}
        elif "/analyze" in url and request.method == "POST":
            body = {"project": project, "slides": [layout], "provider": "local", "warnings": []}
        elif "/vision/" in url:
            body = {"provider": "local", "configured": False}
        elif "/inpainting/" in url:
            body = {"enabled": False, "connected": False, "model": "lama"}
        else:
            route.continue_()
            return
        route.fulfill(status=200, json=body)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.route("**/api/**", api)
        page.goto("http://127.0.0.1:5173/", wait_until="domcontentloaded")
        page.locator("input[type=file]").first.set_input_files(str(source))
        page.get_by_role("button", name=re.compile("AI解析")).click()
        node = page.locator(f'[data-element-id="{target["id"]}"]')
        expect(node).to_be_visible()
        page.wait_for_function("() => [...document.querySelectorAll('.visual-image img')].every(i => i.complete && i.naturalWidth > 0)")
        before = node.bounding_box()
        assert before
        # The lower edge of the recovered ribbon avoids overlaid text labels.
        x = before["x"] + before["width"] * .4
        y = before["y"] + before["height"] * .9
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + 18, y - 20, steps=8)
        page.mouse.up()
        expect(node).not_to_have_css("top", f'{target["y"]}px')
        after = node.bounding_box()
        assert after and abs(after["y"] - before["y"]) > 10
        score = json.loads((root / "visual_score.json").read_text(encoding="utf-8"))
        if score.get("revisionStatus") == "stagnated":
            page.get_by_role("button", name="使用当前结果", exact=True).click()
        page.get_by_role("button", name="通过本页", exact=True).click()
        with page.expect_download() as download:
            page.get_by_role("button", name=re.compile("导出PPT")).click()
        download.value.save_as(str(root / "browser_objectization_export.pptx"))
        print(json.dumps({"projectId": args.project_id, "draggedAsset": target["id"],
                          "deltaY": round(after["y"] - before["y"], 2), "brokenImages": 0,
                          "export": "browser_objectization_export.pptx"}))
        browser.close()


if __name__ == "__main__":
    main()
