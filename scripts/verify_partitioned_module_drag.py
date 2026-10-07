"""Check real editor dragging with an existing local reconstruction, without calling AI."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id", help="Existing project under outputs/")
    parser.add_argument("--url", default="http://127.0.0.1:5173/")
    parser.add_argument("--browser", default=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "outputs" / args.project_id
    layout = json.loads((root / "slides" / "page_1.json").read_text(encoding="utf-8"))
    target = next(item for item in layout["elements"]
                  if (item.get("metadata") or {}).get("partitionedFromResidual")
                  and (item.get("metadata") or {}).get("moduleMemberIds"))
    text_id = target["metadata"]["moduleMemberIds"][0]
    text = next(item for item in layout["elements"] if item["id"] == text_id)
    project = {"id": args.project_id, "name": "Browser module verification",
               "createdAt": "2026-10-02T00:00:00Z", "imageCount": 1}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.browser, headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})

        def mock_project_api(route) -> None:
            request = route.request
            path = request.url.split("?", 1)[0]
            if request.method == "POST" and path.endswith("/api/projects"):
                route.fulfill(json={**project, "imageCount": 0})
            elif request.method == "POST" and path.endswith("/images"):
                route.fulfill(json={"project": project, "images": [
                    {"id": "page-1", "source_url": f"/api/projects/{args.project_id}/artifacts/source.png"}]})
            elif request.method == "POST" and path.endswith("/analyze"):
                route.fulfill(json={"project": project, "slides": [layout],
                                    "provider": "local", "warnings": []})
            else:
                route.continue_()

        page.route("**/api/projects**", mock_project_api)
        page.goto(args.url, wait_until="domcontentloaded")
        expect(page.get_by_role("heading", name="图片转可编辑 PPT")).to_be_visible()
        page.locator("input[type=file]").first.set_input_files(str(root / "source.png"))
        button = page.get_by_role("button", name="AI解析 / 开始转换")
        expect(button).to_be_enabled()
        button.click()
        expect(page.get_by_role("heading", name="当前重建结果")).to_be_visible()
        plate = page.locator(f'[data-element-id="{target["id"]}"]')
        label = page.locator(f'[data-element-id="{text_id}"]')
        expect(plate).to_be_visible()
        expect(label).to_be_visible()

        def drag(element, dx: float, dy: float) -> None:
            element.evaluate("""(element, delta) => {
          const stage = element.closest('.canvas-stage');
          const scale = stage.getBoundingClientRect().width / stage.offsetWidth;
          const box = element.getBoundingClientRect();
          const x = box.left + 5;
          const y = box.top + 5;
          element.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, clientX: x, clientY: y}));
          window.dispatchEvent(new PointerEvent('pointermove', {bubbles: true, clientX: x + delta.dx * scale, clientY: y + delta.dy * scale}));
          window.dispatchEvent(new PointerEvent('pointerup', {bubbles: true, clientX: x + delta.dx * scale, clientY: y + delta.dy * scale}));
        }""", {"dx": dx, "dy": dy})

        drag(plate, 20, 10)
        expect(plate).to_have_css("left", re.compile(rf"^{float(target['x']) + 20:g}px$"))
        expect(label).to_have_css("left", re.compile(rf"^{float(text['x']) + 20:g}px$"))
        expect(plate).to_have_css("top", re.compile(rf"^{float(target['y']) + 10:g}px$"))
        expect(label).to_have_css("top", re.compile(rf"^{float(text['y']) + 10:g}px$"))
        drag(label, -10, 5)
        expect(plate).to_have_css("left", re.compile(rf"^{float(target['x']) + 10:g}px$"))
        expect(label).to_have_css("left", re.compile(rf"^{float(text['x']) + 10:g}px$"))
        expect(plate).to_have_css("top", re.compile(rf"^{float(target['y']) + 15:g}px$"))
        expect(label).to_have_css("top", re.compile(rf"^{float(text['y']) + 15:g}px$"))
        print(json.dumps({"project": args.project_id, "plate": target["id"],
                          "editableText": text_id, "draggedTogetherBothDirections": True}, ensure_ascii=False))
        browser.close()


if __name__ == "__main__":
    main()
