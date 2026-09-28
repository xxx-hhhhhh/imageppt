"""Browser regression for a movable module; run while Vite serves port 5173."""

from __future__ import annotations

import base64
import re
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright


def _data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()


def main() -> None:
    fixture = Path(__file__).resolve().parents[1] / "backend" / "tests" / "fixtures" / "complex_modules.png"
    source = cv2.imread(str(fixture), cv2.IMREAD_COLOR)
    assert source is not None
    badge = np.zeros((64, 64, 4), np.uint8)
    cv2.circle(badge, (32, 32), 31, (52, 91, 204, 255), -1)
    cv2.line(badge, (17, 32), (47, 32), (255, 255, 255, 255), 6)
    project = {"id": "module-qa", "name": "QA", "createdAt": "2026-09-28T00:00:00Z", "imageCount": 1}
    layout = {"version": "1.1", "slide": {"width": 640, "height": 360}, "elements": [
        {"id": "plate", "type": "rectangle", "x": 38, "y": 55, "width": 252, "height": 141, "rotation": 0, "zIndex": 5, "groupId": "module_a", "style": {"fill": "#F7F5F2", "stroke": "#F7F5F2", "strokeWidth": 0}, "metadata": {"layerRole": "container", "moduleMemberIds": ["badge", "title"]}},
        {"id": "badge", "type": "image", "x": 60, "y": 72, "width": 64, "height": 64, "rotation": 0, "zIndex": 12, "groupId": "module_a", "src": _data_url(badge), "style": {}, "metadata": {"reconstructionStrategy": "transparent_image"}},
        {"id": "title", "type": "text", "x": 137, "y": 89, "width": 145, "height": 27, "rotation": 0, "zIndex": 20, "groupId": "module_a", "text": "MODULE A", "style": {"fontSize": 19, "color": "#1F273E"}, "metadata": {"reconstructionStrategy": "editable_text"}},
    ]}

    def api(route):
        request = route.request
        url = request.url
        if url.endswith("/api/projects") and request.method == "POST":
            body = project
        elif "/images" in url and request.method == "POST":
            body = {"project": project, "images": [{"id": "image-1", "source_url": _data_url(source)}]}
        elif "/analyze" in url and request.method == "POST":
            body = {"project": project, "slides": [layout], "provider": "local", "warnings": []}
        elif "/vision/status" in url:
            body = {"provider": "local", "configured": False}
        elif "/inpainting/local/status" in url:
            body = {"enabled": False, "connected": False, "model": "lama", "route": ""}
        elif "visual_validation.json" in url:
            body = {"overall": 0.9}
        else:
            body = {}
        route.fulfill(status=200, json=body)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.route("**/api/**", api)
        page.goto("http://127.0.0.1:5173/", wait_until="domcontentloaded")
        page.locator(".toolbar input[type=file]").set_input_files({"name": "complex_modules.png", "mimeType": "image/png", "buffer": fixture.read_bytes()})
        page.get_by_role("button", name=re.compile("AI解析")).click()
        plate = page.locator('[data-element-id="plate"]')
        title = page.locator('[data-element-id="title"]')
        badge_node = page.locator('[data-element-id="badge"]')
        plate.wait_for(state="visible")
        before = [node.bounding_box() for node in (plate, title, badge_node)]
        assert all(before)
        x = before[1]["x"] + before[1]["width"] / 2
        y = before[1]["y"] + before[1]["height"] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + 45, y + 25, steps=8)
        page.mouse.up()
        after = [node.bounding_box() for node in (plate, title, badge_node)]
        dx = [new["x"] - old["x"] for old, new in zip(before, after)]
        dy = [new["y"] - old["y"] for old, new in zip(before, after)]
        assert min(dx) > 15 and max(dx) - min(dx) < 2, dx
        assert min(dy) > 10 and max(dy) - min(dy) < 2, dy
        print(f"module drag passed: dx={dx[0]:.1f}, dy={dy[0]:.1f}; plate, text and badge moved together")
        browser.close()


if __name__ == "__main__":
    main()
