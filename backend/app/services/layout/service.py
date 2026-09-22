from __future__ import annotations

from pathlib import Path

from app.services.layout.detectors import detect_image_regions, detect_simple_shapes, detect_text_elements
from app.services.layout.label_detector import detect_label_groups
from app.services.layout.position_refiner import refine_layout
from app.services.layout.text_blocks import group_text_elements
from app.services.ocr.provider import OCRResult


class LayoutService:
    def __init__(self) -> None:
        self.last_stats = {"textBlocksMerged": 0, "singleLinePreserved": 0, "wholeBadgeAssets": 0, "duplicateElementsRemoved": 0}

    def analyze(self, image_path: Path, background_url: str | None = None, asset_dir: Path | None = None) -> tuple[dict, list[OCRResult]]:
        from PIL import Image
        with Image.open(image_path) as image:
            width, height = image.size
        return self.build_layout(image_path, width, height, [], background_url, asset_dir)

    def build_layout(
        self,
        image_path: Path,
        width: int,
        height: int,
        ocr_results: list[OCRResult],
        background_url: str | None,
        asset_dir: Path | None,
    ) -> tuple[dict, list[OCRResult]]:
        texts, text_stats = group_text_elements(detect_text_elements(ocr_results), width, height)
        shapes = detect_simple_shapes(image_path, ocr_results)
        project_id = asset_dir.parent.name if asset_dir else None
        elements = detect_label_groups(image_path, width, height, ocr_results, texts + shapes, asset_dir, project_id)
        self.last_stats = {
            **text_stats,
            "wholeBadgeAssets": sum(1 for item in elements if (item.get("metadata") or {}).get("wholeBadgeAsset")),
            "duplicateElementsRemoved": sum(1 for item in elements if (item.get("metadata") or {}).get("duplicateSuppressed")),
        }
        images = detect_image_regions(image_path, ocr_results, shapes)
        if asset_dir and images:
            asset_dir.mkdir(parents=True, exist_ok=True)
            import cv2
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            for item in images:
                x, y, w, h = [int(item[key]) for key in ("x", "y", "width", "height")]
                crop = image[max(0, y):min(height, y + h), max(0, x):min(width, x + w)]
                asset_path = asset_dir / f"{item['id']}.png"
                cv2.imwrite(str(asset_path), crop)
                project_id = asset_dir.parent.name
                item["src"] = f"/media/assets/{project_id}/{asset_path.name}"
        elements = elements + images
        if background_url:
            elements.insert(0, {
                "id": "background_001", "type": "background", "x": 0, "y": 0,
                "width": width, "height": height, "rotation": 0, "zIndex": 0,
                "src": background_url, "style": {"opacity": 1},
            })
        elements = refine_layout(elements, width, height)
        layout = {"version": "1.1", "slide": {"width": width, "height": height}, "source": str(image_path.name), "backgroundUrl": background_url, "coordinateSystem": "source-pixels-left-top", "elements": sorted(elements, key=lambda item: item.get("zIndex", 0))}
        return layout, ocr_results
