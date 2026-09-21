from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from app.config import CONVERSION_MODE, INPAINT_PROVIDER, LAYOUT_PROVIDER, OCR_PROVIDER, OUTPUTS_DIR, SEGMENTATION_PROVIDER, VISION_PROVIDER
from app.services.fusion.adjustment_validator import apply_safe_adjustments
from app.models.project_store import ProjectStore
from app.services.inpainting.service import InpaintingService
from app.services.layout.service import LayoutService
from app.services.ocr.service import OCRService
from app.services.pptx import PPTXRenderer
from app.services.scene.scene_analyzer import SceneAnalyzer
from app.services.segmentation.segmentation_provider import create_segmentation_provider
from app.services.preprocessing.service import preprocess_image
from app.services.visual_qa.analyzer import render_preview, run_visual_qa
from app.services.reconstruction.router import ReconstructionRouter


class ReconstructionPipeline:
    def __init__(self, store: ProjectStore | None = None) -> None:
        self.store = store or ProjectStore()
        self.layout_service = LayoutService()
        self.scene_analyzer = SceneAnalyzer(LAYOUT_PROVIDER, VISION_PROVIDER)
        self.segmentation_provider, self.segmentation_warnings = create_segmentation_provider(SEGMENTATION_PROVIDER)
        self.reconstruction_router = ReconstructionRouter()

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _apply_refined_scene(self, layout: dict, scene: dict) -> dict:
        by_id = {item["id"]: item for item in scene.get("elements", [])}
        refined = copy.deepcopy(layout)
        for item in refined.get("elements", []):
            scene_item = by_id.get(item["id"])
            if not scene_item:
                continue
            box = scene_item.get("bbox") or {}
            item["x"] = float(box.get("left", item.get("x", 0)))
            item["y"] = float(box.get("top", item.get("y", 0)))
            item["width"] = float(box.get("width", item.get("width", 1)))
            item["height"] = float(box.get("height", item.get("height", 1)))
            item["zIndex"] = int(scene_item.get("zIndex", item.get("zIndex", 0)))
            item.setdefault("metadata", {})["sceneId"] = scene_item["id"]
        refined["sceneVersion"] = scene.get("version", "2.0")
        refined["coordinateSystem"] = "source-pixels-left-top"
        return refined

    def analyze_project(self, project_id: str, mode: str | None = None) -> tuple[list[dict], str, list[str]]:
        conversion_mode = mode if mode in {"fast", "standard", "high_quality", "maximum"} else CONVERSION_MODE
        record = self.store.get(project_id)
        ocr = OCRService(OCR_PROVIDER)
        inpainting = InpaintingService(INPAINT_PROVIDER)
        slides: list[dict] = []
        warnings = list(ocr.warnings) + list(inpainting.warnings) + self.segmentation_warnings + self.scene_analyzer.layout_warnings + self.scene_analyzer.vlm_warnings
        project_output = OUTPUTS_DIR / project_id
        for page_index, image in enumerate(record.get("images", []), start=1):
            source_path = Path(image["path"])
            page_output = project_output
            normalized_path = page_output / "normalized" / f"page_{page_index}.png"
            width, height = preprocess_image(source_path, normalized_path)
            original_path = page_output / "original.png" if page_index == 1 else page_output / f"original_{page_index}.png"
            shutil.copy2(normalized_path, original_path)
            regions = ocr.recognize(normalized_path)
            background_path = page_output / "backgrounds" / f"page_{page_index}.png"
            inpainting.restore_background(normalized_path, regions, background_path)
            background_url = f"/media/backgrounds/{project_id}/{background_path.name}"
            layout, _ = self.layout_service.build_layout(normalized_path, width, height, regions, background_url, page_output / "assets")
            segmentation = [] if conversion_mode == "fast" else self.segmentation_provider.segment(normalized_path, page_output / "assets", project_id)
            scene_raw, scene_warnings = self.scene_analyzer.analyze(normalized_path, layout, regions, segmentation, enable_vision=conversion_mode != "fast", mode="fast" if conversion_mode == "fast" else "high" if conversion_mode in {"high_quality", "maximum"} else "standard")
            scene_refined = self.scene_analyzer.refine(copy.deepcopy(scene_raw))
            self.reconstruction_router.apply(scene_refined.get("elements", []))
            layout = self._apply_refined_scene(layout, scene_refined)
            routing = self.scene_analyzer.vision_routing
            layout["metadata"] = {"conversionMode": conversion_mode, "sceneProvider": self.scene_analyzer.layout_provider.name, "visionProvider": routing.get("usedProvider", "none"), "visionModel": routing.get("usedModel"), "requestedVisionProvider": routing.get("requestedProvider"), "segmentationProvider": self.segmentation_provider.name, "backgroundStrategies": inpainting.last_strategies}
            self._write_json(page_output / "scene_raw.json" if page_index == 1 else page_output / f"scene_raw_{page_index}.json", scene_raw)
            self._write_json(page_output / "scene_refined.json" if page_index == 1 else page_output / f"scene_refined_{page_index}.json", scene_refined)
            self._write_json(page_output / "routing.json" if page_index == 1 else page_output / f"routing_{page_index}.json", routing)
            shutil.copy2(background_path, page_output / "background.png" if page_index == 1 else page_output / f"background_{page_index}.png")
            preview_path = page_output / "reconstructed_preview.png" if page_index == 1 else page_output / f"reconstructed_preview_{page_index}.png"
            render_preview(background_path, layout, preview_path)
            critic_rounds = 3 if conversion_mode == "maximum" else 1 if conversion_mode == "high_quality" else 0
            critic_reports: list[dict] = []
            if critic_rounds and routing.get("usedProvider") not in {None, "none"}:
                for round_index in range(critic_rounds):
                    critic = self.scene_analyzer.vision_provider.critique_reconstruction(normalized_path, preview_path, scene_refined)
                    critic_reports.append(critic)
                    scene_refined = apply_safe_adjustments(scene_refined, critic)
                    layout = self._apply_refined_scene(layout, scene_refined)
                    render_preview(background_path, layout, preview_path)
                    self._write_json(page_output / f"visual_critic_{round_index + 1}.json", critic)
            if critic_reports:
                self._write_json(page_output / "visual_critic.json", {"rounds": critic_reports})
            score = run_visual_qa(normalized_path, preview_path, page_output, layout)
            self._write_json(page_output / "visual_score.json" if page_index == 1 else page_output / f"visual_score_{page_index}.json", score)
            self._write_json(page_output / "visual_validation.json" if page_index == 1 else page_output / f"visual_validation_{page_index}.json", score)
            warnings.extend(scene_warnings)
            self.store.save_slide(project_id, page_index, layout)
            slides.append(layout)
        if slides:
            output_path, validation = PPTXRenderer().render_project(project_id, slides)
            shutil.copy2(output_path, project_output / "output.pptx")
            self._write_json(project_output / "conversion_report.json", {"projectId": project_id, "mode": conversion_mode, "ocrProvider": ocr.provider_name, "visionProvider": self.scene_analyzer.vision_routing.get("usedProvider", "none"), "visionModel": self.scene_analyzer.vision_routing.get("usedModel"), "requestedVisionProvider": self.scene_analyzer.vision_routing.get("requestedProvider"), "routing": self.scene_analyzer.vision_routing, "layoutProvider": self.scene_analyzer.layout_provider.name, "segmentationProvider": self.segmentation_provider.name, "slideCount": len(slides), "validation": validation, "warnings": sorted(set(warnings + getattr(self.scene_analyzer.vision_provider, "warnings", [])))})
        warnings.extend(ocr.warnings)
        warnings.extend(inpainting.warnings)
        return slides, ocr.provider_name, sorted(set(warnings))
