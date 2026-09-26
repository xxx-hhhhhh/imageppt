from __future__ import annotations

import copy
import json
import re
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
from app.services.reconstruction.planner import AIReconstructionPlanner
from app.services.reconstruction.quality_guard import preserve_bad_text_regions
from app.services.refinement import TypographyLayoutRefiner


class ReconstructionPipeline:
    def __init__(self, store: ProjectStore | None = None) -> None:
        self.store = store or ProjectStore()
        self.layout_service = LayoutService()
        self.scene_analyzer = SceneAnalyzer(LAYOUT_PROVIDER, VISION_PROVIDER)
        self.segmentation_provider, self.segmentation_warnings = create_segmentation_provider(SEGMENTATION_PROVIDER)
        self.reconstruction_router = ReconstructionRouter()
        self.reconstruction_planner = AIReconstructionPlanner()
        self.typography_layout_refiner = TypographyLayoutRefiner()

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _vision_debug_payload(self) -> dict:
        provider = self.scene_analyzer.vision_provider
        if hasattr(provider, "vision_debug"):
            payload = dict(provider.vision_debug())
        else:
            payload = {
                "provider": getattr(provider, "provider_name", getattr(provider, "name", "local")),
                "model": getattr(provider, "model_name", getattr(provider, "model", None)),
                "rawResponseAvailable": False,
                "repairUsed": False,
                "normalizationApplied": False,
                "validationErrors": [],
                "droppedElements": 0,
                "normalizationWarnings": [],
            }
        return {
            "provider": payload.get("provider", "qwen"),
            "model": payload.get("model"),
            "rawResponseAvailable": bool(payload.get("rawResponseAvailable")),
            "repairUsed": bool(payload.get("repairUsed")),
            "normalizationApplied": bool(payload.get("normalizationApplied")),
            "validationErrors": [_redact_debug_text(value) for value in (payload.get("validationErrors") or [])],
            "droppedElements": int(payload.get("droppedElements") or 0),
            "normalizationWarnings": [_redact_debug_text(value) for value in (payload.get("normalizationWarnings") or [])],
        }

    def _apply_refined_scene(self, layout: dict, scene: dict) -> dict:
        by_id = {item["id"]: item for item in scene.get("elements", [])}
        refined = copy.deepcopy(layout)
        existing_ids = {item["id"] for item in refined.get("elements", [])}
        for scene_item in scene.get("elements", []):
            if scene_item["id"] not in existing_ids and scene_item.get("type") == "image" and scene_item.get("src"):
                refined.setdefault("elements", []).append({"id": scene_item["id"], "type": "image", "src": scene_item["src"], "rotation": 0, "style": copy.deepcopy(scene_item.get("style") or {}), "metadata": {}})
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
            for field in ("role", "componentType", "groupId", "visionConfidence", "finalConfidence"):
                if field in scene_item:
                    item[field] = copy.deepcopy(scene_item[field])
            scene_metadata = scene_item.get("metadata") or {}
            metadata = dict(item.get("metadata") or {})
            for field in ("reconstructionStrategy", "visionSemanticType", "doNotVectorize", "visualComplexity", "visionMatched", "reconstructionStrategySource", "suppressed", "suppressRender", "ownedBy", "plannerModuleId", "preserveWholeAsset", "textCleanedFromAsset", "textCleaned", "editableTextIds", "fallbackReason"):
                if field in scene_metadata:
                    metadata[field] = copy.deepcopy(scene_metadata[field])
            metadata["sceneId"] = scene_item["id"]
            item["metadata"] = metadata
            scene_style = scene_item.get("style") or {}
            layout_style = dict(item.get("style") or {})
            for field in ("fontClass", "fontFamily", "fontWeight", "fontSize", "align"):
                if field in scene_style:
                    layout_style[field] = copy.deepcopy(scene_style[field])
            item["style"] = layout_style
        refined["sceneVersion"] = scene.get("version", "2.0")
        refined["coordinateSystem"] = "source-pixels-left-top"
        return refined

    def analyze_project(self, project_id: str, mode: str | None = None, page: int = 1, allow_fallback: bool = False) -> tuple[list[dict], str, list[str]]:
        conversion_mode = mode if mode in {"fast", "standard", "high_quality", "maximum"} else CONVERSION_MODE
        record = self.store.get(project_id)
        ocr = OCRService(OCR_PROVIDER)
        inpainting = InpaintingService(INPAINT_PROVIDER)
        inpainting.force_clean = False
        inpainting.prefer_inpaint = conversion_mode in {"high_quality", "maximum"}
        slides: list[dict] = []
        warnings = list(ocr.warnings) + list(inpainting.warnings) + self.segmentation_warnings + self.scene_analyzer.layout_warnings + self.scene_analyzer.vlm_warnings
        project_output = OUTPUTS_DIR / project_id
        reconstruction_stats = {
            "aiUsed": False,
            "visionProvider": "local",
            "visionModel": None,
            "visionMatchedElements": 0,
            "aiStrategiesApplied": 0,
            "criticRounds": 0,
            "criticAdjustmentsApplied": 0,
            "textBlocksMerged": 0,
            "singleLinePreserved": 0,
            "ghostingRegionsDetected": 0,
            "ghostingRegionsRecleaned": 0,
            "aiBackgroundRepairs": 0,
            "wholeBadgeAssets": 0,
            "duplicateElementsRemoved": 0,
            "badgeForegroundTransparentExtractions": 0,
            "badgeSyntheticBackgroundsSuppressed": 0,
            "duplicateBadgeLayersRemoved": 0,
            "typographyRefined": False,
            "fontRoleAssignments": 0,
            "fontFamilyAdjustments": 0,
            "fontSizeAdjustments": 0,
            "textPositionAdjustments": 0,
            "textboxResizeAdjustments": 0,
            "pageAlignmentAdjustments": 0,
            "plannedModules": 0,
            "wholeImageRegions": 0,
            "plannerSuppressedElements": 0,
            "plannerDuplicateTexts": 0,
            "plannerSnappedRegions": 0,
            "plannerCvVisualRegions": 0,
            "mixedModules": 0,
            "editableTextboxes": 0,
            "suppressedDuplicates": 0,
            "visualTextFallbacks": 0,
            "restoredModules": 0,
        }
        typography_layout_refiner = getattr(self, "typography_layout_refiner", None) or TypographyLayoutRefiner()
        images = record.get("images", [])
        if page < 1 or page > len(images):
            raise ValueError("Page does not exist")
        for page_index, image in [(page, images[page - 1])]:
            source_path = Path(image["path"])
            page_output = project_output
            normalized_path = page_output / "normalized" / f"page_{page_index}.png"
            width, height = preprocess_image(source_path, normalized_path)
            original_path = page_output / "original.png" if page_index == 1 else page_output / f"original_{page_index}.png"
            shutil.copy2(normalized_path, original_path)
            regions = ocr.recognize(normalized_path)
            background_path = page_output / "backgrounds" / f"page_{page_index}.png"
            background_url = f"/media/backgrounds/{project_id}/{background_path.name}"
            layout, _ = self.layout_service.build_layout(normalized_path, width, height, regions, background_url, page_output / "assets")
            preserve_regions = [
                [float(item.get("x", 0)), float(item.get("y", 0)), float(item.get("x", 0)) + float(item.get("width", 0)), float(item.get("y", 0)) + float(item.get("height", 0))]
                for item in layout.get("elements", [])
                if (item.get("metadata") or {}).get("preserveAsImage")
            ]
            segmentation = [] if conversion_mode == "fast" else self.segmentation_provider.segment(normalized_path, page_output / "assets", project_id)
            scene_raw, scene_warnings = self.scene_analyzer.analyze(normalized_path, layout, regions, segmentation, enable_vision=conversion_mode != "fast", mode="fast" if conversion_mode == "fast" else "high" if conversion_mode in {"high_quality", "maximum"} else "standard")
            self._write_json(project_output / "vision_debug.json", self._vision_debug_payload())
            self._write_json(page_output / "routing.json" if page_index == 1 else page_output / f"routing_{page_index}.json", self.scene_analyzer.vision_routing)
            if conversion_mode in {"high_quality", "maximum"} and not allow_fallback and not self.scene_analyzer.vision_routing.get("aiUsed"):
                attempts = self.scene_analyzer.vision_routing.get("attempts") or []
                reason = str(attempts[-1].get("error") or "Qwen 视觉规划不可用") if attempts else "Qwen 视觉规划不可用"
                raise AIUnavailableError(f"{reason}；处理已暂停。请检查 AI 设置并重试，或明确选择基础模式继续。")
            planned = ((scene_raw.get("vision") or {}).get("reconstructionPlan") or {}).get("modules") or []
            if conversion_mode in {"high_quality", "maximum"} and not allow_fallback and not planned:
                raise AIUnavailableError("Qwen 未返回可用的模块规划；处理已暂停，请重试或明确选择基础模式继续。")
            self._write_json(project_output / "vision_debug.json", self._vision_debug_payload())
            scene_refined = self.scene_analyzer.refine(copy.deepcopy(scene_raw))
            planner = getattr(self, "reconstruction_planner", None) or AIReconstructionPlanner()
            planner_stats = planner.apply(scene_refined, normalized_path, page_output / "assets", project_id, page_index)
            self._write_json(page_output / "reconstruction_plan.json" if page_index == 1 else page_output / f"reconstruction_plan_{page_index}.json", scene_refined.get("reconstructionPlan", {}))
            reconstruction_stats["mixedModules"] += sum(1 for item in (scene_refined.get("reconstructionPlan") or {}).get("modules", []) if item.get("reconstructionStrategy") == "mixed_component")
            for key, value in planner_stats.items():
                reconstruction_stats[key] += value
            for item in scene_refined.get("elements", []):
                if (item.get("metadata") or {}).get("reconstructionStrategySource") == "planner" and item.get("type") == "image":
                    box = item["bbox"]
                    preserve_regions.append([box["left"], box["top"], box["left"] + box["width"], box["top"] + box["height"]])
            inpainting.restore_background(normalized_path, regions, background_path, preserve_regions=preserve_regions)
            reconstruction_stats["aiBackgroundRepairs"] += int(getattr(inpainting, "ai_repaired_regions", 0))
            _apply_preserved_text_ownership(layout, inpainting.last_strategies)
            for key in ("textBlocksMerged", "wholeBadgeAssets", "duplicateElementsRemoved", "badgeForegroundTransparentExtractions", "badgeSyntheticBackgroundsSuppressed", "duplicateBadgeLayersRemoved"):
                reconstruction_stats[key] += int(getattr(self.layout_service, "last_stats", {}).get(key, 0))
            for key in ("ghostingRegionsDetected", "ghostingRegionsRecleaned"):
                reconstruction_stats[key] += int(getattr(inpainting, "last_stats", {}).get(key, 0))
            if conversion_mode in {"high_quality", "maximum"}:
                for item in scene_refined.get("elements", []):
                    metadata = item.setdefault("metadata", {})
                    if item.get("type") not in {"text", "background"} and not metadata.get("preserveWholeAsset"):
                        metadata.update({"suppressed": True, "suppressRender": True, "ownedBy": "source_background", "reconstructionStrategySource": "background-ownership"})
            self.reconstruction_router.apply(scene_refined.get("elements", []))
            routing = self.scene_analyzer.vision_routing
            reconstruction_stats["aiUsed"] = reconstruction_stats["aiUsed"] or bool(routing.get("aiUsed"))
            if routing.get("usedProvider") == "qwen":
                reconstruction_stats["visionProvider"] = "qwen"
                reconstruction_stats["visionModel"] = routing.get("usedModel") or reconstruction_stats["visionModel"]
            reconstruction_stats["visionMatchedElements"] += sum(
                1 for item in scene_refined.get("elements", []) if (item.get("metadata") or {}).get("visionMatched")
            )
            reconstruction_stats["aiStrategiesApplied"] += sum(
                1
                for item in scene_refined.get("elements", [])
                if (item.get("metadata") or {}).get("reconstructionStrategySource") == "vision"
            )
            layout = self._apply_refined_scene(layout, scene_refined)
            layout, typography_stats = typography_layout_refiner.refine(layout)
            reconstruction_stats["suppressedDuplicates"] += sum(1 for item in layout.get("elements", []) if (item.get("metadata") or {}).get("duplicateSuppressed"))
            reconstruction_stats["typographyRefined"] = bool(reconstruction_stats["typographyRefined"]) or bool(typography_stats["typographyRefined"])
            for key in ("fontRoleAssignments", "fontFamilyAdjustments", "fontSizeAdjustments", "textPositionAdjustments", "textboxResizeAdjustments", "singleLinePreserved", "pageAlignmentAdjustments"):
                reconstruction_stats[key] += int(typography_stats[key])
            layout.setdefault("metadata", {}).update({"conversionMode": conversion_mode, "sceneProvider": self.scene_analyzer.layout_provider.name, "layoutProvider": self.scene_analyzer.layout_provider.name, "ocrProvider": ocr.provider_name, "visionProvider": routing.get("usedProvider", "none"), "visionModel": routing.get("usedModel"), "requestedVisionProvider": routing.get("requestedProvider"), "segmentationProvider": self.segmentation_provider.name, "backgroundStrategies": inpainting.last_strategies, "typographyLayoutRefinement": typography_stats, "reconstructionPlan": scene_refined.get("reconstructionPlan", {})})
            self._write_json(page_output / "scene_raw.json" if page_index == 1 else page_output / f"scene_raw_{page_index}.json", scene_raw)
            self._write_json(page_output / "scene_refined.json" if page_index == 1 else page_output / f"scene_refined_{page_index}.json", scene_refined)
            self._write_json(page_output / "routing.json" if page_index == 1 else page_output / f"routing_{page_index}.json", routing)
            shutil.copy2(background_path, page_output / "background.png" if page_index == 1 else page_output / f"background_{page_index}.png")
            preview_path = page_output / "reconstructed_preview.png" if page_index == 1 else page_output / f"reconstructed_preview_{page_index}.png"
            render_preview(background_path, layout, preview_path)
            shutil.copy2(preview_path, page_output / "initial_preview.png" if page_index == 1 else page_output / f"initial_preview_{page_index}.png")
            if conversion_mode in {"high_quality", "maximum"}:
                guard = preserve_bad_text_regions(normalized_path, background_path, preview_path, layout, page_output / "assets", page_index)
                reconstruction_stats["visualTextFallbacks"] += guard["preservedTextRegions"]
                reconstruction_stats["restoredModules"] += guard["restoredModules"]
                if guard["preservedTextRegions"]:
                    render_preview(background_path, layout, preview_path)
            critic_rounds = {"fast": 0, "standard": 1, "high_quality": 8, "maximum": 12}[conversion_mode]
            critic_reports: list[dict] = []
            best_score = run_visual_qa(normalized_path, preview_path, page_output, layout)
            if conversion_mode in {"high_quality", "maximum"} and float(best_score.get("overall", 0)) < 0.85:
                guard = preserve_bad_text_regions(normalized_path, background_path, preview_path, layout, page_output / "assets", page_index, minimum_f1=0.8)
                reconstruction_stats["visualTextFallbacks"] += guard["preservedTextRegions"]
                reconstruction_stats["restoredModules"] += guard["restoredModules"]
                if guard["preservedTextRegions"]:
                    render_preview(background_path, layout, preview_path)
                    best_score = run_visual_qa(normalized_path, preview_path, page_output, layout)
            stagnation = 0
            revision_status = "not_requested" if critic_rounds == 0 else "converged"
            if critic_rounds and routing.get("usedProvider") not in {None, "none", "local"}:
                for round_index in range(critic_rounds):
                    critic = self.scene_analyzer.vision_provider.critique_reconstruction(normalized_path, preview_path, scene_refined)
                    critic_reports.append(critic)
                    before_scene, before_layout = copy.deepcopy(scene_refined), copy.deepcopy(layout)
                    before_background = background_path.read_bytes()
                    ghost_boxes = _critic_ghosting_boxes(critic, layout)
                    if ghost_boxes:
                        recleaned = inpainting.reclean_background(background_path, ghost_boxes)
                        reconstruction_stats["ghostingRegionsDetected"] += recleaned
                        reconstruction_stats["ghostingRegionsRecleaned"] += recleaned
                    scene_refined = apply_safe_adjustments(scene_refined, critic)
                    reconstruction_stats["criticRounds"] += 1
                    applied = len((scene_refined.get("criticAdjustments") or {}).get("applied", []))
                    reconstruction_stats["criticAdjustmentsApplied"] += applied
                    layout = self._apply_refined_scene(layout, scene_refined)
                    layout, _ = typography_layout_refiner.refine(layout)
                    render_preview(background_path, layout, preview_path)
                    candidate_score = run_visual_qa(normalized_path, preview_path, page_output, layout)
                    improvement = float(candidate_score.get("overall", 0)) - float(best_score.get("overall", 0))
                    critic["visualImprovement"] = round(improvement, 4)
                    if improvement > 0.002:
                        best_score = candidate_score
                        stagnation = 0
                    else:
                        scene_refined, layout = before_scene, before_layout
                        background_path.write_bytes(before_background)
                        render_preview(background_path, layout, preview_path)
                        stagnation += 1
                    self._write_json(page_output / f"visual_critic_{round_index + 1}.json", critic)
                    if not applied or stagnation >= 2:
                        revision_status = "stagnated" if critic.get("issues") else "converged"
                        break
                else:
                    revision_status = "round_limit_reached"
            if critic_reports:
                self._write_json(page_output / "visual_critic.json", {"rounds": critic_reports})
                self._write_json(page_output / "scene_refined.json" if page_index == 1 else page_output / f"scene_refined_{page_index}.json", scene_refined)
            score = run_visual_qa(normalized_path, preview_path, page_output, layout)
            if page_index > 1 and (page_output / "difference.png").is_file():
                shutil.copy2(page_output / "difference.png", page_output / f"difference_{page_index}.png")
            score["revisionStatus"] = revision_status
            score["revisionRounds"] = len(critic_reports)
            shutil.copy2(preview_path, page_output / "final_preview.png" if page_index == 1 else page_output / f"final_preview_{page_index}.png")
            shutil.copy2(normalized_path, page_output / "source.png" if page_index == 1 else page_output / f"source_{page_index}.png")
            shutil.copy2(background_path, page_output / "clean_background.png" if page_index == 1 else page_output / f"clean_background_{page_index}.png")
            self._write_json(page_output / "visual_score.json" if page_index == 1 else page_output / f"visual_score_{page_index}.json", score)
            self._write_json(page_output / "visual_validation.json" if page_index == 1 else page_output / f"visual_validation_{page_index}.json", score)
            page_archive = page_output / "pages" / f"page_{page_index}"
            page_archive.mkdir(parents=True, exist_ok=True)
            archive_sources = {
                "source.png": normalized_path,
                "clean_background.png": background_path,
                "reconstruction_plan.json": page_output / ("reconstruction_plan.json" if page_index == 1 else f"reconstruction_plan_{page_index}.json"),
                "scene_raw.json": page_output / ("scene_raw.json" if page_index == 1 else f"scene_raw_{page_index}.json"),
                "scene_refined.json": page_output / ("scene_refined.json" if page_index == 1 else f"scene_refined_{page_index}.json"),
                "initial_preview.png": page_output / ("initial_preview.png" if page_index == 1 else f"initial_preview_{page_index}.png"),
                "final_preview.png": preview_path,
                "visual_validation.json": page_output / ("visual_validation.json" if page_index == 1 else f"visual_validation_{page_index}.json"),
                "vision_debug.json": page_output / "vision_debug.json",
            }
            for name, source in archive_sources.items():
                if source.is_file():
                    shutil.copy2(source, page_archive / name)
            for directory in ("module_assets", "text_clean_assets"):
                source_dir = page_output / directory / f"page_{page_index}"
                (page_archive / directory).mkdir(exist_ok=True)
                if source_dir.is_dir():
                    shutil.copytree(source_dir, page_archive / directory, dirs_exist_ok=True)
            warnings.extend(scene_warnings)
            reconstruction_stats["editableTextboxes"] += sum(1 for item in layout.get("elements", []) if item.get("type") == "text" and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy")))
            self.store.save_slide(project_id, page_index, layout)
            slides.append(layout)
        if slides:
            output_path, validation = PPTXRenderer().render_project(project_id, slides)
            shutil.copy2(output_path, project_output / "output.pptx")
            self._write_json(project_output / "conversion_report.json", {
                "projectId": project_id,
                "mode": conversion_mode,
                "ocrProvider": ocr.provider_name,
                "visionProvider": reconstruction_stats["visionProvider"],
                "visionModel": reconstruction_stats["visionModel"],
                "aiUsed": reconstruction_stats["aiUsed"],
                "visionMatchedElements": reconstruction_stats["visionMatchedElements"],
                "aiStrategiesApplied": reconstruction_stats["aiStrategiesApplied"],
                "criticRounds": reconstruction_stats["criticRounds"],
                "criticAdjustmentsApplied": reconstruction_stats["criticAdjustmentsApplied"],
                "textBlocksMerged": reconstruction_stats["textBlocksMerged"],
                "singleLinePreserved": reconstruction_stats["singleLinePreserved"],
                "ghostingRegionsDetected": reconstruction_stats["ghostingRegionsDetected"],
                "ghostingRegionsRecleaned": reconstruction_stats["ghostingRegionsRecleaned"],
                "aiBackgroundRepairs": reconstruction_stats["aiBackgroundRepairs"],
                "wholeBadgeAssets": reconstruction_stats["wholeBadgeAssets"],
                "duplicateElementsRemoved": reconstruction_stats["duplicateElementsRemoved"],
                "badgeForegroundTransparentExtractions": reconstruction_stats["badgeForegroundTransparentExtractions"],
                "badgeSyntheticBackgroundsSuppressed": reconstruction_stats["badgeSyntheticBackgroundsSuppressed"],
                "duplicateBadgeLayersRemoved": reconstruction_stats["duplicateBadgeLayersRemoved"],
                "typographyRefined": reconstruction_stats["typographyRefined"],
                "fontRoleAssignments": reconstruction_stats["fontRoleAssignments"],
                "fontFamilyAdjustments": reconstruction_stats["fontFamilyAdjustments"],
                "fontSizeAdjustments": reconstruction_stats["fontSizeAdjustments"],
                "textPositionAdjustments": reconstruction_stats["textPositionAdjustments"],
                "textboxResizeAdjustments": reconstruction_stats["textboxResizeAdjustments"],
                "pageAlignmentAdjustments": reconstruction_stats["pageAlignmentAdjustments"],
                "plannedModules": reconstruction_stats["plannedModules"],
                "wholeImageRegions": reconstruction_stats["wholeImageRegions"],
                "plannerSuppressedElements": reconstruction_stats["plannerSuppressedElements"],
                "plannerDuplicateTexts": reconstruction_stats["plannerDuplicateTexts"],
                "plannerSnappedRegions": reconstruction_stats["plannerSnappedRegions"],
                "plannerCvVisualRegions": reconstruction_stats["plannerCvVisualRegions"],
                "mixedModules": reconstruction_stats["mixedModules"],
                "editableTextboxes": reconstruction_stats["editableTextboxes"],
                "suppressedDuplicates": reconstruction_stats["suppressedDuplicates"],
                "visualTextFallbacks": reconstruction_stats["visualTextFallbacks"],
                "restoredModules": reconstruction_stats["restoredModules"],
                "revisionStatus": revision_status,
                "requestedVisionProvider": self.scene_analyzer.vision_routing.get("requestedProvider"),
                "routing": self.scene_analyzer.vision_routing,
                "layoutProvider": self.scene_analyzer.layout_provider.name,
                "segmentationProvider": self.segmentation_provider.name,
                "slideCount": len(slides),
                "validation": validation,
                "warnings": sorted(set(warnings + getattr(self.scene_analyzer.vision_provider, "warnings", []))),
            })
        warnings.extend(ocr.warnings)
        warnings.extend(inpainting.warnings)
        return slides, ocr.provider_name, sorted(set(warnings))


class AIUnavailableError(RuntimeError):
    """High quality analysis paused until the user explicitly chooses recovery."""


def _redact_debug_text(value: object) -> str:
    message = str(value)
    message = re.sub(r"(?i)(authorization|api[_ -]?key)\s*[:=]\s*\S+", r"\1=[redacted]", message)
    message = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted]", message)
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted]", message)
    message = re.sub(r"://[^/@\s]+@", "://[redacted]@", message)
    return message[:1000]


def _critic_ghosting_boxes(critic: dict, layout: dict) -> list[list[float]]:
    by_id = {str(item.get("id")): item for item in layout.get("elements", [])}
    boxes: list[list[float]] = []
    for issue in critic.get("issues", []) if isinstance(critic, dict) else []:
        problem = str(issue.get("problem") or "").lower()
        ghosting = bool(issue.get("oldTextGhosting")) or problem == "oldtextghosting"
        if not ghosting:
            continue
        element = by_id.get(str(issue.get("elementId") or issue.get("element_id") or ""))
        if not element or element.get("type") != "text":
            continue
        metadata = element.get("metadata") or {}
        raw = metadata.get("rawOCRBBox")
        if isinstance(raw, list) and len(raw) == 4:
            boxes.append([float(value) for value in raw])
        else:
            x, y = float(element.get("x", 0)), float(element.get("y", 0))
            boxes.append([x, y, x + float(element.get("width", 0)), y + float(element.get("height", 0))])
    return boxes


def _apply_preserved_text_ownership(layout: dict, strategies: list[dict]) -> None:
    preserved = [item.get("bbox") for item in strategies if item.get("sourceContentPreserved") and item.get("reconstructionStrategy") == "preserve_complex_text"]
    for element in layout.get("elements", []):
        if element.get("type") != "text":
            continue
        metadata = element.setdefault("metadata", {})
        raw = metadata.get("rawOCRBBox")
        if not isinstance(raw, list) or len(raw) != 4:
            continue
        if any(_bbox_overlap_ratio(raw, bbox) >= 0.55 for bbox in preserved if isinstance(bbox, list) and len(bbox) == 4):
            metadata.update({
                "preserveAsImage": True,
                "sourceContentPreserved": True,
                "willReconstruct": False,
                "suppressRender": True,
                "reconstructionStrategy": "group",
                "reconstructionStrategySource": "background-preservation",
            })


def _bbox_overlap_ratio(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = map(float, left)
    rx1, ry1, rx2, ry2 = map(float, right)
    overlap = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    return overlap / max(1.0, (lx2 - lx1) * (ly2 - ly1))
