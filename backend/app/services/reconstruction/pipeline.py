from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path

from app.config import (
    CONVERSION_MODE,
    INPAINT_PROVIDER,
    LAYOUT_PROVIDER,
    OCR_PROVIDER,
    OUTPUTS_DIR,
    SEGMENTATION_PROVIDER,
    VISION_PROVIDER,
)
from app.models.project_store import ProjectStore
from app.services.inpainting.service import InpaintingService
from app.services.layout.service import LayoutService
from app.services.ocr.service import OCRService
from app.services.pptx import PPTXRenderer
from app.services.preprocessing.service import preprocess_image
from app.services.reconstruction.object_first import ObjectFirstBuilder
from app.services.reconstruction.revisions import RevisionManager
from app.services.scene.ownership import canonicalize, resolve_asset, scene_view
from app.services.scene.scene_analyzer import SceneAnalyzer
from app.services.segmentation.segmentation_provider import create_segmentation_provider
from app.services.visual_qa.analyzer import render_preview, run_visual_qa


class ReconstructionPipeline:
    def __init__(self, store: ProjectStore | None = None) -> None:
        self.store = store or ProjectStore()
        self.layout_service = LayoutService()
        self.scene_analyzer = SceneAnalyzer(LAYOUT_PROVIDER, VISION_PROVIDER)
        self.segmentation_provider, self.segmentation_warnings = create_segmentation_provider(SEGMENTATION_PROVIDER)

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
            for field in ("reconstructionStrategy", "visionSemanticType", "doNotVectorize", "visualComplexity", "visionMatched", "reconstructionStrategySource", "suppressed", "suppressRender", "ownedBy", "plannerModuleId", "preserveWholeAsset"):
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

    def analyze_project(self, project_id: str, mode: str | None = None) -> tuple[list[dict], str, list[str]]:
        conversion_mode = mode if mode in {"fast", "standard", "high_quality", "maximum"} else CONVERSION_MODE
        record = self.store.get(project_id)
        ocr = OCRService(OCR_PROVIDER)
        inpainting = InpaintingService(INPAINT_PROVIDER)
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
        }
        for page_index, image in enumerate(record.get("images", []), start=1):
            source_path = Path(image["path"])
            page_output = project_output
            fingerprint = hashlib.sha256(source_path.read_bytes()).hexdigest()
            previous = None
            if hasattr(self.store, "get_slide"):
                try:
                    previous = self.store.get_slide(project_id, page_index)
                except FileNotFoundError:
                    pass
            if previous and previous.get("sceneVersion") == "3.0":
                if (previous.get("metadata") or {}).get("sourceFingerprint") != fingerprint:
                    # Page order/source mismatch cannot authorize replacement of
                    # a manually edited graph with a newly generated page.
                    warnings.append(f"Page {page_index}: source mismatch; existing owned objects retained without regeneration")
                    slides.append(previous)
                    continue
                # Repeated optimize is an object-only revision, never page regeneration.
                previous = self._revise_existing(project_id, page_index, source_path, previous, conversion_mode)
                routing = self.scene_analyzer.vision_routing
                reconstruction_stats["aiUsed"] |= bool(routing.get("aiUsed"))
                if routing.get("usedProvider") == "qwen":
                    reconstruction_stats["visionProvider"] = "qwen"
                    reconstruction_stats["visionModel"] = routing.get("usedModel")
                self.store.save_slide(project_id, page_index, previous)
                slides.append(previous)
                continue
            normalized_path = page_output / "normalized" / f"page_{page_index}.png"
            width, height = preprocess_image(source_path, normalized_path)
            original_path = page_output / "original.png" if page_index == 1 else page_output / f"original_{page_index}.png"
            shutil.copy2(normalized_path, original_path)
            regions = ocr.recognize(normalized_path)
            background_path = page_output / "backgrounds" / f"page_{page_index}_{uuid.uuid4().hex[:12]}.png"
            background_url = f"/media/backgrounds/{project_id}/{background_path.name}"
            candidate_builder = getattr(self.layout_service, "build_object_candidates", self.layout_service.build_layout)
            layout, _ = candidate_builder(normalized_path, width, height, regions, background_url, page_output / "assets")
            scene_raw, scene_warnings = self.scene_analyzer.analyze(normalized_path, layout, regions, [], enable_vision=conversion_mode != "fast", mode="fast" if conversion_mode == "fast" else "high" if conversion_mode in {"high_quality", "maximum"} else "standard")
            if hasattr(self.segmentation_provider, "set_prompt_regions"):
                self.segmentation_provider.set_prompt_regions((scene_raw.get("vision") or {}).get("regions") or [])
            segmentation = self.segmentation_provider.segment(normalized_path, page_output / "assets", project_id)
            scene_raw["segmentation"] = segmentation
            self._write_json(project_output / "vision_debug.json", self._vision_debug_payload())
            # CV/VLM candidates cannot suppress pixels. The final owner graph is built
            # from OCR ink + verified geometry + segmentation + exact residual assets.
            layout = ObjectFirstBuilder(OUTPUTS_DIR).build(normalized_path, layout, scene_raw, regions, segmentation, project_id, page_index, background_path, inpainting)
            scene_refined = scene_view(layout)
            for key in ("textBlocksMerged", "wholeBadgeAssets", "duplicateElementsRemoved", "badgeForegroundTransparentExtractions", "badgeSyntheticBackgroundsSuppressed", "duplicateBadgeLayersRemoved"):
                reconstruction_stats[key] += int(getattr(self.layout_service, "last_stats", {}).get(key, 0))
            for key in ("ghostingRegionsDetected", "ghostingRegionsRecleaned"):
                reconstruction_stats[key] += int(getattr(inpainting, "last_stats", {}).get(key, 0))
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
            # Typography heuristics must not move/resize OCR geometry before extraction.
            typography_stats = {"typographyRefined": False, "fontRoleAssignments": 0, "fontFamilyAdjustments": 0, "fontSizeAdjustments": 0, "textPositionAdjustments": 0, "textboxResizeAdjustments": 0, "singleLinePreserved": 0, "pageAlignmentAdjustments": 0}
            reconstruction_stats["typographyRefined"] = bool(reconstruction_stats["typographyRefined"]) or bool(typography_stats["typographyRefined"])
            for key in ("fontRoleAssignments", "fontFamilyAdjustments", "fontSizeAdjustments", "textPositionAdjustments", "textboxResizeAdjustments", "singleLinePreserved", "pageAlignmentAdjustments"):
                reconstruction_stats[key] += int(typography_stats[key])
            layout.setdefault("metadata", {}).update({"conversionMode": conversion_mode, "sceneProvider": self.scene_analyzer.layout_provider.name, "layoutProvider": self.scene_analyzer.layout_provider.name, "ocrProvider": ocr.provider_name, "visionProvider": routing.get("usedProvider", "none"), "visionModel": routing.get("usedModel"), "requestedVisionProvider": routing.get("requestedProvider"), "segmentationProvider": self.segmentation_provider.name, "backgroundStrategies": inpainting.last_strategies, "typographyLayoutRefinement": typography_stats, "reconstructionPlan": scene_refined.get("reconstructionPlan", {})})
            layout["metadata"]["sourceFingerprint"] = fingerprint
            layout["metadata"]["sourceImagePath"] = str(normalized_path.resolve())
            manager = RevisionManager(OUTPUTS_DIR, project_id, page_index)
            _, score = manager.snapshot(layout, normalized_path, background_path, "committed")
            self._write_json(page_output / "scene_raw.json" if page_index == 1 else page_output / f"scene_raw_{page_index}.json", scene_raw)
            self._write_json(page_output / "scene_refined.json" if page_index == 1 else page_output / f"scene_refined_{page_index}.json", scene_refined)
            self._write_json(page_output / "routing.json" if page_index == 1 else page_output / f"routing_{page_index}.json", routing)
            shutil.copy2(background_path, page_output / "background.png" if page_index == 1 else page_output / f"background_{page_index}.png")
            preview_path = page_output / "reconstructed_preview.png" if page_index == 1 else page_output / f"reconstructed_preview_{page_index}.png"
            render_preview(background_path, layout, preview_path)
            critic_rounds = {"fast": 0, "standard": 1, "high_quality": 2, "maximum": 3}[conversion_mode]
            critic_reports: list[dict] = []
            if critic_rounds and routing.get("usedProvider") not in {None, "none", "local"}:
                for round_index in range(critic_rounds):
                    try:
                        critic = self.scene_analyzer.vision_provider.critique_reconstruction(normalized_path, preview_path, scene_refined)
                    except Exception:  # noqa: BLE001 -- optional critic must not destroy the graph
                        critic = {"issues": []}
                        warnings.append("Visual critic unavailable; unchanged candidate will roll back")
                    critic_reports.append(critic)
                    layout, score, decision = manager.revise(layout, normalized_path, background_path, critic, score)
                    scene_refined = scene_view(layout)
                    reconstruction_stats["criticRounds"] += 1
                    reconstruction_stats["criticAdjustmentsApplied"] += len((layout.get("metadata", {}).get("criticAdjustments") or {}).get("applied", [])) if decision["accepted"] else 0
                    render_preview(background_path, layout, preview_path)
                    self._write_json(page_output / f"visual_critic_{round_index + 1}.json", critic)
            if critic_reports:
                self._write_json(page_output / "visual_critic.json", {"rounds": critic_reports})
                self._write_json(page_output / "scene_refined.json" if page_index == 1 else page_output / f"scene_refined_{page_index}.json", scene_refined)
            score.update(run_visual_qa(normalized_path, preview_path, page_output, layout))
            layout["metadata"]["preservationQA"] = score
            self._write_json(page_output / "visual_score.json" if page_index == 1 else page_output / f"visual_score_{page_index}.json", score)
            self._write_json(page_output / "visual_validation.json" if page_index == 1 else page_output / f"visual_validation_{page_index}.json", score)
            warnings.extend(scene_warnings)
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
        warnings.extend(getattr(self.segmentation_provider, "warnings", []))
        return slides, ocr.provider_name, sorted(set(warnings))

    def _revise_existing(self, project_id: str, page: int, source: Path, layout: dict, mode: str) -> dict:
        background = resolve_asset(layout.get("backgroundUrl"), OUTPUTS_DIR)
        if not background or not background.is_file():
            raise ValueError("Previous revision background missing; original project retained")
        original = Path((layout.get("metadata") or {}).get("sourceImagePath") or OUTPUTS_DIR / project_id / "normalized" / f"page_{page}.png")
        if not original.is_file():
            original = source
        manager = RevisionManager(OUTPUTS_DIR, project_id, page)
        previous, qa = manager.snapshot(layout, original, background, "baseline")
        critic = {"issues": []}
        if mode != "fast":
            try:
                critic = self.scene_analyzer.vision_provider.critique_reconstruction(original, previous / "preview.png", scene_view(layout))
                critic["aiUsed"] = critic.get("aiUsed", critic.get("provider") == "qwen")
                critic["model"] = critic.get("model") or getattr(self.scene_analyzer.vision_provider, "model_name", None)
            except Exception as exc:  # noqa: BLE001 -- record failure; empty candidate rolls back
                layout["metadata"]["revisionWarning"] = f"Critic unavailable ({type(exc).__name__}); original objects retained"
        layout, qa, _decision = manager.revise(layout, original, background, critic, qa)
        layout["metadata"]["preservationQA"] = qa
        self.scene_analyzer.vision_routing = {"usedProvider": critic.get("provider", "local"), "usedModel": critic.get("model"), "aiUsed": bool(critic.get("aiUsed")), "requestedProvider": "qwen", "fallbackCount": 0}
        root = OUTPUTS_DIR / project_id
        self._write_json(root / ("scene_refined.json" if page == 1 else f"scene_refined_{page}.json"), scene_view(layout))
        self._write_json(root / ("visual_validation.json" if page == 1 else f"visual_validation_{page}.json"), qa)
        render_preview(background, layout, root / ("reconstructed_preview.png" if page == 1 else f"reconstructed_preview_{page}.png"))
        return canonicalize(layout)


def _redact_debug_text(value: object) -> str:
    message = str(value)
    message = re.sub(r"(?i)(authorization|api[_ -]?key)\s*[:=]\s*\S+", r"\1=[redacted]", message)
    message = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted]", message)
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted]", message)
    message = re.sub(r"://[^/@\s]+@", "://[redacted]@", message)
    return message[:1000]
