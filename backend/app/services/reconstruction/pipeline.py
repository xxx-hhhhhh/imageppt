from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path

from app.config import (
    CONVERSION_MODE,
    LAYOUT_PROVIDER,
    OCR_PROVIDER,
    RECONSTRUCTION_SURFACE_MODE,
    SEGMENTATION_PROVIDER,
    VISION_PROVIDER,
)
from app.models.project_store import ProjectStore
from app.services.layout.service import LayoutService
from app.services.ocr.service import OCRService
from app.services.pptx import PPTXRenderer
from app.services.preprocessing.service import preprocess_image
from app.services.reconstruction.planner import AIReconstructionPlanner
from app.services.reconstruction.router import ReconstructionRouter
from app.services.refinement import TypographyLayoutRefiner
from app.services.scene.scene_analyzer import SceneAnalyzer
from app.services.segmentation.segmentation_provider import create_segmentation_provider
from app.services.visual_qa.analyzer import render_preview, run_visual_qa


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
            for field in ("role", "componentType", "groupId", "visionConfidence", "finalConfidence", "source", "owner", "contour", "editable"):
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

    def analyze_project(self, project_id: str, mode: str | None = None, page: int = 1, allow_fallback: bool = False, *, progress=None, prepare_only: bool = False) -> tuple[list[dict], str, list[str]]:
        """The sole production route: proposals -> verified owners -> background.

        Old erase/repair helpers remain for reading historical artifacts, but
        this entry point cannot invoke them or an automatic revision loop.
        """
        report = progress or (lambda stage: None)
        from app.services.layout.detectors import (
            detect_simple_shapes,
            detect_text_elements,
        )
        from app.services.reconstruction.exclusive_ownership import (
            audit_raster_text,
            build_exclusive_scene,
        )
        if RECONSTRUCTION_SURFACE_MODE != "exclusive_object_first":
            raise ValueError("Erase-first/white-objectized reconstruction is disabled")
        conversion_mode = mode if mode in {"fast", "standard", "high_quality", "maximum"} else CONVERSION_MODE
        record = self.store.get(project_id)
        images = record.get("images", [])
        if not 1 <= page <= len(images):
            raise ValueError("Page does not exist")
        root = self.store.root / project_id
        root.mkdir(parents=True, exist_ok=True)
        # Preserve the previous scene, assets, background and QA before analysis.
        # A failed API call never destroys an already usable result.
        from datetime import datetime, timezone
        from uuid import uuid4
        token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:6]
        candidate = root / "candidates" / token
        candidate.mkdir(parents=True)
        original_source = Path(images[page-1]["path"])
        normalized = candidate / "normalized" / f"page_{page}.png"
        report("preprocessing")
        width, height = preprocess_image(original_source, normalized)
        report("ocr")
        ocr = OCRService(OCR_PROVIDER)
        regions = ocr.recognize(normalized)
        # Do NOT call LayoutService's badge suppressor, pre-cleaner or planner.
        # Scene fusion is used for proposals, never OCR body or position edits.
        ocr_layout = {"version": "1.1", "slide": {"width": width, "height": height}, "elements": detect_text_elements(regions)}
        ocr_layout["elements"].extend(detect_simple_shapes(normalized, regions))
        if hasattr(self.scene_analyzer.vision_provider, "strict"):
            self.scene_analyzer.vision_provider.strict = conversion_mode != "fast" and not allow_fallback
        self.segmentation_provider.text_regions = [e for e in ocr_layout['elements'] if e.get('type')=='text']
        report("segmentation")
        segments = [] if conversion_mode == "fast" else self.segmentation_provider.segment(normalized, candidate / "assets", project_id)
        report("layout")
        self.scene_analyzer.progress = report
        scene, scene_warnings = self.scene_analyzer.analyze(normalized, ocr_layout, regions, segments,
            enable_vision=conversion_mode != "fast", mode="high" if conversion_mode in {"maximum", "high_quality"} else conversion_mode)
        routing = self.scene_analyzer.vision_routing
        self._write_json(candidate / "routing.json", routing)
        self._write_json(candidate / "vision_debug.json", self._vision_debug_payload())
        if conversion_mode != "fast" and not allow_fallback and not routing.get("aiUsed"):
            raise AIUnavailableError("Qwen 页面理解未成功；候选已保留，原结果没有被覆盖。请检查 AI 设置后重试。")
        self._write_json(candidate / "scene_raw.json", scene)
        self._write_json(candidate / "ocr_layout.json", ocr_layout)
        report("text_reconstruction")
        layout = build_exclusive_scene(normalized, ocr_layout, scene, segments, candidate, project_id, page)
        layout["metadata"].update({"conversionMode": conversion_mode, "ocrProvider": ocr.provider_name,
            "visionProvider": routing.get("usedProvider"), "visionModel": routing.get("usedModel"),
            "requestedVisionProvider": routing.get("requestedProvider"), "segmentationProvider": self.segmentation_provider.name})
        report("validation")
        background = candidate / "backgrounds" / f"page_{page}.png"
        # Preview renders the candidate paths before assets have been committed.
        preview_layout = copy.deepcopy(layout)
        for item in preview_layout["elements"]:
            if item.get("src"):
                folder = "backgrounds" if item["type"] == "background" else "assets"
                item["src"] = str(candidate / folder / item["src"].split("/")[-1])
        preview = candidate / "reconstructed_preview.png"
        render_preview(background, preview_layout, preview)
        audit = audit_raster_text(normalized, layout, candidate)
        self._write_json(candidate / "raster_text_audit.json", audit)
        if audit["missingAssetCount"] or audit["rasterNativeDuplicateTextCount"]:
            raise ValueError("Candidate has missing assets or retained original glyphs; previous result kept")
        from app.services.reconstruction.preservation_checks import inspect_preservation
        preservation = inspect_preservation(normalized,layout,candidate,preview)
        self._write_json(candidate / "preservation_checks.json",preservation)
        if any(preservation[k] for k in ("unownedSourcePixels", "multiplyOwnedSourcePixels", "unownedGlyphPaperPixels", "multiplyOwnedGlyphPaperPixels")):
            raise ValueError("Source owners are incomplete or duplicated")
        outside_text_changed = preservation["unexpectedVisualChangedPixels"]
        if outside_text_changed:
            raise ValueError(f"Candidate loses or changes {outside_text_changed} non-text pixels; no commit allowed")
        detected = len(layout["metadata"]["ownershipAudit"]["textDecisions"])
        editable = sum(e.get("owner") == "editable_text" for e in layout["elements"])
        decisions = layout["metadata"]["ownershipAudit"]["textDecisions"]
        editable_lines = sum(d.get('editableCharacterCount',0)==d.get('sourceCharacterCount',1) for d in decisions)
        source_chars = sum(d.get('sourceCharacterCount',0) for d in decisions)
        editable_chars = sum(d.get('editableCharacterCount',0) for d in decisions)
        reliable_ocr = sum(e.get("type") == "text" and float(e.get("confidence") or 0)>=.9 for e in ocr_layout["elements"])
        if reliable_ocr and not editable:
            raise ReconstructionQualityError("检测到了可靠文字，但尚未生成原生文本框。候选已保留，当前结果没有被覆盖；本次不能标记为可编辑重建成功。")
        score = run_visual_qa(normalized, preview, candidate, layout)
        score.update({"editableTextCoverage": editable_lines/max(1,detected), "detectedTextCount": detected,
            "visionProvider": routing.get("usedProvider"), "visionModel": routing.get("usedModel"),
            "editableTextCount": editable_lines, "nativeTextBoxCount":editable,
            "editableCharacterCoverage":editable_chars/max(1,source_chars),
            "nonEditableTextCount": detected-editable_lines, "nonTextChangedPixelCount": outside_text_changed,
            "preservationChecks":preservation,"reliableOCRTextCount":reliable_ocr,
            "nativeReliableTextCoverage":min(1,editable_lines/max(1,reliable_ocr)),
            "imageAssetCount": audit["activeImageCount"], "rasterNativeDuplicateTextCount": 0,
            "revisionRounds": 0, "revisionRound": 0, "revisionStatus": "automatic_revisions_paused",
            "stagnationReason": "ownership_audit_pause",
            "semanticObjectCoverage":None,
            "metricScope": "source-pixel ownership and glyph-local repairs; semantic extraction completeness is not implied; font substitution/verified antialias edges explicit; PowerPoint render separate"})
        from app.services.validation.reconstruction_quality import (
            reconstruction_quality,
        )
        score.update(reconstruction_quality(layout,audit,preservation))
        score['issuesAfter'] = score.get('issues',[]) + score['issuesAfter']
        score['visualSimilarityScore'] = score.get('overall')
        score['overall'] = min(float(score.get('overall') or 0),score['editableCharacterCoverage'])
        self._write_json(candidate / "visual_score.json", score)
        self._write_json(candidate / "visual_validation.json", score)
        self._write_json(candidate / f"scene_final_{page}.json", layout)
        prior_slides = self.store.list_slides(project_id)
        candidate_slides = [preview_layout if index == page else slide for index, slide in enumerate(prior_slides, 1)]
        if page > len(prior_slides):
            candidate_slides.append(preview_layout)
        report("pptx")
        output_path, validation = PPTXRenderer().render_project(project_id, candidate_slides, output_dir=candidate)
        if not validation['valid']:
            raise ReconstructionQualityError('PPTX 重新打开后的内容与 Layout 不一致；候选保留，未覆盖上一版本。')
        warnings = list(ocr.warnings) + scene_warnings + list(getattr(self.segmentation_provider, "warnings", []))
        warnings.append(f"自动优化已暂停；{detected-editable_lines}/{detected} 条文字尚未完整转为可编辑文本。当前为部分可编辑结果，需人工检查字体与图片拆分。")
        self._write_json(candidate / "publication.json", {"projectId": project_id, "page": page,
            "mode": conversion_mode, "ocrProvider": ocr.provider_name, "routing": routing,
            "warnings": warnings, "quality": score, "outputName": output_path.name})
        self.prepared_candidate = candidate
        if not prepare_only:
            report("publishing")
            self.publish_candidate(self.store, candidate)
        return [layout], ocr.provider_name, warnings

    @staticmethod
    def publish_candidate(store: ProjectStore, candidate: Path) -> None:
        """Only the supervisor publishes; a cancelled worker cannot alter results."""
        manifest = json.loads((candidate / "publication.json").read_text(encoding="utf-8"))
        project_id, page = manifest["projectId"], manifest["page"]
        root = store.root / project_id
        if candidate.resolve().parent != (root / "candidates").resolve():
            raise ValueError("Invalid candidate directory")
        token = candidate.name
        layout = json.loads((candidate / f"scene_final_{page}.json").read_text(encoding="utf-8"))
        normalized = candidate / "normalized" / f"page_{page}.png"
        background = candidate / "backgrounds" / f"page_{page}.png"
        preview = candidate / "reconstructed_preview.png"
        output_path = candidate / manifest["outputName"]
        validation = json.loads((candidate / "validation.json").read_text(encoding="utf-8"))
        def write_json(path, payload):
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        # Freeze the prior working artifacts. Never delete an old asset.
        if (root / "slides" / f"page_{page}.json").is_file():
            archive = root / "snapshots" / token
            archive.mkdir(parents=True)
            for name in ("assets", "backgrounds", "slides", "ownership_masks"):
                if (root / name).is_dir():
                    shutil.copytree(root / name, archive / name)
            for filename in ("reconstructed_preview.png", "visual_score.json", "scene_raw.json", "editable.pptx"):
                if (root / filename).is_file():
                    shutil.copy2(root / filename, archive / filename)
        from app.services.reconstruction.publication import preserve_published_state
        suffix = "" if page == 1 else f"_{page}"
        protected_files = [f"slides/page_{page}.json", f"backgrounds/page_{page}.png",
                           f"normalized/page_{page}.png", "editable.pptx", "output.pptx", "validation.json",
                           "conversion_report.json", f"ownership_labels_{page}.png",
                           f"ownership_audit_{page}.json", f"scene_final_{page}.json"]
        protected_files += [f"{name}{suffix}.json" for name in
                            ("scene_raw", "visual_score", "visual_validation", "routing", "vision_debug", "raster_text_audit", "preservation_checks")]
        protected_files += [f"{name}{suffix}.png" for name in
                            ("source", "original", "reconstructed_preview", "final_preview", "initial_preview", "clean_background", "editable_text_removed_preview")]
        with preserve_published_state(root, protected_files):
            for name in ("assets", "ownership_masks", "normalized"):
                if (candidate / name).is_dir():
                    shutil.copytree(candidate / name, root / name, dirs_exist_ok=True)
            (root / "backgrounds").mkdir(exist_ok=True)
            shutil.copy2(background, root / "backgrounds" / background.name)
            suffix = "" if page == 1 else f"_{page}"
            for filename in ("scene_raw", "visual_score", "visual_validation", "routing", "vision_debug", "raster_text_audit", "preservation_checks"):
                write_json(root / f"{filename}{suffix}.json", json.loads((candidate / f"{filename}.json").read_text(encoding="utf-8")))
            for filename in ("source", "original"):
                shutil.copy2(normalized, root / f"{filename}{suffix}.png")
            for filename in ("reconstructed_preview", "final_preview", "initial_preview"):
                shutil.copy2(preview, root / f"{filename}{suffix}.png")
            shutil.copy2(candidate / "editable_text_removed_preview.png",root / f"editable_text_removed_preview{suffix}.png")
            shutil.copy2(background, root / f"clean_background{suffix}.png")
            for filename in (f"ownership_labels_{page}.png", f"ownership_audit_{page}.json", f"scene_final_{page}.json"):
                shutil.copy2(candidate / filename, root / filename)
            store.save_slide(project_id, page, layout)
            shutil.copy2(output_path, root / "editable.pptx")
            shutil.copy2(candidate / "validation.json", root / "validation.json")
            shutil.copy2(output_path, root / "output.pptx")
            routing = manifest["routing"]
            write_json(root / "conversion_report.json", {"projectId": project_id, "mode": manifest["mode"],
                "ocrProvider": manifest["ocrProvider"], "visionProvider": routing.get("usedProvider"), "visionModel": routing.get("usedModel"),
                "aiUsed": bool(routing.get("aiUsed")), "routing": routing, "criticRounds": 0,
                "pipeline": "exclusive_object_first", "revisionStatus": "automatic_revisions_paused", "validation": validation,
                "warnings": manifest["warnings"], "quality": manifest["quality"]})


class AIUnavailableError(RuntimeError):
    """High quality analysis paused until the user explicitly chooses recovery."""


class ReconstructionQualityError(RuntimeError):
    """A raster-only slide containing reliable OCR is not editable success."""


def _ensure_uncovered_text_owners(layout: dict, uncovered_ids: list[str]) -> None:
    """Keep OCR lines missed by the visual plan without duplicating merged lines."""
    elements = layout.get("elements", [])
    active = [item for item in elements if item.get("type") == "text" and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    owned_ids = {str(source_id) for item in active for source_id in ((item.get("metadata") or {}).get("sourceOcrIds") or [item.get("id")])}
    for text_id in uncovered_ids:
        if text_id in owned_ids:
            continue
        item = next((entry for entry in elements if entry.get("id") == text_id and entry.get("type") == "text"), None)
        if item is None:
            continue
        metadata = item.setdefault("metadata", {})
        for key in ("suppressed", "suppressRender", "ownedBy"):
            metadata.pop(key, None)
        metadata.update({"reconstructionStrategy": "editable_text", "reconstructionStrategySource": "ocr_plan_fallback", "textOwner": text_id})
        owned_ids.add(text_id)


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
