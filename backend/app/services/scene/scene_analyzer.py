from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.services.scene.alignment_analyzer import analyze_alignment
from app.services.scene.component_classifier import classify_element
from app.services.scene.geometry_optimizer import optimize_layout
from app.services.scene.grid_detector import detect_grid
from app.services.scene.layer_analyzer import infer_layers
from app.services.scene.layout_analyzer import create_layout_provider
from app.services.scene.relation_analyzer import analyze_relations
from app.services.scene.repetition_detector import detect_repetition
from app.services.style.color_analyzer import analyze_colors
from app.services.typography.text_analyzer import analyze_text_regions
from app.services.fusion.scene_fusion import fuse_scene
from app.services.vision.factory import create_vision_provider


class SceneAnalyzer:
    def __init__(self, layout_provider: str = "auto", vlm_provider: str = "none") -> None:
        self.layout_provider, self.layout_warnings = create_layout_provider(layout_provider)
        self.vision_provider, self.vision_warnings = create_vision_provider(vlm_provider)
        self.vlm_provider = self.vision_provider
        self.vlm_warnings = self.vision_warnings
        self.ai_used = False
        local_only = self.vision_provider.provider_name == "none"
        self.vision_routing: dict[str, Any] = {"requestedProvider": "local" if local_only else "qwen", "usedProvider": "local", "usedModel": None, "fallbackCount": 0, "aiUsed": False, "attempts": []}

    def analyze(self, image_path: Path, layout: dict[str, Any], ocr_results: list[Any], segmentation: list[dict[str, Any]], enable_vision: bool = True, mode: str = "standard") -> tuple[dict[str, Any], list[str]]:
        width, height = int(layout["slide"]["width"]), int(layout["slide"]["height"])
        elements: list[dict[str, Any]] = []
        for item in layout.get("elements", []):
            element_type, role = classify_element(item)
            elements.append({"id": item["id"], "type": element_type, "role": role, "bbox": {"left": float(item.get("x", 0)), "top": float(item.get("y", 0)), "width": float(item.get("width", 0)), "height": float(item.get("height", 0))}, "rotation": float(item.get("rotation", 0)), "zIndex": int(item.get("zIndex", 0)), "groupId": item.get("groupId") or (item.get("metadata") or {}).get("groupId"), "editable": item.get("type") not in {"background", "group"}, "confidence": float(item.get("confidence") or 0.7), "text": item.get("text"), "src": item.get("src"), "style": item.get("style") or {}, "metadata": item.get("metadata") or {}})
        text_analysis = analyze_text_regions(ocr_results, width, height)
        provider_regions = self.layout_provider.analyze(image_path)
        relations = analyze_relations(elements) + analyze_alignment(elements) + detect_repetition(elements) + detect_grid(elements)
        groups = [{"id": group_id, "role": next((item.get("role") for item in elements if item.get("groupId") == group_id), "component"), "members": [item["id"] for item in elements if item.get("groupId") == group_id]} for group_id in sorted({item.get("groupId") for item in elements if item.get("groupId")})]
        ocr_payload = [{"id": f"ocr_{index + 1}", "text": item.text, "bbox": item.bbox, "confidence": item.confidence} for index, item in enumerate(ocr_results)]
        if enable_vision:
            try:
                candidates = [
                    {"id": item["id"], "type": item["type"], "bbox": item["bbox"], "text": str(item.get("text") or "")[:80]}
                    for item in elements if item.get("type") != "background"
                ]
                vision = self.vision_provider.analyze_scene(image_path, {"ocr_elements": ocr_payload, "layout_regions": provider_regions, "candidate_elements": candidates[:160], "width": width, "height": height}, mode)
                self.ai_used = self.ai_used or bool(vision.get("aiUsed"))
                self.vision_routing = vision.get("routing") or {"requestedProvider": self.vision_provider.provider_name, "usedProvider": vision.get("provider"), "usedModel": vision.get("model"), "fallbackCount": 0, "aiUsed": self.ai_used, "attempts": []}
            except Exception as exc:
                requested_provider = "local" if self.vision_provider.provider_name == "none" else "qwen"
                reason = _safe_vision_failure(exc)
                vision = {"provider": "local", "page": {}, "regions": [], "elements": [], "groups": [], "relations": [], "repeatedComponents": [], "layers": [], "confidence": 0.0, "aiUsed": False}
                self.vision_routing = {"requestedProvider": requested_provider, "usedProvider": "local", "usedModel": None, "fallbackCount": 1 if requested_provider == "qwen" else 0, "aiUsed": False, "attempts": [{"provider": requested_provider, "success": False, "error": reason}]}
                self.vision_warnings.append(f"{reason}; using OCR+CV fallback")
        else:
            vision = {"provider": "none", "page": {}, "elements": [], "groups": [], "relations": [], "repeatedComponents": [], "layers": [], "confidence": 0.0, "aiUsed": False}
            self.vision_routing = {"requestedProvider": "qwen", "usedProvider": "local", "usedModel": None, "fallbackCount": 0, "aiUsed": False, "attempts": []}
        fused = fuse_scene({"elements": elements, "groups": groups, "relations": relations}, ocr_results, vision)
        scene = {"version": "2.0", "canvas": {"width": width, "height": height, "backgroundColor": analyze_colors(image_path).get("background", "#FFFFFF")}, "regions": provider_regions, "elements": fused["elements"], "groups": fused["groups"], "relations": fused["relations"], "repeatedComponents": fused["repeatedComponents"], "layers": fused["layers"], "page": fused["page"], "styleTokens": analyze_colors(image_path), "confidence": {"ocr": sum((item.confidence for item in ocr_results), 0.0) / max(1, len(ocr_results)), "layout": self.layout_provider.name, "segmentation": len(segmentation), "vision": vision.get("confidence", 0.0), "final": sum((item.get("finalConfidence", 0.0) for item in fused["elements"]), 0.0) / max(1, len(fused["elements"]))}, "textLines": text_analysis["lines"], "paragraphs": text_analysis["paragraphs"], "segmentation": segmentation, "vision": vision, "visionRouting": self.vision_routing}
        warnings = self.layout_warnings + list(getattr(self.layout_provider, "warnings", [])) + self.vision_warnings + list(getattr(self.vision_provider, "warnings", []))
        return scene, warnings

    def refine(self, scene: dict[str, Any]) -> dict[str, Any]:
        width, height = int(scene["canvas"]["width"]), int(scene["canvas"]["height"])
        layout_elements = []
        for item in scene["elements"]:
            box = item["bbox"]
            layout_elements.append({"id": item["id"], "type": item["type"], "x": box["left"], "y": box["top"], "width": box["width"], "height": box["height"], "rotation": item.get("rotation", 0), "zIndex": item.get("zIndex", 0), "text": item.get("text"), "src": item.get("src"), "style": item.get("style", {}), "metadata": {**item.get("metadata", {}), "groupId": item.get("groupId"), "rawBBox": box}})
        refined = infer_layers(optimize_layout(layout_elements, width, height))
        by_id = {item["id"]: item for item in refined}
        for item in scene["elements"]:
            updated = by_id[item["id"]]
            item["bbox"] = {"left": updated["x"], "top": updated["y"], "width": updated["width"], "height": updated["height"]}
            item.setdefault("metadata", {})["refinedBBox"] = item["bbox"]
            item["zIndex"] = updated["zIndex"]
        scene["refined"] = True
        return scene


def _safe_vision_failure(error: Exception) -> str:
    details = getattr(error, "errors", None)
    if callable(details):
        formatted = []
        for item in details(include_url=False, include_context=False, include_input=False):
            location = ".".join(str(part) for part in item.get("loc", []))
            formatted.append(f"{location}: {item.get('msg', 'invalid value')}")
        message = "Vision validation failed: " + "; ".join(formatted)
    else:
        message = str(error).strip() or type(error).__name__
        if "validation" in type(error).__name__.lower() and not message.lower().startswith("vision validation failed"):
            message = "Vision validation failed: " + message
        elif not message.lower().startswith("vision"):
            message = "Vision analysis failed: " + message
    message = re.sub(r"(?i)(authorization|api[_ -]?key)\s*[:=]\s*\S+", r"\1=[redacted]", message)
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted]", message)
    message = re.sub(r"://[^/@\s]+@", "://[redacted]@", message)
    return message[:1000]
