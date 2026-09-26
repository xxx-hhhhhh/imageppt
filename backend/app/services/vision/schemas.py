from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


ReconstructionStrategy = Literal[
    "editable_text",
    "native_shape",
    "transparent_image",
    "local_image",
    "background_image",
    "group",
]


class QwenSceneElement(BaseModel):
    model_config = ConfigDict(extra="allow")
    ocrId: str | None = None
    id: str | None = None
    role: str = "unknown"
    semanticType: str = "unknown"
    groupId: str | None = None
    zLayer: str = "foreground"
    fontClass: str = "unknown"
    fontWeight: int = 400
    alignment: Literal["left", "center", "right"] = "left"
    reconstructionStrategy: ReconstructionStrategy | None = None
    doNotVectorize: bool = False
    visualComplexity: float = Field(default=0.0, ge=0, le=1)
    visionConfidence: float = Field(default=0.5, ge=0, le=1)


class QwenPlanBox(BaseModel):
    left: float = Field(ge=0, le=1)
    top: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class QwenPlanModule(BaseModel):
    id: str
    moduleId: str | None = None
    role: str = "component"
    bbox: QwenPlanBox
    strategy: Literal["editable", "whole_image", "hybrid", "ignore"] = "editable"
    reconstructionStrategy: Literal["editable_text", "native_shape", "whole_image", "mixed_component", "ignore"] | None = None
    visualComplexity: float = Field(default=0.5, ge=0, le=1)
    editablePriority: float = Field(default=0.5, ge=0, le=1)
    preserveWhole: bool = False
    children: list[dict[str, Any]] = Field(default_factory=list)
    ownership: dict[str, Any] = Field(default_factory=dict)
    memberIds: list[str] = Field(default_factory=list)
    editableIds: list[str] = Field(default_factory=list)
    ignoreIds: list[str] = Field(default_factory=list)
    preserveRegions: list[QwenPlanBox] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)


class QwenReconstructionPlan(BaseModel):
    page: dict[str, Any] = Field(default_factory=dict)
    sections: list[dict[str, Any]] = Field(default_factory=list)
    textRegions: list[dict[str, Any]] = Field(default_factory=list)
    visualRegions: list[dict[str, Any]] = Field(default_factory=list)
    modules: list[QwenPlanModule] = Field(default_factory=list)


class QwenScene(BaseModel):
    model_config = ConfigDict(extra="allow")
    page: dict[str, Any] = Field(default_factory=dict)
    regions: list[dict[str, Any]] = Field(default_factory=list)
    elements: list[QwenSceneElement] = Field(default_factory=list)
    groups: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    repeatedComponents: list[dict[str, Any]] = Field(default_factory=list)
    layers: list[str] = Field(default_factory=lambda: ["background", "containers", "images", "icons", "text"])
    confidence: float = Field(default=0.5, ge=0, le=1)
    reconstructionPlan: QwenReconstructionPlan = Field(default_factory=QwenReconstructionPlan)


class QwenCritic(BaseModel):
    model_config = ConfigDict(extra="allow")
    overallAssessment: dict[str, float] = Field(default_factory=dict)
    issues: list[dict[str, Any]] = Field(default_factory=list)


def extract_json(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in vision response")
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Qwen occasionally omits the closing bracket in a four-number bbox,
        # while the rest of its plan is valid. Repair only this narrow case.
        repaired = re.sub(r'("bbox"\s*:\s*\[[^\[\]{}]*)(?=\})', r'\1]', candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            from json_repair import loads as repair_json

            payload = repair_json(candidate)
            if not isinstance(payload, dict):
                raise ValueError("Vision response could not be repaired into a JSON object")
            return payload


def normalize_scene_payload(payload: Any, diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    diag = _diagnostics(diagnostics)
    if not isinstance(payload, dict):
        _warning(diag, "scene: invalid root -> {}")
        payload = {}
    normalized = dict(payload)
    if "page" in normalized and not isinstance(normalized.get("page"), dict):
        _warning(diag, f"page: {normalized.get('page')!r} -> {{}}")
        normalized["page"] = {}
    if "confidence" in normalized:
        normalized["confidence"] = _normalize_probability(normalized["confidence"], 0.5, "confidence", diag)

    raw_elements = _coerce_array(normalized.get("elements"), "elements", diag)
    elements: list[dict[str, Any]] = []
    for index, element in enumerate(raw_elements):
        if not isinstance(element, dict):
            diag["droppedElements"] += 1
            _warning(diag, f"elements[{index}]: non-object dropped")
            continue
        elements.append(normalize_scene_element(element, diagnostics=diag, path=f"elements[{index}]"))
    normalized["elements"] = elements

    for field in ("groups", "relations", "repeatedComponents", "regions"):
        normalized[field] = _dict_array(normalized.get(field), field, diag)
    normalized["layers"] = _normalize_layers(normalized.get("layers"), diag)
    raw_plan = normalized.get("reconstructionPlan")
    raw_modules = raw_plan.get("modules", []) if isinstance(raw_plan, dict) else []
    if not isinstance(raw_modules, list):
        raw_modules = []
    modules = []
    for index, module in enumerate(raw_modules[:40]):
        try:
            candidate = dict(module)
            candidate.setdefault("id", candidate.get("moduleId"))
            candidate["bbox"] = _normalize_plan_box(candidate.get("bbox"))
            candidate["preserveRegions"] = [
                _normalize_plan_box(region.get("bbox", region) if isinstance(region, dict) else region)
                for region in candidate.get("preserveRegions", []) if isinstance(region, (dict, list, tuple))
            ]
            candidate["visualComplexity"] = _normalize_plan_score(candidate.get("visualComplexity"), 0.5)
            candidate["editablePriority"] = _normalize_plan_score(candidate.get("editablePriority"), 0.5)
            if isinstance(candidate.get("ownership"), str):
                candidate["ownership"] = {"owner": candidate["ownership"]}
            elif not isinstance(candidate.get("ownership", {}), dict):
                candidate["ownership"] = {}
            modules.append(QwenPlanModule.model_validate(candidate).model_dump(mode="json"))
        except (TypeError, ValueError, ValidationError):
            _warning(diag, f"reconstructionPlan.modules[{index}]: invalid module ignored")
    plan_fields = {key: raw_plan.get(key, {} if key == "page" else []) for key in ("page", "sections", "textRegions", "visualRegions")} if isinstance(raw_plan, dict) else {}
    normalized["reconstructionPlan"] = {**plan_fields, "modules": modules}
    diag["normalizedPayload"] = normalized
    return normalized


def _normalize_plan_box(value: Any) -> dict[str, float]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        left, top, right, bottom = (float(item) for item in value)
        return {"left": left, "top": top, "width": round(right - left, 6), "height": round(bottom - top, 6)}
    if isinstance(value, dict):
        return value
    raise ValueError("invalid plan bbox")


def _normalize_plan_score(value: Any, default: float) -> float:
    if isinstance(value, str):
        labels = {"low": 0.2, "medium": 0.5, "high": 0.8}
        if value.lower() in labels:
            return labels[value.lower()]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number > 1 and number <= 3:
        number /= 3
    return max(0.0, min(1.0, number))


def normalize_scene_element(element: Any, diagnostics: dict[str, Any] | None = None, path: str = "element") -> dict[str, Any]:
    diag = _diagnostics(diagnostics)
    if not isinstance(element, dict):
        _warning(diag, f"{path}: invalid element -> {{}}")
        return {}
    normalized = dict(element)
    for field in ("id", "ocrId", "groupId"):
        if field in normalized and normalized[field] is not None and not isinstance(normalized[field], str):
            original = normalized[field]
            normalized[field] = str(original)
            _warning(diag, f"{path}.{field}: {original!r} -> {normalized[field]!r}")
    for field, default in (("role", "unknown"), ("semanticType", "unknown"), ("zLayer", "foreground"), ("fontClass", "unknown")):
        if field in normalized and not isinstance(normalized[field], str):
            original = normalized[field]
            normalized[field] = default if original is None else str(original)
            _warning(diag, f"{path}.{field}: {original!r} -> {normalized[field]!r}")
    for field, default in (("visualComplexity", 0.0), ("visionConfidence", 0.5), ("confidence", 0.5)):
        if field in normalized:
            normalized[field] = _normalize_probability(normalized[field], default, f"{path}.{field}", diag)
    if "fontWeight" in normalized:
        normalized["fontWeight"] = _normalize_font_weight(normalized["fontWeight"], f"{path}.fontWeight", diag)
    if "alignment" in normalized:
        normalized["alignment"] = _normalize_alignment(normalized["alignment"], f"{path}.alignment", diag)
    if "reconstructionStrategy" in normalized:
        normalized["reconstructionStrategy"] = _normalize_strategy(normalized["reconstructionStrategy"], f"{path}.reconstructionStrategy", diag)
    if "doNotVectorize" in normalized:
        normalized["doNotVectorize"] = _normalize_bool(normalized["doNotVectorize"], f"{path}.doNotVectorize", diag)
    return normalized


def normalize_critic_payload(payload: Any, diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    diag = _diagnostics(diagnostics)
    if not isinstance(payload, dict):
        _warning(diag, "critic: invalid root -> {}")
        payload = {}
    normalized = dict(payload)
    issues = _dict_array(normalized.get("issues"), "issues", diag)
    normalized_issues = []
    for index, issue in enumerate(issues):
        item = dict(issue)
        if not item.get("elementId") and item.get("element_id"):
            item["elementId"] = str(item["element_id"])
            _warning(diag, f"issues[{index}].element_id -> elementId")
        if "oldTextGhosting" in item:
            item["oldTextGhosting"] = _normalize_bool(item["oldTextGhosting"], f"issues[{index}].oldTextGhosting", diag)
        normalized_issues.append(item)
    normalized["issues"] = normalized_issues
    assessment = normalized.get("overallAssessment")
    if not isinstance(assessment, dict):
        if assessment is not None:
            _warning(diag, f"overallAssessment: {assessment!r} -> {{}}")
        assessment = {}
    normalized["overallAssessment"] = {
        str(key): _normalize_probability(value, 0.0, f"overallAssessment.{key}", diag)
        for key, value in assessment.items()
    }
    diag["normalizedPayload"] = normalized
    return normalized


def validate_json(payload: dict[str, Any], kind: str = "scene", diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    diag = _diagnostics(diagnostics)
    if kind != "scene":
        normalized_critic = normalize_critic_payload(payload, diag)
        try:
            return QwenCritic.model_validate(normalized_critic).model_dump(mode="json", exclude_unset=True)
        except ValidationError as exc:
            diag["validationErrors"].extend(_format_validation_errors(exc))
            return QwenCritic(issues=[]).model_dump(mode="json", exclude_unset=True)

    normalized = normalize_scene_payload(payload, diag)
    validated_elements: list[dict[str, Any]] = []
    for index, element in enumerate(normalized["elements"]):
        try:
            validated = QwenSceneElement.model_validate(element)
        except ValidationError as exc:
            diag["validationErrors"].extend(_format_validation_errors(exc, f"elements[{index}]"))
            fallback = {key: element[key] for key in ("id", "ocrId") if key in element}
            validated = QwenSceneElement.model_validate(fallback)
            _warning(diag, f"elements[{index}]: invalid fields replaced with defaults")
        validated_elements.append(validated.model_dump(mode="json", exclude_unset=True))
    normalized["elements"] = validated_elements
    diag["normalizedPayload"] = normalized
    try:
        return QwenScene.model_validate(normalized).model_dump(mode="json", exclude_unset=True)
    except ValidationError as exc:
        errors = _format_validation_errors(exc)
        diag["validationErrors"].extend(errors)
        raise ValueError("Vision validation failed: " + "; ".join(errors)) from exc


def repair_json(value: str | dict[str, Any], kind: str = "scene", diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    return validate_json(extract_json(value), kind, diagnostics)


def _diagnostics(value: dict[str, Any] | None) -> dict[str, Any]:
    diagnostics = value if value is not None else {}
    diagnostics.setdefault("normalizationWarnings", [])
    diagnostics.setdefault("validationErrors", [])
    diagnostics.setdefault("droppedElements", 0)
    return diagnostics


def _warning(diagnostics: dict[str, Any], message: str) -> None:
    diagnostics["normalizationWarnings"].append(message)


def _normalize_probability(value: Any, default: float, path: str, diagnostics: dict[str, Any]) -> float:
    original = value
    aliases = {"high": 0.85, "medium": 0.5, "low": 0.2}
    normalized: float | None = None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in aliases:
            normalized = aliases[text]
        elif text.endswith("%"):
            try:
                normalized = float(text[:-1].strip()) / 100.0
            except ValueError:
                normalized = None
        else:
            try:
                normalized = float(text)
            except ValueError:
                normalized = None
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        normalized = float(value)
    if normalized is not None and 1 < normalized <= 100:
        normalized /= 100.0
    if normalized is None or not 0 <= normalized <= 1:
        normalized = default
    if original != normalized:
        _warning(diagnostics, f"{path}: {original!r} -> {normalized}")
    return normalized


def _normalize_font_weight(value: Any, path: str, diagnostics: dict[str, Any]) -> int:
    aliases = {"normal": 400, "regular": 400, "sans": 400, "serif": 400, "medium": 500, "semibold": 600, "bold": 700, "bold-sans": 700, "bold sans": 700, "black": 900}
    original = value
    if isinstance(value, str) and value.strip().lower() in aliases:
        normalized = aliases[value.strip().lower()]
    else:
        try:
            numeric = int(float(value))
            normalized = min((400, 500, 600, 700, 900), key=lambda item: abs(item - numeric))
        except (TypeError, ValueError):
            normalized = 400
    if original != normalized:
        _warning(diagnostics, f"{path}: {original!r} -> {normalized}")
    return normalized


def _normalize_alignment(value: Any, path: str, diagnostics: dict[str, Any]) -> str:
    aliases = {"left": "left", "centre": "center", "center": "center", "middle": "center", "right": "right", "左对齐": "left", "居中": "center", "右对齐": "right"}
    original = value
    normalized = aliases.get(str(value).strip().lower(), "left")
    if original != normalized:
        _warning(diagnostics, f"{path}: {original!r} -> {normalized!r}")
    return normalized


def _normalize_strategy(value: Any, path: str, diagnostics: dict[str, Any]) -> str | None:
    aliases = {
        "editable_text": "editable_text",
        "text": "editable_text",
        "native_shape": "native_shape",
        "shape": "native_shape",
        "transparent_image": "transparent_image",
        "transparent": "transparent_image",
        "local_image": "local_image",
        "image": "local_image",
        "background_image": "background_image",
        "background": "background_image",
        "group": "group",
    }
    original = value
    normalized = aliases.get(str(value).strip().lower()) if value is not None else None
    if original != normalized:
        _warning(diagnostics, f"{path}: {original!r} -> {normalized!r}")
    return normalized


def _normalize_bool(value: Any, path: str, diagnostics: dict[str, Any]) -> bool:
    original = value
    if isinstance(value, bool):
        normalized = value
    elif isinstance(value, (int, float)):
        normalized = bool(value)
    else:
        text = str(value).strip().lower()
        normalized = text in {"true", "1", "yes", "y", "on"}
    if original != normalized:
        _warning(diagnostics, f"{path}: {original!r} -> {normalized}")
    return normalized


def _coerce_array(value: Any, path: str, diagnostics: dict[str, Any]) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, dict):
        if not value:
            return []
        if any(key in value for key in ("id", "ocrId", "name", "type", "role")):
            _warning(diagnostics, f"{path}: object -> single-item array")
            return [value]
        if all(isinstance(item, dict) for item in value.values()):
            _warning(diagnostics, f"{path}: object map -> array")
            return [{**item, "id": item.get("id", str(key))} for key, item in value.items()]
        _warning(diagnostics, f"{path}: unusable object -> []")
        return []
    _warning(diagnostics, f"{path}: {type(value).__name__} -> []")
    return []


def _dict_array(value: Any, path: str, diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(_coerce_array(value, path, diagnostics)):
        if isinstance(item, dict):
            result.append(item)
        else:
            _warning(diagnostics, f"{path}[{index}]: non-object ignored")
    return result


def _normalize_layers(value: Any, diagnostics: dict[str, Any]) -> list[str]:
    layers: list[str] = []
    for index, item in enumerate(_coerce_array(value, "layers", diagnostics)):
        name = item.get("name") if isinstance(item, dict) else item
        if isinstance(name, str) and name.strip():
            layers.append(name.strip())
        else:
            _warning(diagnostics, f"layers[{index}]: invalid item ignored")
    return layers


def _format_validation_errors(error: ValidationError, prefix: str | None = None) -> list[str]:
    result: list[str] = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in item.get("loc", []))
        path = ".".join(part for part in (prefix, location) if part)
        result.append(f"{path}: {item.get('msg', 'invalid value')}")
    return result


__all__ = [
    "QwenScene",
    "QwenSceneElement",
    "QwenCritic",
    "extract_json",
    "normalize_scene_payload",
    "normalize_scene_element",
    "normalize_critic_payload",
    "validate_json",
    "repair_json",
]
