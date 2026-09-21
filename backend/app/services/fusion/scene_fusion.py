from __future__ import annotations

from typing import Any


def fuse_scene(layout: dict[str, Any], ocr_results: list[Any], vision: dict[str, Any]) -> dict[str, Any]:
    """Fuse semantic vision output into CV/OCR geometry without trusting model bboxes."""
    scene_elements = []
    elements = layout.get("elements", [])
    text_elements = [item for item in elements if item.get("type") == "text"]
    vision_items = vision.get("elements", []) if isinstance(vision, dict) else []
    used_text: set[str] = set()
    for item in elements:
        fused = dict(item)
        match = _match_vision(item, vision_items, text_elements, used_text)
        if match:
            role = str(match.get("role") or fused.get("role") or "unknown")
            fused["role"] = _normalize_role(role)
            fused["componentType"] = _component_type(match, fused)
            if match.get("groupId"):
                fused["groupId"] = str(match["groupId"])
            metadata = dict(fused.get("metadata") or {})
            strategy = match.get("reconstructionStrategy")
            if strategy:
                metadata["reconstructionStrategy"] = strategy
            metadata["visionSemanticType"] = match.get("semanticType", "unknown")
            fused["metadata"] = metadata
            if match.get("fontClass") and fused.get("type") == "text":
                fused.setdefault("style", {})["fontClass"] = match["fontClass"]
            if match.get("fontWeight") and fused.get("type") == "text":
                weight = match["fontWeight"]
                fused.setdefault("style", {})["fontWeight"] = 700 if str(weight).lower() in {"bold", "semibold", "700"} else 400
            vision_conf = _confidence(match.get("visionConfidence", vision.get("confidence", 0.5)))
        else:
            vision_conf = 0.0
        ocr_conf = float(fused.get("confidence") or 0.0) if fused.get("type") == "text" else 0.0
        cv_conf = float(fused.get("confidence") or 0.0)
        final = (vision_conf * 0.35 + ocr_conf * 0.4 + cv_conf * 0.25) if vision_conf else cv_conf
        fused["visionConfidence"] = vision_conf
        fused["ocrConfidence"] = ocr_conf
        fused["cvConfidence"] = cv_conf
        fused["finalConfidence"] = round(max(0.0, min(1.0, final)), 4)
        if fused["finalConfidence"] < 0.65 and fused.get("type") in {"image", "ellipse"}:
            fused.setdefault("metadata", {})["preferredFallback"] = "transparent_image"
        scene_elements.append(fused)

    qwen_groups = _fuse_groups(vision.get("groups", []) if isinstance(vision, dict) else [], scene_elements)
    existing_groups = _fuse_groups(layout.get("groups", []), scene_elements)
    return {
        "elements": scene_elements,
        "groups": _merge_groups(existing_groups, qwen_groups),
        "relations": list(layout.get("relations", [])) + list(vision.get("relations", []) if isinstance(vision, dict) else []),
        "repeatedComponents": list(vision.get("repeatedComponents", []) if isinstance(vision, dict) else []),
        "layers": vision.get("layers", []) if isinstance(vision, dict) else [],
        "page": vision.get("page", {}) if isinstance(vision, dict) else {},
    }


def _match_vision(item: dict[str, Any], vision_items: list[dict[str, Any]], text_elements: list[dict[str, Any]], used_text: set[str]) -> dict[str, Any] | None:
    item_id = item.get("id")
    for candidate in vision_items:
        if candidate.get("id") == item_id or candidate.get("ocrId") == item_id:
            return candidate
    if item.get("type") != "text":
        return None
    numeric = _numeric_id(item_id)
    for candidate in vision_items:
        if _numeric_id(candidate.get("ocrId")) == numeric and numeric is not None:
            used_text.add(str(item_id))
            return candidate
    for candidate in vision_items:
        if candidate.get("ocrId") and str(candidate["ocrId"]) not in used_text and candidate.get("semanticType") == "text":
            used_text.add(str(candidate["ocrId"]))
            return candidate
    return None


def _numeric_id(value: Any) -> int | None:
    if value is None:
        return None
    digits = "".join(char for char in str(value) if char.isdigit())
    return int(digits) if digits else None


def _normalize_role(role: str) -> str:
    aliases = {"body": "body_text", "label": "label_text", "title": "main_title", "cardTitle": "card_title"}
    return aliases.get(role, role)


def _component_type(candidate: dict[str, Any], item: dict[str, Any]) -> str:
    semantic = str(candidate.get("semanticType") or "")
    if semantic in {"card", "container"}:
        return "container"
    if semantic in {"icon", "logo"}:
        return "icon"
    return str(item.get("componentType") or semantic or "element")


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _fuse_groups(groups: list[dict[str, Any]], elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = {item["id"] for item in elements}
    numeric_ids = {_numeric_id(item["id"]): item["id"] for item in elements if _numeric_id(item["id"]) is not None}
    result = []
    for group in groups:
        group_id = str(group.get("id") or "")
        members = []
        for member in group.get("members", []):
            resolved = member if member in ids else numeric_ids.get(_numeric_id(member))
            if resolved and resolved not in members:
                members.append(resolved)
        if group_id and members:
            result.append({**group, "id": group_id, "members": members})
    return result


def _merge_groups(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = {item["id"]: item for item in left}
    for item in right:
        result[item["id"]] = {**result.get(item["id"], {}), **item}
    return list(result.values())
