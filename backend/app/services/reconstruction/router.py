from __future__ import annotations

from typing import Any


_STRATEGIES = {
    "editable_text",
    "native_shape",
    "transparent_image",
    "local_image",
    "background_image",
    "group",
}


class ReconstructionRouter:
    """Choose a safe reconstruction primitive from fused scene evidence.

    Coordinates and pixels remain owned by OCR/CV.  This router only chooses
    how to represent an element in the editable document, so a VLM cannot
    accidentally move an element by inventing geometry.
    """

    def route(self, element: dict[str, Any]) -> str:
        metadata = element.get("metadata") or {}
        requested = metadata.get("reconstructionStrategy")
        if requested in _STRATEGIES:
            return str(requested)

        element_type = element.get("type")
        role = element.get("role") or metadata.get("visualClass")
        if element_type == "text":
            return "editable_text"
        if element_type == "group":
            return "group"
        if element_type == "background" or role in {"background", "photo", "watercolor", "ribbon"}:
            return "background_image"
        if element_type in {"rectangle", "roundedRectangle", "ellipse", "line", "arrow"}:
            confidence = float(element.get("finalConfidence", element.get("confidence", 0.0)) or 0.0)
            complexity = float(metadata.get("visualComplexity", 0.0) or 0.0)
            return "native_shape" if confidence >= 0.65 and complexity < 0.55 else "local_image"
        if element_type == "image":
            return "transparent_image" if metadata.get("transparent", True) else "local_image"
        return "local_image"

    def apply(self, elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for element in elements:
            element.setdefault("metadata", {})["reconstructionStrategy"] = self.route(element)
        return elements


__all__ = ["ReconstructionRouter"]
