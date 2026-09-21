from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QwenSceneElement(BaseModel):
    model_config = ConfigDict(extra="allow")
    ocrId: str | None = None
    id: str | None = None
    role: str = "unknown"
    semanticType: str = "unknown"
    groupId: str | None = None
    zLayer: str = "foreground"
    fontClass: str = "unknown"
    fontWeight: int | str = 400
    alignment: str = "left"
    reconstructionStrategy: str | None = None
    doNotVectorize: bool = False
    visualComplexity: float = Field(default=0.0, ge=0, le=1)
    visionConfidence: float = Field(default=0.5, ge=0, le=1)


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
    return json.loads(text[start:end + 1])


def validate_json(payload: dict[str, Any], kind: str = "scene") -> dict[str, Any]:
    model = QwenScene if kind == "scene" else QwenCritic
    return model.model_validate(payload).model_dump(mode="json", exclude_unset=True)


def repair_json(value: str | dict[str, Any], kind: str = "scene") -> dict[str, Any]:
    return validate_json(extract_json(value), kind)
