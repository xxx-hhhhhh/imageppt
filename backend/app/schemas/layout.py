from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.project import ProjectResponse


ElementType = Literal[
    "text", "image", "rectangle", "roundedRectangle", "ellipse", "line",
    "arrow", "group", "background"
]


class ElementStyle(BaseModel):
    model_config = ConfigDict(extra="allow")
    fontClass: str = "unknown"
    fontFamily: str = "Microsoft YaHei"
    fontSize: float = 24
    fontWeight: int = 400
    fontStyle: Literal["normal", "italic"] = "normal"
    color: str = "#111827"
    align: Literal["left", "center", "right"] = "left"
    lineSpacing: float = 1.12
    letterSpacing: float = 0
    verticalAlign: Literal["top", "middle", "bottom"] = "top"
    fill: str = "#DCE6F1"
    stroke: str = "#17365D"
    strokeWidth: float = 1
    opacity: float = Field(default=1, ge=0, le=1)


class LayoutElement(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    type: ElementType
    x: float
    y: float
    width: float = Field(ge=0)
    height: float = Field(ge=0)
    rotation: float = 0
    zIndex: int = 0
    text: str | None = None
    src: str | None = None
    crop: dict[str, float] | None = None
    style: ElementStyle = Field(default_factory=ElementStyle)
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    groupId: str | None = None
    role: str | None = None
    componentType: str | None = None


class SlideSize(BaseModel):
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class LayoutJSON(BaseModel):
    model_config = ConfigDict(extra="allow")
    version: str = "1.0"
    slide: SlideSize
    source: str | None = None
    backgroundUrl: str | None = None
    coordinateSystem: str = "source-pixels-left-top"
    elements: list[LayoutElement] = Field(default_factory=list)


class ProjectInfo(BaseModel):
    id: str
    name: str
    createdAt: str
    imageCount: int = 0


class AnalyzeResponse(BaseModel):
    project: ProjectResponse
    slides: list[LayoutJSON]
    provider: str
    ocrProvider: str | None = None
    visionProvider: str | None = None
    aiUsed: bool = False
    visionModel: str | None = None
    requestedVisionProvider: str | None = None
    fallbackCount: int = 0
    visionWarning: str | None = None
    warnings: list[str] = Field(default_factory=list)
