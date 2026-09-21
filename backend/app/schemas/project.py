from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(default="Image2EditablePPT", min_length=1, max_length=120)


class ProjectResponse(BaseModel):
    id: str
    name: str
    createdAt: str
    imageCount: int = 0


class UploadResponse(BaseModel):
    project: ProjectResponse
    images: list[dict]

