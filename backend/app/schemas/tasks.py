from typing import Literal

from pydantic import BaseModel, Field


class AnalysisTaskRequest(BaseModel):
    page: int = Field(default=1, ge=1)
    mode: Literal["fast", "standard", "high_quality", "maximum"] = "maximum"
    allowFallback: bool = False
