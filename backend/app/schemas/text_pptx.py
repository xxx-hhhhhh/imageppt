from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TextPptxRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=12000)
    slideCount: int = Field(default=6, ge=3, le=12)
    style: Literal['business', 'academic', 'minimal'] = 'business'

    @field_validator('content')
    @classmethod
    def plain_text(cls, value: str) -> str:
        if any(ord(c) < 32 and c not in '\n\r\t' for c in value):
            raise ValueError('正文含不支持的控制字符')
        return value
