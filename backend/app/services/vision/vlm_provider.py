from __future__ import annotations

from .base import NullVisionProvider, VisionProvider


VLMProvider = VisionProvider
NoneVLMProvider = NullVisionProvider


def create_vlm_provider(preferred: str = "none") -> tuple[VisionProvider, list[str]]:
    from .factory import create_vision_provider
    return create_vision_provider(preferred)
