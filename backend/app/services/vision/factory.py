from __future__ import annotations

from app.services.settings.runtime_settings import VisionSettings, load_vision_settings

from .base import NullVisionProvider, VisionProvider
from .qwen_provider import QwenProvider


def create_vision_provider(preferred: str = "qwen", vision_settings: VisionSettings | None = None) -> tuple[VisionProvider, list[str]]:
    """Create only the configured Qwen provider; local mode remains a safe fallback."""
    runtime = vision_settings or load_vision_settings()
    if preferred in {"none", "local", "off"} or not runtime.enabled or runtime.selected_provider == "local":
        return NullVisionProvider(), []
    try:
        return QwenProvider(vision_settings=runtime), []
    except Exception as exc:
        return NullVisionProvider(), [f"Qwen unavailable, using OCR+CV fallback: {exc}"]


__all__ = ["create_vision_provider"]
