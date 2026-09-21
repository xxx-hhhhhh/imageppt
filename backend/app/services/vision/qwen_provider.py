from __future__ import annotations

from app.services.settings.runtime_settings import VisionSettings, load_vision_settings

from .openai_compatible_provider import OpenAICompatibleVisionProvider, VisionProviderError, image_to_data_url
from .proxy import resolve_proxy


class QwenProvider(OpenAICompatibleVisionProvider):
    name = "qwen"

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None, vision_settings: VisionSettings | None = None) -> None:
        runtime = vision_settings or load_vision_settings()
        config = runtime.providers["qwen"]
        proxy = resolve_proxy(runtime.proxy_mode, runtime.manual_proxy)
        super().__init__(provider_name="qwen", api_key=api_key if api_key is not None else config.api_key, base_url=base_url if base_url is not None else config.base_url, model=model or config.model or "qwen3-vl-flash", timeout=runtime.timeout, max_retries=1, proxy_url=proxy.url)


def create_qwen_provider(vision_settings: VisionSettings | None = None) -> QwenProvider:
    return QwenProvider(vision_settings=vision_settings)


__all__ = ["QwenProvider", "VisionProviderError", "image_to_data_url", "create_qwen_provider"]
