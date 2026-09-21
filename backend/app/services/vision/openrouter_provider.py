from __future__ import annotations

from app.services.settings.runtime_settings import VisionSettings, load_vision_settings

from .openai_compatible_provider import OpenAICompatibleVisionProvider
from .proxy import resolve_proxy


class OpenRouterProvider(OpenAICompatibleVisionProvider):
    name = "openrouter"

    def __init__(self, vision_settings: VisionSettings | None = None, model: str | None = None) -> None:
        runtime = vision_settings or load_vision_settings()
        config = runtime.providers["openrouter"]
        proxy = resolve_proxy(runtime.proxy_mode, runtime.manual_proxy)
        super().__init__(provider_name="openrouter", api_key=config.api_key, base_url=config.base_url or "https://openrouter.ai/api/v1", model=model or config.model or "openrouter/free", timeout=runtime.timeout, max_retries=1, proxy_url=proxy.url)


__all__ = ["OpenRouterProvider"]
