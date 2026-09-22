from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from app.services.settings.runtime_settings import VisionSettings, load_vision_settings

from .base import NullVisionProvider, VisionProvider
from .openai_compatible_provider import VisionProviderError
from .qwen_provider import QwenProvider, test_qwen_connectivity


class VisionRouter(VisionProvider):
    """Stable pipeline boundary for the single supported Qwen provider."""

    name = "qwen"

    def __init__(self, vision_settings: VisionSettings | None = None, provider_factory: Callable[..., VisionProvider] | None = None) -> None:
        self.settings = vision_settings or load_vision_settings()
        self.provider_factory = provider_factory
        self.used_provider = "local"
        self.used_model: str | None = None
        self.requested_provider = "qwen" if self.settings.selected_provider != "local" else "local"
        self.attempts: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self._active: VisionProvider = NullVisionProvider()

    def _provider(self, model: str | None = None) -> VisionProvider:
        if self.provider_factory:
            return self.provider_factory("qwen", model)
        return QwenProvider(vision_settings=self.settings, model=model or "qwen3-vl-flash")

    def _configured(self) -> bool:
        config = self.settings.providers["qwen"]
        return bool(self.settings.enabled and config.enabled and config.api_key and config.base_url and config.model)

    def _candidate(self) -> tuple[str, str] | None:
        if self.settings.selected_provider == "local" or not self.settings.enabled or not self._configured():
            return None
        return "qwen", self.settings.providers["qwen"].model or "qwen3-vl-flash"

    def analyze_scene(self, image_path: Path, context: dict | None = None, mode: str = "standard") -> dict[str, Any]:
        self.attempts = []
        candidate = self._candidate()
        if candidate is None:
            self._active = NullVisionProvider()
            self.used_provider = "local"
            self.used_model = None
            return _local_scene(self.routing_result())
        _, model = candidate
        try:
            provider = self._provider(model)
            self.attempts.append({"provider": "qwen", "model": provider.model_name, "success": False})
            result = provider.analyze_scene(image_path, context, mode)
            self.attempts[-1]["success"] = True
            self._active = provider
            self.used_provider = "qwen"
            self.used_model = provider.model_name
            return {**result, "provider": "qwen", "model": self.used_model, "aiUsed": True, "routing": self.routing_result()}
        except Exception as exc:
            if self.attempts:
                self.attempts[-1]["error"] = _friendly_error(exc)
            self.warnings.append(f"Qwen unavailable: {_friendly_error(exc)}")
            self._active = NullVisionProvider()
            self.used_provider = "local"
            self.used_model = None
            return _local_scene(self.routing_result())

    def critique_reconstruction(self, original_path: Path, reconstructed_path: Path, scene: dict) -> dict[str, Any]:
        if self.used_provider != "qwen":
            return {"provider": "local", "issues": []}
        try:
            return self._active.critique_reconstruction(original_path, reconstructed_path, scene)
        except Exception as exc:
            self.warnings.append(f"Qwen critic unavailable: {_friendly_error(exc)}")
            return {"provider": "qwen", "issues": []}

    def test_connection(self) -> dict[str, Any]:
        return test_qwen_connectivity(self.settings)

    def routing_result(self) -> dict[str, Any]:
        fallback_count = 1 if self.requested_provider == "qwen" and self.used_provider == "local" and bool(self.attempts) else 0
        return {"requestedProvider": self.requested_provider, "attempts": self.attempts, "usedProvider": self.used_provider, "usedModel": self.used_model, "fallbackCount": fallback_count, "aiUsed": self.used_provider == "qwen"}

    def status(self) -> dict[str, Any]:
        candidate = self._candidate()
        return {"provider": self.used_provider if self.used_provider == "qwen" else self.requested_provider, "configured": candidate is not None, "model": self.used_model or (candidate[1] if candidate else None)}


def _local_scene(routing: dict[str, Any]) -> dict[str, Any]:
    return {"provider": "local", "model": None, "aiUsed": False, "page": {}, "regions": [], "elements": [], "groups": [], "relations": [], "repeatedComponents": [], "layers": [], "confidence": 0.0, "routing": routing}


def _test_failure(message: str, diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {"success": False, "provider": "qwen", "model": None, "message": message, "diagnostics": diagnostics}


def _friendly_error(error: Exception) -> str:
    if isinstance(error, VisionProviderError):
        return str(error)[:1000]
    code = getattr(error, "status_code", getattr(error, "code", None))
    text = str(error).lower()
    if "vision validation failed" in text:
        return str(error)[:1000]
    if code in {401, 403}:
        return "API Key 无效"
    if code == 429:
        return "额度不足或请求频率达到限制"
    if code == 404:
        return "模型不可用或 Base URL 错误"
    if any(term in text for term in ("timeout", "timed out", "deadline", "504")):
        return "连接超时"
    if code == 400:
        return "请求格式错误，请检查模型和 Base URL"
    return "Qwen API 请求失败"


__all__ = ["VisionRouter"]
