from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Callable

import httpx

from app.services.settings.runtime_settings import VisionSettings, load_vision_settings

from .base import NullVisionProvider, VisionProvider
from .proxy import display_proxy_url, proxy_tcp_test, resolve_proxy
from .qwen_provider import QwenProvider


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
        diagnostics = _network_diagnostics(self.settings)
        if diagnostics["proxyTcp"]["status"] == "FAIL":
            return _test_failure("代理不可用：当前系统代理端口不可用，请检查代理软件。", diagnostics)
        if diagnostics["qwenBaseUrl"]["status"] == "FAIL":
            return _test_failure("无法连接 Qwen Base URL", diagnostics)
        candidate = self._candidate()
        if candidate is None:
            diagnostics["qwenApi"] = {"status": "SKIPPED", "message": "Qwen API Key 未配置或 AI 已关闭"}
            return _test_failure("Qwen API Key 未配置或 AI 已关闭", diagnostics)
        _, model = candidate
        try:
            result = _run_connection_test(self._provider(model), model, timeout_seconds=60)
            diagnostics["qwenApi"] = {"status": "PASS" if result.get("success") else "FAIL", "message": result.get("message", ""), "httpStatus": result.get("statusCode")}
            result.update({"provider": "qwen", "model": result.get("model") or model, "diagnostics": diagnostics})
            return result
        except Exception as exc:
            message = _friendly_error(exc)
            diagnostics["qwenApi"] = {"status": "FAIL", "message": message}
            return {"success": False, "provider": "qwen", "model": model, "message": message, "diagnostics": diagnostics}

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


def _run_connection_test(provider: VisionProvider, model: str, timeout_seconds: int = 60) -> dict[str, Any]:
    result: dict[str, Any] = {}
    error: list[Exception] = []

    def worker() -> None:
        try:
            result.update(provider.test_connection())
        except Exception as exc:
            error.append(exc)

    thread = threading.Thread(target=worker, name="vision-test-qwen", daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        return {"success": False, "provider": "qwen", "model": model, "message": "连接超时"}
    if error:
        return {"success": False, "provider": "qwen", "model": model, "message": _friendly_error(error[0]), "statusCode": getattr(error[0], "status_code", getattr(error[0], "code", None))}
    return result or {"success": False, "provider": "qwen", "model": model, "message": "Qwen API 请求失败"}


def _network_diagnostics(settings: VisionSettings) -> dict[str, Any]:
    proxy = resolve_proxy(settings.proxy_mode, settings.manual_proxy)
    tcp = proxy_tcp_test(proxy.url)
    base_url = settings.providers["qwen"].base_url
    diagnostics: dict[str, Any] = {
        "systemProxy": display_proxy_url(resolve_proxy("system").url),
        "proxyUrl": proxy.url or "direct",
        "proxySource": proxy.source,
        "proxyTcp": tcp,
        "qwenBaseUrl": {"status": "SKIPPED", "message": "未执行"},
        "qwenApi": {"status": "SKIPPED", "message": "未执行"},
        "networkPath": proxy.url or "direct",
    }
    if tcp["status"] == "FAIL":
        diagnostics["qwenBaseUrl"] = {"status": "SKIPPED", "message": "代理端口不可用，未发送请求"}
        return diagnostics
    try:
        kwargs = {"proxy": proxy.url, "timeout": 10, "trust_env": False} if proxy.url else {"timeout": 10, "trust_env": False}
        with httpx.Client(**kwargs) as client:
            response = client.get(base_url.rstrip("/"))
        diagnostics["qwenBaseUrl"] = {"status": "PASS", "message": f"HTTP {response.status_code}", "httpStatus": response.status_code}
    except Exception as exc:
        diagnostics["qwenBaseUrl"] = {"status": "FAIL", "message": "无法连接 Qwen Base URL", "detail": type(exc).__name__}
    return diagnostics


__all__ = ["VisionRouter"]
