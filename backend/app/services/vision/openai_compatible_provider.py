from __future__ import annotations

import base64
import logging
import mimetypes
import time
from pathlib import Path
from typing import Any

import httpx

from .proxy import resolve_proxy
from .base import VisionProvider
from .prompts import CRITIC_PROMPT, SCENE_REPAIR_PROMPT, SCENE_SYSTEM_PROMPT, scene_user_prompt
from .schemas import extract_json, validate_json
from .test_assets import cleanup_test_image, create_test_image


logger = logging.getLogger(__name__)
_PROXY_UNSET = object()
NETWORK_ERROR_TYPES = {"connection_timeout", "connect_error", "tls_error", "network_error"}


class VisionProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        *,
        stage: str = "request",
        error_type: str = "request_error",
        error_code: str | None = None,
        connection_path: str | None = None,
        latency_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.stage = stage
        self.error_type = error_type
        self.error_code = error_code
        self.connection_path = connection_path
        self.latency_ms = latency_ms

    @property
    def network_failure(self) -> bool:
        return self.error_type in NETWORK_ERROR_TYPES


def image_to_data_url(path: Path) -> str:
    allowed = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime = allowed.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]
    if mime not in allowed.values():
        raise ValueError(f"Unsupported vision image type: {path.suffix or 'unknown'}")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


class OpenAICompatibleVisionProvider(VisionProvider):
    def __init__(self, *, provider_name: str, api_key: str, base_url: str, model: str, timeout: int = 120, max_retries: int = 1, proxy_url: str | None | object = _PROXY_UNSET) -> None:
        self.name = provider_name
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.proxy_url = resolve_proxy().url if proxy_url is _PROXY_UNSET else proxy_url
        self.connection_path = _connection_path(self.proxy_url)
        self.last_status_code: int | None = None
        self.last_latency_ms: int | None = None
        self._reset_scene_diagnostics()
        if not api_key:
            raise VisionProviderError(f"{provider_name} API Key 未配置")
        if not base_url:
            raise VisionProviderError(f"{provider_name} Base URL 未配置")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise VisionProviderError("openai package 未安装") from exc
        self._openai_type = OpenAI
        self._configure_client(self.proxy_url)

    def _configure_client(self, proxy_url: str | None) -> None:
        old_http_client = getattr(self, "http_client", None)
        self.proxy_url = proxy_url
        self.connection_path = _connection_path(proxy_url)
        self.http_client = httpx.Client(proxy=proxy_url, timeout=self.timeout, trust_env=False)
        self.client = self._openai_type(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0, http_client=self.http_client)
        if old_http_client is not None:
            try:
                old_http_client.close()
            except Exception:
                pass

    def status(self) -> dict[str, Any]:
        return {"provider": self.name, "configured": True, "model": self.model, "baseUrlConfigured": bool(self.base_url)}

    def _request(self, messages: list[dict[str, Any]], *, repair_prompt: str | None = None) -> str:
        request_messages = [*messages, {"role": "user", "content": repair_prompt}] if repair_prompt else messages
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(model=self.model, messages=request_messages)
                self.last_status_code = 200
                self.last_latency_ms = round((time.perf_counter() - started) * 1000)
                logger.info(
                    "vision_request provider=%s model=%s base_url=%s connection_path=%s status_code=200 latency_ms=%s error_type=none",
                    self.name,
                    self.model,
                    self.base_url,
                    self.connection_path,
                    self.last_latency_ms,
                )
                return str(response.choices[0].message.content or "")
            except Exception as exc:
                last_error = exc
                error = classify_vision_error(self.name, exc, self.connection_path, round((time.perf_counter() - started) * 1000))
                self.last_status_code = error.status_code
                self.last_latency_ms = error.latency_ms
                logger.warning(
                    "vision_request provider=%s model=%s base_url=%s connection_path=%s status_code=%s latency_ms=%s error_type=%s",
                    self.name,
                    self.model,
                    self.base_url,
                    self.connection_path,
                    error.status_code,
                    error.latency_ms,
                    error.error_type,
                )
                if attempt < self.max_retries and (error.network_failure or error.status_code in {408, 500, 502, 503, 504}):
                    time.sleep(min(2, 2**attempt))
                    continue
                raise error from exc
        raise classify_vision_error(self.name, last_error or RuntimeError("request failed"), self.connection_path, self.last_latency_ms)

    def _messages(self, image_path: Path, context: dict | None) -> list[dict[str, Any]]:
        return [{"role": "system", "content": SCENE_SYSTEM_PROMPT}, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}}, {"type": "text", "text": scene_user_prompt((context or {}).get("ocr_elements", []), context)}]}]

    def analyze_scene(self, image_path: Path, context: dict | None = None, mode: str = "standard") -> dict[str, Any]:
        self._reset_scene_diagnostics()
        messages = self._messages(image_path, context)
        raw = self._request(messages)
        self.last_raw_response = raw
        try:
            payload = extract_json(raw)
        except (TypeError, ValueError):
            repaired = self._request(messages, repair_prompt=SCENE_REPAIR_PROMPT)
            self.last_repaired_response = repaired
            self.repair_used = True
            payload = extract_json(repaired)
        diagnostics: dict[str, Any] = {}
        self.normalization_applied = True
        try:
            scene = validate_json(payload, "scene", diagnostics)
        except Exception as exc:
            self._capture_scene_diagnostics(diagnostics)
            details = diagnostics.get("validationErrors") or [str(exc)]
            raise VisionProviderError("Vision validation failed: " + "; ".join(str(item) for item in details)) from exc
        self._capture_scene_diagnostics(diagnostics)
        return {"provider": self.name, "model": getattr(self, "model", None), "aiUsed": True, **scene}

    def critique_reconstruction(self, original_path: Path, reconstructed_path: Path, scene: dict) -> dict[str, Any]:
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_to_data_url(original_path)}}, {"type": "image_url", "image_url": {"url": image_to_data_url(reconstructed_path)}}, {"type": "text", "text": CRITIC_PROMPT + "\nSCENE_SUMMARY:\n" + str({"elements": len(scene.get("elements", [])), "groups": len(scene.get("groups", []))})}]}]
        raw = self._request(messages)
        try:
            payload = extract_json(raw)
        except (TypeError, ValueError):
            repaired = self._request(messages, repair_prompt=SCENE_REPAIR_PROMPT)
            payload = extract_json(repaired)
        return {"provider": self.name, "model": self.model, **validate_json(payload, "critic")}

    def _reset_scene_diagnostics(self) -> None:
        self.last_raw_response: str | None = None
        self.last_repaired_response: str | None = None
        self.last_validation_errors: list[str] = []
        self.last_normalized_payload: dict[str, Any] | None = None
        self.normalization_warnings: list[str] = []
        self.dropped_elements = 0
        self.repair_used = False
        self.normalization_applied = False

    def _capture_scene_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        self.last_validation_errors = list(diagnostics.get("validationErrors") or [])
        self.last_normalized_payload = diagnostics.get("normalizedPayload")
        self.normalization_warnings = list(diagnostics.get("normalizationWarnings") or [])
        self.dropped_elements = int(diagnostics.get("droppedElements") or 0)

    def vision_debug(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": getattr(self, "model", None),
            "rawResponseAvailable": bool(self.last_raw_response),
            "repairUsed": self.repair_used,
            "normalizationApplied": self.normalization_applied,
            "validationErrors": self.last_validation_errors,
            "droppedElements": self.dropped_elements,
            "normalizationWarnings": self.normalization_warnings,
        }

    def test_connection(self, image_path: Path | None = None) -> dict[str, Any]:
        owned_path = image_path is None
        test_path = image_path
        try:
            if test_path is None:
                test_path = create_test_image()
            messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_to_data_url(test_path)}}, {"type": "text", "text": "Reply exactly with: VISION_OK"}]}]
            answer = self._request(messages)
            success = "VISION_OK" in answer.upper()
            return {
                "success": success,
                "provider": self.name,
                "model": self.model,
                "message": "连接成功" if success else "模型未返回 VISION_OK",
                "proxyUrl": self.proxy_url,
                "connectionPath": self.connection_path,
                "latencyMs": self.last_latency_ms,
                "statusCode": self.last_status_code or 200,
                "stage": "request",
                "errorType": None if success else "unexpected_response",
                "errorCode": None if success else "vision_ok_missing",
            }
        except RuntimeError as exc:
            if str(exc) == "本地测试图片创建失败":
                raise VisionProviderError(str(exc)) from exc
            raise
        finally:
            if owned_path:
                cleanup_test_image(test_path)


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    return int(value) if isinstance(value, int) else None


def classify_vision_error(provider: str, exc: Exception, connection_path: str | None = None, latency_ms: int | None = None) -> VisionProviderError:
    if isinstance(exc, VisionProviderError):
        if not exc.connection_path:
            exc.connection_path = connection_path
        if exc.latency_ms is None:
            exc.latency_ms = latency_ms
        return exc
    status = _status_code(exc)
    error_code = _error_code(exc)
    detail = (" ".join(_exception_text(exc)) + " " + str(error_code or "") + " " + str(getattr(exc, "body", ""))).lower()
    if status == 401:
        return VisionProviderError("Qwen API Key 无效", status, stage="auth", error_type="authentication_error", error_code=error_code or "invalid_api_key", connection_path=connection_path, latency_ms=latency_ms)
    if status == 403:
        return VisionProviderError("Qwen 无访问权限，请检查区域或模型权限", status, stage="auth", error_type="permission_denied", error_code=error_code or "permission_denied", connection_path=connection_path, latency_ms=latency_ms)
    if status == 404:
        return VisionProviderError("Qwen 模型不可用或端点错误", status, stage="model", error_type="model_not_found", error_code=error_code or "model_not_found", connection_path=connection_path, latency_ms=latency_ms)
    if status == 429:
        quota = "insufficient_quota" in detail or "quota" in detail or "余额" in detail
        return VisionProviderError("当前 API Key 额度不足" if quota else "Qwen 请求过于频繁", status, stage="request", error_type="insufficient_quota" if quota else "rate_limit", error_code=error_code or ("insufficient_quota" if quota else "rate_limit"), connection_path=connection_path, latency_ms=latency_ms)
    if status == 400:
        return VisionProviderError("Qwen 请求格式或模型参数错误", status, stage="request", error_type="invalid_request", error_code=error_code or "bad_request", connection_path=connection_path, latency_ms=latency_ms)
    if status is not None and status >= 500:
        return VisionProviderError("DashScope 服务暂时不可用", status, stage="request", error_type="server_error", error_code=error_code or "dashscope_server_error", connection_path=connection_path, latency_ms=latency_ms)
    if "ssl" in detail or "tls" in detail or "certificate" in detail:
        return VisionProviderError("DashScope TLS 连接失败", None, stage="network", error_type="tls_error", error_code="tls_error", connection_path=connection_path, latency_ms=latency_ms)
    if "timeout" in detail or "timed out" in detail or "deadline" in detail:
        return VisionProviderError("直连 DashScope 超时", None, stage="network", error_type="connection_timeout", error_code="connection_timeout", connection_path=connection_path, latency_ms=latency_ms)
    if "connect" in detail or "network" in detail or "connection" in detail:
        return VisionProviderError("无法连接 DashScope", None, stage="network", error_type="connect_error", error_code="connect_error", connection_path=connection_path, latency_ms=latency_ms)
    return VisionProviderError(f"{provider} API 请求失败", status, stage="request", error_type="request_error", error_code=error_code, connection_path=connection_path, latency_ms=latency_ms)


def classify_http_status(status: int, body: str = "") -> tuple[str, str, str, str]:
    detail = body.lower()
    if status == 401:
        return "auth", "authentication_error", "invalid_api_key", "Qwen API Key 无效"
    if status == 403:
        return "auth", "permission_denied", "permission_denied", "Qwen 无访问权限，请检查区域或模型权限"
    if status == 404:
        return "model", "model_not_found", "model_not_found", "Qwen 模型不可用或端点错误"
    if status == 429:
        quota = "insufficient_quota" in detail or "quota" in detail or "余额" in detail
        return "request", "insufficient_quota" if quota else "rate_limit", "insufficient_quota" if quota else "rate_limit", "当前 API Key 额度不足" if quota else "Qwen 请求过于频繁"
    if status == 400:
        return "request", "invalid_request", "bad_request", "Qwen 请求格式或模型参数错误"
    if status >= 500:
        return "request", "server_error", "dashscope_server_error", "DashScope 服务暂时不可用"
    return "request", "http_error", f"http_{status}", f"DashScope 返回 HTTP {status}"


def _error_code(exc: Exception) -> str | None:
    code = getattr(exc, "code", None)
    if code:
        return str(code)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        nested = body.get("error")
        value = nested.get("code") if isinstance(nested, dict) else body.get("code")
        return str(value) if value else None
    return None


def _exception_text(exc: Exception) -> list[str]:
    parts: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.extend((type(current).__name__, str(current)))
        current = current.__cause__ or current.__context__
    return parts


def _connection_path(proxy_url: object) -> str:
    if isinstance(proxy_url, str) and proxy_url:
        from .proxy import display_proxy_url

        return f"PROXY {display_proxy_url(proxy_url)}"
    return "DIRECT"


__all__ = ["NETWORK_ERROR_TYPES", "OpenAICompatibleVisionProvider", "VisionProviderError", "classify_http_status", "classify_vision_error", "image_to_data_url"]
