from __future__ import annotations

import logging
import socket
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.services.settings.runtime_settings import (
    QWEN_BASE_URL,
    QWEN_MODEL,
    VisionSettings,
    load_vision_settings,
    normalize_qwen_base_url,
)

from .openai_compatible_provider import (
    OpenAICompatibleVisionProvider,
    VisionProviderError,
    classify_http_status,
    classify_vision_error,
    image_to_data_url,
)
from .proxy import display_proxy_url, proxy_tcp_test, resolve_proxy
from .prompts import RECONSTRUCTION_PLAN_PROMPT, SCENE_REPAIR_PROMPT, reconstruction_plan_user_prompt
from .schemas import extract_json, validate_json


logger = logging.getLogger(__name__)


def _plan_covers_text(plan: dict[str, Any] | None, context: dict | None) -> bool:
    modules = (plan or {}).get("modules") or []
    if not modules:
        return False
    width, height = float((context or {}).get("width") or 0), float((context or {}).get("height") or 0)
    if width <= 0 or height <= 0:
        return True
    centers = []
    for item in (context or {}).get("candidate_elements", []):
        if item.get("type") != "text" or not str(item.get("text") or "").strip():
            continue
        box = item.get("bbox") or {}
        try:
            centers.append(((float(box["left"]) + float(box["width"]) / 2) / width,
                            (float(box["top"]) + float(box["height"]) / 2) / height))
        except (KeyError, TypeError, ValueError):
            continue
    if len(centers) < 3:
        return True
    covered = 0
    for x, y in centers:
        if any(_point_inside_plan_box(x, y, module.get("bbox") or {}) for module in modules):
            covered += 1
    return covered / len(centers) >= 0.65


def _point_inside_plan_box(x: float, y: float, box: dict[str, Any]) -> bool:
    try:
        left, top = float(box["left"]), float(box["top"])
        return left <= x <= left + float(box["width"]) and top <= y <= top + float(box["height"])
    except (KeyError, TypeError, ValueError):
        return False


class QwenProvider(OpenAICompatibleVisionProvider):
    name = "qwen"

    def plan_reconstruction(self, image_path: Path, context: dict | None = None) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": RECONSTRUCTION_PLAN_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                {"type": "text", "text": reconstruction_plan_user_prompt(context)},
            ]},
        ]
        raw = self._request(messages)
        try:
            payload = extract_json(raw)
        except (TypeError, ValueError):
            repair_messages = [*messages, {"role": "assistant", "content": raw}, {"role": "user", "content": SCENE_REPAIR_PROMPT}]
            payload = extract_json(self._request(repair_messages))
        if "reconstructionPlan" in payload:
            payload = payload["reconstructionPlan"]
        plan = validate_json({"reconstructionPlan": payload}, "scene")["reconstructionPlan"]
        return plan

    def analyze_scene(self, image_path: Path, context: dict | None = None, mode: str = "standard") -> dict[str, Any]:
        scene: dict[str, Any] | None = None
        try:
            scene = super().analyze_scene(image_path, context, mode)
        except Exception as exc:
            logger.warning("qwen_scene_analysis_failed error_type=%s", type(exc).__name__)
            if not (context or {}).get("candidate_elements"):
                raise
            plan = self.plan_reconstruction(image_path, context)
            if not plan.get("modules"):
                raise
            logger.warning("qwen_scene_analysis_failed_using_page_plan")
            scene = {"provider": self.name, "model": self.model, "aiUsed": True, "page": {}, "regions": [], "elements": [], "groups": [], "relations": [], "repeatedComponents": [], "layers": [], "confidence": 0.5, "reconstructionPlan": plan}
        if not _plan_covers_text(scene.get("reconstructionPlan"), context) and (context or {}).get("candidate_elements"):
            try:
                replacement = self.plan_reconstruction(image_path, context)
                if _plan_covers_text(replacement, context):
                    scene["reconstructionPlan"] = replacement
            except Exception as exc:
                logger.warning("qwen_reconstruction_plan_failed error_type=%s", type(exc).__name__)
        if mode == "high" and not _plan_covers_text(scene.get("reconstructionPlan"), context):
            raise VisionProviderError("Qwen reconstruction plan does not cover the page text")
        return scene


    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None, vision_settings: VisionSettings | None = None) -> None:
        runtime = vision_settings or load_vision_settings()
        config = runtime.providers["qwen"]
        self.proxy_mode = (runtime.proxy_mode or "system").lower()
        self.manual_proxy = runtime.manual_proxy
        self.connection_options = _connection_options(runtime)
        self.connection_attempts: list[dict[str, Any]] = []
        first_proxy, _ = self.connection_options[0]
        super().__init__(
            provider_name="qwen",
            api_key=api_key if api_key is not None else config.api_key,
            base_url=normalize_qwen_base_url(base_url if base_url is not None else config.base_url, api_key if api_key is not None else config.api_key),
            model=QWEN_MODEL,
            timeout=runtime.timeout,
            max_retries=0,
            proxy_url=first_proxy,
        )

    def _request(self, messages: list[dict[str, Any]], *, repair_prompt: str | None = None) -> str:
        self.connection_attempts = []
        last_error: VisionProviderError | None = None
        for index, (proxy_url, path) in enumerate(self.connection_options):
            self._configure_client(proxy_url)
            started = time.perf_counter()
            try:
                answer = super()._request(messages, repair_prompt=repair_prompt)
                self.connection_attempts.append({
                    "path": path,
                    "success": True,
                    "statusCode": self.last_status_code or 200,
                    "latencyMs": self.last_latency_ms,
                })
                return answer
            except VisionProviderError as exc:
                exc.connection_path = path
                exc.latency_ms = exc.latency_ms or round((time.perf_counter() - started) * 1000)
                self.connection_attempts.append({
                    "path": path,
                    "success": False,
                    "statusCode": exc.status_code,
                    "latencyMs": exc.latency_ms,
                    "errorType": exc.error_type,
                    "errorCode": exc.error_code,
                })
                last_error = exc
                if index + 1 < len(self.connection_options) and exc.network_failure:
                    continue
                raise
        raise last_error or VisionProviderError("Qwen API 请求失败")


def create_qwen_provider(vision_settings: VisionSettings | None = None) -> QwenProvider:
    return QwenProvider(vision_settings=vision_settings)


def test_qwen_connectivity(vision_settings: VisionSettings | None = None) -> dict[str, Any]:
    settings = vision_settings or load_vision_settings()
    config = settings.providers["qwen"]
    base_url = normalize_qwen_base_url(config.base_url, config.api_key)
    model = QWEN_MODEL
    proxy = resolve_proxy(settings.proxy_mode, settings.manual_proxy)
    proxy_tcp = proxy_tcp_test(proxy.url)
    parsed = urlparse(base_url)
    host = parsed.hostname or "dashscope.aliyuncs.com"
    port = parsed.port or 443
    diagnostics: dict[str, Any] = {
        "tcp": False,
        "httpsReachable": False,
        "auth": False,
        "modelReachable": False,
        "visionRequest": False,
        "proxyTcp": proxy_tcp,
        "qwenBaseUrl": {"status": "SKIPPED", "message": "未执行"},
        "qwenApi": {"status": "SKIPPED", "message": "未执行"},
        "networkPath": "DIRECT",
        "connectionPath": "DIRECT",
        "directTried": False,
        "proxyTried": False,
        "attempts": [],
    }
    started = time.perf_counter()

    try:
        _connection_options(settings)
    except VisionProviderError as error:
        return _failure_result(error, diagnostics, model, started)

    try:
        with socket.create_connection((host, port), timeout=5):
            diagnostics["tcp"] = True
    except OSError as exc:
        diagnostics["tcpError"] = type(exc).__name__

    direct_started = time.perf_counter()
    try:
        with httpx.Client(proxy=None, timeout=10, trust_env=False, follow_redirects=True) as client:
            response = client.get(f"{base_url}/models")
        diagnostics["httpsReachable"] = response.status_code < 500
        diagnostics["qwenBaseUrl"] = {
            "status": "PASS" if response.status_code < 500 else "FAIL",
            "message": f"HTTP {response.status_code}",
            "httpStatus": response.status_code,
            "latencyMs": round((time.perf_counter() - direct_started) * 1000),
        }
    except Exception as exc:
        direct_error = classify_vision_error("qwen", exc, "DIRECT", round((time.perf_counter() - direct_started) * 1000))
        diagnostics["qwenBaseUrl"] = {
            "status": "FAIL",
            "message": str(direct_error),
            "errorType": direct_error.error_type,
            "latencyMs": direct_error.latency_ms,
        }

    if not settings.enabled or settings.selected_provider == "local" or not config.api_key:
        error = VisionProviderError("Qwen API Key 未配置或 AI 已关闭", stage="auth", error_type="not_configured", error_code="api_key_missing")
        return _failure_result(error, diagnostics, model, started)

    auth_result = _test_authenticated_models(settings, base_url, config.api_key, diagnostics)
    if isinstance(auth_result, VisionProviderError):
        return _failure_result(auth_result, diagnostics, model, started)
    diagnostics["auth"] = True
    diagnostics["modelReachable"] = True

    try:
        provider = QwenProvider(vision_settings=settings)
        result = provider.test_connection()
        diagnostics["attempts"].extend(provider.connection_attempts)
        diagnostics["directTried"] = diagnostics["directTried"] or any(item["path"] == "DIRECT" for item in provider.connection_attempts)
        diagnostics["proxyTried"] = diagnostics["proxyTried"] or any(item["path"].startswith("PROXY") for item in provider.connection_attempts)
        final_path = result.get("connectionPath") or (provider.connection_attempts[-1]["path"] if provider.connection_attempts else "DIRECT")
        diagnostics["networkPath"] = final_path
        diagnostics["connectionPath"] = final_path
        diagnostics["visionRequest"] = bool(result.get("success"))
        diagnostics["qwenApi"] = {
            "status": "PASS" if result.get("success") else "FAIL",
            "message": result.get("message", ""),
            "httpStatus": result.get("statusCode"),
            "latencyMs": result.get("latencyMs"),
        }
        total_latency = round((time.perf_counter() - started) * 1000)
        logger.info(
            "qwen_connectivity provider=qwen model=%s base_url=%s proxy_mode=%s connection_path=%s status_code=%s latency_ms=%s error_type=none",
            model,
            base_url,
            settings.proxy_mode,
            final_path,
            result.get("statusCode", 200),
            total_latency,
        )
        return {
            "success": bool(result.get("success")),
            "provider": "qwen",
            "model": model,
            "stage": "request",
            "statusCode": result.get("statusCode", 200),
            "errorType": result.get("errorType"),
            "errorCode": result.get("errorCode"),
            "message": result.get("message", "连接成功"),
            "connectionPath": final_path,
            "latencyMs": total_latency,
            "diagnostics": diagnostics,
        }
    except Exception as exc:
        error = classify_vision_error("qwen", exc, getattr(exc, "connection_path", None), getattr(exc, "latency_ms", None))
        provider_attempts = getattr(locals().get("provider"), "connection_attempts", [])
        diagnostics["attempts"].extend(provider_attempts)
        diagnostics["directTried"] = diagnostics["directTried"] or any(item["path"] == "DIRECT" for item in provider_attempts)
        diagnostics["proxyTried"] = diagnostics["proxyTried"] or any(item["path"].startswith("PROXY") for item in provider_attempts)
        return _failure_result(error, diagnostics, model, started)


def _connection_options(settings: VisionSettings) -> list[tuple[str | None, str]]:
    mode = (settings.proxy_mode or "system").lower()
    if mode in {"none", "direct", "off"}:
        return [(None, "DIRECT")]
    proxy = resolve_proxy(mode, settings.manual_proxy)
    if mode == "manual" and not proxy.url:
        raise VisionProviderError("手动代理地址未配置", stage="network", error_type="proxy_config_error", error_code="manual_proxy_missing")
    proxy_path = f"PROXY {display_proxy_url(proxy.url)}" if proxy.url else "DIRECT"
    if mode == "manual":
        return [(proxy.url, proxy_path)]
    options: list[tuple[str | None, str]] = [(None, "DIRECT")]
    if proxy.url:
        options.append((proxy.url, proxy_path))
    return options


def _test_authenticated_models(settings: VisionSettings, base_url: str, api_key: str, diagnostics: dict[str, Any]) -> bool | VisionProviderError:
    options = _connection_options(settings)
    last_error: VisionProviderError | None = None
    for index, (proxy_url, path) in enumerate(options):
        diagnostics["directTried"] = diagnostics["directTried"] or path == "DIRECT"
        diagnostics["proxyTried"] = diagnostics["proxyTried"] or path.startswith("PROXY")
        started = time.perf_counter()
        try:
            with httpx.Client(proxy=proxy_url, timeout=settings.timeout, trust_env=False, follow_redirects=True) as client:
                response = client.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"})
            latency = round((time.perf_counter() - started) * 1000)
            diagnostics["attempts"].append({"stage": "auth", "path": path, "statusCode": response.status_code, "latencyMs": latency})
            diagnostics["networkPath"] = path
            diagnostics["connectionPath"] = path
            if response.status_code == 200:
                return True
            stage, error_type, error_code, message = classify_http_status(response.status_code, response.text[:2000])
            return VisionProviderError(message, response.status_code, stage=stage, error_type=error_type, error_code=error_code, connection_path=path, latency_ms=latency)
        except Exception as exc:
            error = classify_vision_error("qwen", exc, path, round((time.perf_counter() - started) * 1000))
            diagnostics["attempts"].append({"stage": "auth", "path": path, "statusCode": error.status_code, "latencyMs": error.latency_ms, "errorType": error.error_type})
            last_error = error
            if index + 1 < len(options) and error.network_failure:
                continue
            return error
    return last_error or VisionProviderError("Qwen 认证请求失败", stage="auth", error_type="authentication_error")


def _failure_result(error: VisionProviderError, diagnostics: dict[str, Any], model: str, started: float) -> dict[str, Any]:
    diagnostics["networkPath"] = error.connection_path or diagnostics.get("networkPath") or "DIRECT"
    diagnostics["connectionPath"] = diagnostics["networkPath"]
    diagnostics["qwenApi"] = {
        "status": "FAIL",
        "message": str(error),
        "httpStatus": error.status_code,
        "errorType": error.error_type,
        "errorCode": error.error_code,
    }
    total_latency = round((time.perf_counter() - started) * 1000)
    logger.warning(
        "qwen_connectivity provider=qwen model=%s base_url=%s connection_path=%s status_code=%s latency_ms=%s error_type=%s",
        model,
        QWEN_BASE_URL,
        diagnostics["networkPath"],
        error.status_code,
        total_latency,
        error.error_type,
    )
    return {
        "success": False,
        "provider": "qwen",
        "model": model,
        "stage": error.stage,
        "statusCode": error.status_code,
        "errorType": error.error_type,
        "errorCode": error.error_code,
        "message": str(error),
        "connectionPath": diagnostics["networkPath"],
        "latencyMs": total_latency,
        "diagnostics": diagnostics,
    }


__all__ = ["QwenProvider", "VisionProviderError", "create_qwen_provider", "image_to_data_url", "test_qwen_connectivity"]
