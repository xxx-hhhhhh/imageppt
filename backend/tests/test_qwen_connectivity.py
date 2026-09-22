from __future__ import annotations

import httpx

import app.services.vision.openai_compatible_provider as compatible
import app.services.vision.qwen_provider as qwen_module
from app.services.settings.runtime_settings import ProviderSettings, VisionSettings
from app.services.vision.openai_compatible_provider import OpenAICompatibleVisionProvider, VisionProviderError, classify_vision_error
from app.services.vision.qwen_provider import QwenProvider, _connection_options


def _settings(mode: str = "system") -> VisionSettings:
    return VisionSettings(
        enabled=True,
        selected_provider="qwen",
        proxy_mode=mode,
        providers={"qwen": ProviderSettings(enabled=True, api_key="sk-test", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen3-vl-flash")},
    )


def test_none_mode_builds_direct_httpx_client_without_environment(monkeypatch):
    captured = {}

    class FakeHTTPClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def close(self):
            pass

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    provider = object.__new__(OpenAICompatibleVisionProvider)
    provider.api_key = "sk-test"
    provider.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    provider.timeout = 60
    provider._openai_type = FakeOpenAI
    monkeypatch.setattr(compatible.httpx, "Client", FakeHTTPClient)

    provider._configure_client(None)

    assert captured["proxy"] is None
    assert captured["trust_env"] is False
    assert _connection_options(_settings("none")) == [(None, "DIRECT")]


def test_auto_direct_success_does_not_try_proxy(monkeypatch):
    provider = object.__new__(QwenProvider)
    provider.connection_options = [(None, "DIRECT"), ("http://127.0.0.1:7897", "PROXY 127.0.0.1:7897")]
    provider.connection_attempts = []
    provider.last_status_code = 200
    provider.last_latency_ms = 12
    configured = []
    monkeypatch.setattr(provider, "_configure_client", lambda value: configured.append(value))
    monkeypatch.setattr(OpenAICompatibleVisionProvider, "_request", lambda self, messages, repair_prompt=None: "VISION_OK")

    assert provider._request([]) == "VISION_OK"
    assert configured == [None]
    assert provider.connection_attempts[0]["path"] == "DIRECT"


def test_auto_falls_back_to_proxy_only_after_network_failure(monkeypatch):
    provider = object.__new__(QwenProvider)
    provider.connection_options = [(None, "DIRECT"), ("http://127.0.0.1:7897", "PROXY 127.0.0.1:7897")]
    provider.connection_attempts = []
    provider.last_status_code = 200
    provider.last_latency_ms = 15
    configured = []
    responses = iter([VisionProviderError("timeout", stage="network", error_type="connection_timeout"), "VISION_OK"])
    monkeypatch.setattr(provider, "_configure_client", lambda value: configured.append(value))

    def request(self, messages, repair_prompt=None):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(OpenAICompatibleVisionProvider, "_request", request)

    assert provider._request([]) == "VISION_OK"
    assert configured == [None, "http://127.0.0.1:7897"]
    assert provider.connection_attempts[0]["errorType"] == "connection_timeout"
    assert provider.connection_attempts[1]["success"] is True


class _HTTPError(Exception):
    def __init__(self, status_code: int, body=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.body = body or {}


def test_qwen_http_errors_are_classified():
    assert classify_vision_error("qwen", _HTTPError(401)).error_type == "authentication_error"
    assert classify_vision_error("qwen", _HTTPError(403)).error_type == "permission_denied"
    quota = classify_vision_error("qwen", _HTTPError(429, {"error": {"code": "insufficient_quota", "message": "quota exhausted"}}))
    assert quota.error_type == "insufficient_quota"
    assert quota.error_code == "insufficient_quota"
    timeout = classify_vision_error("qwen", httpx.ReadTimeout("timed out"))
    assert timeout.error_type == "connection_timeout"


def test_public_test_endpoint_returns_detailed_failure(monkeypatch):
    import app.main as main_module
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main_module.VisionRouter, "test_connection", lambda self: {
        "success": False,
        "provider": "qwen",
        "model": "qwen3-vl-flash",
        "stage": "request",
        "statusCode": 429,
        "errorType": "insufficient_quota",
        "errorCode": "insufficient_quota",
        "message": "当前 API Key 额度不足",
        "connectionPath": "DIRECT",
        "latencyMs": 123,
        "diagnostics": {},
    })
    monkeypatch.setattr(main_module, "PUBLIC_SHARED_MODE", True)

    payload = TestClient(main_module.app).post("/api/settings/vision/test").json()

    assert payload["statusCode"] == 429
    assert payload["errorType"] == "insufficient_quota"
    assert payload["connectionPath"] == "DIRECT"
